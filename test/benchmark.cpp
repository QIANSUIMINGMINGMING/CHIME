#include "Timer.h"
// #include "Tree.h"
#include <city.h>
#include <libcgroup.h>
#include <numa.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Common.h"
#include "Tree.h"
#include "gaussian_generator.h"
#include "uniform.h"
#include "uniform_generator.h"
#include "zipf.h"

int kMaxThread = MAX_APP_THREAD;
// std::thread th[MAX_APP_THREAD];

uint64_t xmd_latency[MAX_APP_THREAD][LATENCY_WINDOWS]{0};
uint64_t latency_th_all[LATENCY_WINDOWS]{0};
uint64_t tp[MAX_APP_THREAD][8];

// uint64_t total_tp[MAX_APP_THREAD];
uint64_t total_time[MAX_APP_THREAD];

int64_t kCPUPercentage = 10;
std::mutex mtx;
std::condition_variable cv;
uint32_t kReadRatio;
uint32_t kInsertRatio;  // hybrid read ratio
uint32_t kUpdateRatio;
uint32_t kDeleteRatio;
uint32_t kRangeRatio;
int kThreadCount;
int totalThreadCount;
int memThreadCount;
int kNodeCount;
int CNodeCount;
std::vector<Key> sharding;
uint64_t cache_mb;
// uint64_t kKeySpace = 128 * define::MB; // 268M KVs; 107M
uint64_t kKeySpace;
uint64_t threadKSpace;
uint64_t partition_space;
uint64_t left_bound = 0;
uint64_t right_bound = 0;
uint64_t op_num = 0;  // Total operation num

uint64_t per_node_op_num = 0;
uint64_t per_node_warmup_num = 0;
uint64_t per_thread_op_num = 0;
uint64_t per_thread_warmup_num = 0;

uint64_t *bulk_array = nullptr;
uint64_t bulk_load_num = 0;
uint64_t warmup_num = 0;  // 10M for warmup
int node_id = 0;
double zipfian;
uint64_t *insert_array = nullptr;
uint64_t insert_array_size = 0;
int tree_index = 0;
int check_correctness = 0;
int time_based = 1;
int early_stop = 1;

struct zipf_gen_state state;
// int uniform_workload = 0;
uniform_key_generator_t *uniform_generator = nullptr;
gaussian_key_generator_t *gaussian_generator = nullptr;
enum WorkLoadType {
  uniform,      // 0
  zipf,         // 1
  gaussian_01,  // 2
  gaussian_001  // 3
};
WorkLoadType workload_type;

std::vector<uint64_t> throughput_vec;
std::vector<uint64_t> straggler_throughput_vec;

// std::array<uint8_t, 8> *workload_array = nullptr;
// std::array<uint8_t, 8> *warmup_array = nullptr;
uint64_t *workload_array = nullptr;
uint64_t *warmup_array = nullptr;

enum op_type : uint8_t { Insert, Update, Lookup, Delete, Range };
uint64_t op_mask = (1ULL << 56) - 1;

Tree *tree;
DSM *dsm;

inline uint64_t to_key(uint64_t k) {
  uint64_t hash = (CityHash64((char *)&k, sizeof(k)) + 1) % kKeySpace;
  // hash cant be 0, 0 rehash
  if (hash == 0) {
    hash = (CityHash64((char *)&hash, sizeof(k)) + 1) % kKeySpace;
  }

  return hash;
}

std::atomic<int64_t> warmup_cnt{0};
std::atomic<uint64_t> worker{0};
std::atomic<uint64_t> execute_op{0};
std::atomic_bool ready{false};
std::atomic_bool one_finish{false};
std::atomic_bool ready_to_report{false};

void reset_all_params() {
  warmup_cnt.store(0);
  worker.store(0);
  ready.store(false);
  one_finish.store(false);
  ready_to_report.store(false);
}

void init_key_generator() {
  if (workload_type == WorkLoadType::uniform) {
    uniform_generator = new uniform_key_generator_t(kKeySpace);
  } else if (workload_type == WorkLoadType::zipf) {
    mehcached_zipf_init(&state, kKeySpace, zipfian,
                        (rdtsc() & (0x0000ffffffffffffull)) ^ node_id);
  } else if (workload_type == WorkLoadType::gaussian_01) {
    gaussian_generator =
        new gaussian_key_generator_t(0.4 * kKeySpace, 0.1 * kKeySpace);
  } else if (workload_type == WorkLoadType::gaussian_001) {
    gaussian_generator =
        new gaussian_key_generator_t(0.4 * kKeySpace, 0.04 * kKeySpace);
  } else {
    assert(false);
  }
}

static int key_id = 0;

uint64_t generate_key() {
  // static int counter = 0;
  // Key key;
  uint64_t key;
  uint64_t dis;
  while (true) {
    if (workload_type == WorkLoadType::uniform) {
      dis = uniform_generator->next_id();
      key = to_key(dis);
    } else if (workload_type == WorkLoadType::zipf) {
      dis = mehcached_zipf_next(&state);
      key = to_key(dis);
    } else if (workload_type == WorkLoadType::gaussian_01 ||
               workload_type == WorkLoadType::gaussian_001) {
      key = gaussian_generator->next_id();
    } else {
      assert(false);
    }
    if (key >= 0 && key < kKeySpace) {
      if (key_id < 10) {
        std::cout << key << std::endl;
      }
      key_id++;
      break;
    }
  }
  return key;
}

void generate_workload() {
  // Generate workload for bulk_loading
  uint64_t *space_array = new uint64_t[kKeySpace];
  for (uint64_t i = 0; i < kKeySpace; ++i) {
    space_array[i] = i;
  }
  bulk_array = new uint64_t[bulk_load_num];
  uint64_t thread_warmup_insert_num =
      (kInsertRatio / 100.0) * (warmup_num / totalThreadCount);
  uint64_t warmup_insert_key_num = thread_warmup_insert_num * kThreadCount;

  uint64_t thread_workload_insert_num =
      (kInsertRatio / 100.0) * (op_num / totalThreadCount);
  uint64_t workload_insert_key_num = thread_warmup_insert_num * kThreadCount;
  uint64_t *insert_array = nullptr;

  partition_space = kKeySpace;
  left_bound = 0;
  right_bound = kKeySpace;
  std::mt19937 gen(0xc70f6907UL);
  std::shuffle(&space_array[0], &space_array[kKeySpace - 1], gen);
  memcpy(&bulk_array[0], &space_array[0], sizeof(uint64_t) * bulk_load_num);

  uint64_t regular_node_insert_num =
      static_cast<uint64_t>(thread_warmup_insert_num * kThreadCount) +
      static_cast<uint64_t>(thread_workload_insert_num * kThreadCount);
  insert_array =
      space_array + bulk_load_num + regular_node_insert_num * node_id;
  // assert((bulk_load_num + regular_node_insert_num * node_id +
  //         warmup_insert_key_num + workload_insert_key_num) <= kKeySpace);
  std::cout << "First key of bulkloading = " << bulk_array[0] << std::endl;
  std::cout << "Last key of bulkloading = " << bulk_array[bulk_load_num - 1]
            << std::endl;

  init_key_generator();

  // srand((unsigned)time(NULL));
  // UniformRandom rng(rand());
  UniformRandom rng(rdtsc() ^ node_id);
  uint32_t random_num;
  auto insertmark = kReadRatio + kInsertRatio;
  auto updatemark = insertmark + kUpdateRatio;
  auto deletemark = updatemark + kDeleteRatio;
  auto rangemark = deletemark + kRangeRatio;
  assert(rangemark == 100);

  // auto updatemark = insertmark + kUpdateRatio;
  std::cout << "node warmup insert num = " << warmup_insert_key_num
            << std::endl;
  warmup_array = new uint64_t[warmup_num];
  std::cout << "kReadRatio =" << kReadRatio << std::endl;
  std::cout << "insertmark =" << insertmark << std::endl;
  std::cout << "updatemark =" << updatemark << std::endl;
  std::cout << "deletemark =" << deletemark << std::endl;
  std::cout << "rangemark =" << rangemark << std::endl;

  uint64_t i = 0;
  uint64_t insert_counter = 0;
  per_node_warmup_num = 0;

  if (kInsertRatio == 100) {
    // "Load workload" => need to guarantee all keys are new key
    assert(workload_type == WorkLoadType::uniform);
    while (i < warmup_insert_key_num) {
      uint64_t key = (insert_array[insert_counter] |
                      (static_cast<uint64_t>(op_type::Insert) << 56));
      warmup_array[i] = key;
      ++insert_counter;
      ++i;
    }
    per_node_warmup_num = insert_counter;
    assert(insert_counter <= warmup_insert_key_num);
  } else {
    per_node_warmup_num = (warmup_num / totalThreadCount) * kThreadCount;
    while (i < per_node_warmup_num) {
      random_num = rng.next_uint32() % 100;
      uint64_t key_value = generate_key();

      // Convert Key to uint64_t for bitwise operations

      // Perform bitwise operations on the uint64_t value
      if (random_num < kReadRatio) {
        key_value = key_value | (static_cast<uint64_t>(op_type::Lookup) << 56);
      } else if (random_num < insertmark) {
        key_value = key_value | (static_cast<uint64_t>(op_type::Insert) << 56);
      } else if (random_num < updatemark) {
        key_value = key_value | (static_cast<uint64_t>(op_type::Update) << 56);
      } else if (random_num < deletemark) {
        key_value = key_value | (static_cast<uint64_t>(op_type::Delete) << 56);
      } else {
        key_value = key_value | (static_cast<uint64_t>(op_type::Range) << 56);
      }

      // Store the uint64_t value directly in warmup_array
      warmup_array[i] = key_value;
      ++i;
    }
  }
  std::cout << "node warmup num: " << per_node_warmup_num << std::endl;

  if (per_node_warmup_num > 0) {
    std::shuffle(&warmup_array[0], &warmup_array[per_node_warmup_num - 1], gen);
  }
  std::cout << "Finish warmup workload generation" << std::endl;

  workload_array = new uint64_t[op_num];
  i = 0;
  insert_array = insert_array + insert_counter;
  insert_counter = 0;
  std::unordered_map<uint64_t, uint64_t> key_count;

  per_node_op_num = 0;

  if (kInsertRatio == 100) {
    assert(workload_type == WorkLoadType::uniform);
    while (i < workload_insert_key_num) {
      uint64_t key = (insert_array[insert_counter] |
                      (static_cast<uint64_t>(op_type::Insert) << 56));
      workload_array[i] = key;
      ++insert_counter;
      ++i;
    }
    per_node_op_num = insert_counter;
    assert(insert_counter <= workload_insert_key_num);
  } else {
    per_node_op_num = (op_num / totalThreadCount) * kThreadCount;
    while (i < per_node_op_num) {
      random_num = rng.next_uint32() % 100;
      uint64_t key_value = generate_key();

      if (key_value)
        if (random_num < kReadRatio) {
          key_value =
              key_value | (static_cast<uint64_t>(op_type::Lookup) << 56);
        } else if (random_num < insertmark) {
          key_value =
              key_value | (static_cast<uint64_t>(op_type::Insert) << 56);
        } else if (random_num < updatemark) {
          key_value =
              key_value | (static_cast<uint64_t>(op_type::Update) << 56);
        } else if (random_num < deletemark) {
          key_value =
              key_value | (static_cast<uint64_t>(op_type::Delete) << 56);
        } else {
          key_value = key_value | (static_cast<uint64_t>(op_type::Range) << 56);
        }
      workload_array[i] = key_value;
      ++i;
    }
  }
  std::cout << "node op num: " << per_node_op_num << std::endl;
  // std::shuffle(&workload_array[0], &workload_array[node_op_num - 1], gen);
  per_thread_op_num = per_node_op_num / kThreadCount;
  per_thread_warmup_num = per_node_warmup_num / kThreadCount;
  std::cout << "thread op num: " << per_thread_op_num;
  std::cout << "and thread warm num: " << per_thread_warmup_num << std::endl;

  // std::vector<std::pair<uint64_t, uint64_t>> keyValuePairs;
  // for (const auto &entry : key_count) {
  //   keyValuePairs.push_back(entry);
  // }
  // std::sort(keyValuePairs.begin(), keyValuePairs.end(),
  //           [](const auto &a, const auto &b) { return a.second > b.second;
  //           });
  // for (int i = 0; i < 20; ++i) {
  //   std::cout << i << " key: " << keyValuePairs[i].first
  //             << " counter: " << keyValuePairs[i].second << std::endl;
  // }
  delete[] space_array;
  std::cout << "Finish all workload generation" << std::endl;
}

#define LOADER_NUM 8  // [CONFIG] 8
std::default_random_engine e;
std::uniform_int_distribution<Value> randval(define::kValueMin,
                                             define::kValueMax);

struct partition_info {
  uint64_t *array;
  uint64_t num;
  int id;
};

void bulk_load() {
  // uint64_t cluster_num = my_dsm->getClusterSize();
  uint32_t node_id = dsm->getMyNodeID();
  uint32_t compute_num = dsm->getComputeNum();
  if (node_id >= compute_num) {
    return;
  }
  // std::cout << "Smart real leaf size = " << sizeof(smart::Leaf) <<
  // std::endl;

  partition_info *all_partition = new partition_info[LOADER_NUM];
  uint64_t each_partition = bulk_load_num / (LOADER_NUM * CNodeCount);

  for (uint64_t i = 0; i < LOADER_NUM; ++i) {
    all_partition[i].id = i + node_id * LOADER_NUM;
    all_partition[i].array =
        bulk_array + (all_partition[i].id * each_partition);
    all_partition[i].num = each_partition;
  }

  if (node_id == (compute_num - 1)) {
    all_partition[LOADER_NUM - 1].num =
        bulk_load_num - (each_partition * (LOADER_NUM * compute_num - 1));
  }

  auto bulk_thread = [&](void *bulk_info) {
    auto my_parition = reinterpret_cast<partition_info *>(bulk_info);
    // bindCore((my_parition->id % bulk_threads) * 2);
    dsm->registerThread();
    auto num = my_parition->num;
    auto array = my_parition->array;

    for (uint64_t i = 0; i < num; ++i) {
      Key smart_k = int2key(array[i]);
      // std::cout << i << " start insert key-------------- " << array[i]
      //           << std::endl;
      tree->insert(smart_k, randval(e));
      // std::cout << i << " finish insert key------------- " << array[i]
      //           << std::endl;
      // std::cout << std::endl;
      if ((i + 1) % 1000000 == 0) {
        std::cout << "Thread " << my_parition->id << " finishes insert " << i
                  << " keys" << std::endl;
      }
    }
  };

  std::thread loader_th[LOADER_NUM];
  for (uint64_t i = 0; i < LOADER_NUM; i++) {
    loader_th[i] =
        std::thread(bulk_thread, reinterpret_cast<void *>(all_partition + i));
  }

  for (uint64_t i = 0; i < LOADER_NUM; i++) {
    loader_th[i].join();
  }
}

int kCoroCnt = 8;

class RequsetGenBench : public RequstGen {
 public:
  RequsetGenBench(DSM *dsm, Request *req, int req_num, int coro_id,
                  int coro_cnt)
      : dsm(dsm),
        req(req),
        req_num(req_num),
        coro_id(coro_id),
        coro_cnt(coro_cnt) {
    local_thread_id = dsm->getMyThreadID();
    cur = coro_id;
    epoch_id = 0;
    extra_k = MAX_KEY_SPACE_SIZE +
              kThreadCount * kCoroCnt * dsm->getMyNodeID() +
              local_thread_id * kCoroCnt + coro_id;
    flag = false;
  }

  Request next() override {
    cur = (cur + coro_cnt) % req_num;
    if (req[cur].req_type == INSERT) {
      if (cur + coro_cnt >= req_num) {
        // need_stop = true;
        ++epoch_id;
        flag = true;
      }
      if (flag) {
        req[cur].k = int2key(extra_k);
        extra_k += kThreadCount * kCoroCnt * dsm->getClusterSize();
      }
    }
    tp[local_thread_id][coro_id]++;
    req[cur].v = randval(e);  // make value different per-epoch
    return req[cur];
  }

 private:
  DSM *dsm;
  Request *req;
  int req_num;
  int coro_id;
  int coro_cnt;
  int local_thread_id;
  int cur;
  uint8_t epoch_id;
  uint64_t extra_k;
  bool flag;
};

RequstGen *gen_func(DSM *dsm, Request *req, int req_num, int coro_id,
                    int coro_cnt) {
  return new RequsetGenBench(dsm, req, req_num, coro_id, coro_cnt);
}

void work_func(Tree *tree, const Request &r, CoroPull *sink) {
  if (r.req_type == SEARCH) {
    Value v;
    tree->search(r.k, v, sink);
  } else if (r.req_type == INSERT) {
    tree->insert(r.k, r.v, sink);
  } else if (r.req_type == UPDATE) {
    tree->update(r.k, r.v, sink);
  } else {
    std::map<Key, Value> ret;
    tree->range_query(r.k, r.k + r.range_size, ret);
  }
}

void thread_run(int id) {
  // Interleave the thread binding
  // bindCore(id);
  // numa_set_localalloc();
  //  std::cout << "Before register the thread" << std::endl;
  dsm->registerThread();
  tp[id][0] = 0;
  // total_tp[id] = 0;
  total_time[id] = 0;
  uint64_t my_id = kMaxThread * node_id + id;
  worker.fetch_add(1);
  printf("I am %lu\n", my_id);

  // auto idx = cur_run % rpc_rate_vec.size();
  // cachepush::decision.clear();
  // cachepush::decision.set_total_num(total_num[idx]);
  // Every thread set its own warmup/workload range
  uint64_t *thread_workload_array = workload_array + id * per_thread_op_num;
  uint64_t *thread_warmup_array = warmup_array + id * per_thread_warmup_num;
  // uint64_t *thread_workload_array = new uint64_t[thread_op_num];
  // uint64_t *thread_warmup_array = new uint64_t[thread_warmup_num];
  // memcpy(thread_workload_array, thread_workload_array_in_global,
  //        sizeof(uint64_t) * thread_op_num);
  // memcpy(thread_warmup_array, thread_warmup_array_in_global,
  //        sizeof(uint64_t) * thread_warmup_num);
  size_t counter = 0;
  size_t success_counter = 0;
  uint32_t scan_num = 50;

  int pre_counter = 0;
  while (counter < per_thread_warmup_num) {
    // if (counter - pre_counter > 1000) {
    //   std::cout << "warm counter: " << counter << std::endl;
    //   pre_counter = counter;

    // void work_func(Tree * tree, const Request &r, CoroPull *sink) {
    //   if (r.req_type == SEARCH) {
    //     Value v;
    //     tree->search(r.k, v, sink);
    //   } else if (r.req_type == INSERT) {
    //     tree->insert(r.k, r.v, sink);
    //   } else if (r.req_type == UPDATE) {
    //     tree->update(r.k, r.v, sink);
    //   } else {
    //     std::map<Key, Value> ret;
    //     tree->range_query(r.k, r.k + r.range_size, ret);
    //   }
    // }

    // }
    uint64_t key_value = thread_warmup_array[counter];
    op_type cur_op = static_cast<op_type>(key_value >> 56);
    key_value = key_value & op_mask;
    Key key = int2key(key_value);
    switch (cur_op) {
      case op_type::Lookup: {
        Value v;
        auto flag = tree->search(key, v, nullptr);
        if (flag) ++success_counter;
      } break;

      case op_type::Insert: {
        Value v = rand() % 1000000;
        tree->insert(key, v);
      } break;

      case op_type::Update: {
        Value v = rand() % 1000000;
        tree->update(key, v);
      } break;

      case op_type::Range: {
        std::map<Key, Value> ret;
        tree->range_query(key, key + scan_num, ret);
      } break;

      default:
        std::cout << "OP Type NOT MATCH!" << std::endl;
    }
    ++counter;
  }
  //}

  warmup_cnt.fetch_add(1);
  if (id == 0) {
    std::cout << "Thread_op_num = " << per_thread_op_num << std::endl;
    while (warmup_cnt.load() != kThreadCount);
    // delete[] warmup_array;
    // if (cur_run == (run_times - 1)) {
    //   delete[] warmup_array;
    //   delete[] workload_array;
    // }
    printf("node %d finish warmup\n", dsm->getMyNodeID());

    dsm->clear_rdma_statistic();
    dsm->barrier(std::string("warm_finish"), CNodeCount);
    ready.store(true);
    warmup_cnt.store(0);
  }

  // Sync to the main thread
  while (!ready_to_report.load());

  // std::cout << "My thread ID = " << dsm->getMyThreadID() << std::endl;
  // std::cout << "Thread op num = " << thread_op_num << std::endl;

  // Start the real execution of the workload
  counter = 0;
  pre_counter = 0;
  success_counter = 0;
  auto start = std::chrono::high_resolution_clock::now();
  Timer thread_timer;
  while (counter < per_thread_op_num && !one_finish.load()) {
    // if (counter - pre_counter > 1000) {
    //   std::cout << "work counter: " << counter << std::endl;
    //   pre_counter = counter;
    // }
    uint64_t key_value = thread_workload_array[counter];
    op_type cur_op = static_cast<op_type>(key_value >> 56);
    key_value = key_value & op_mask;
    Key key = int2key(key_value);

    if (counter % 20 == 0) {
      thread_timer.begin();
    }

    switch (cur_op) {
      case op_type::Lookup: {
        Value v;
        auto flag = tree->search(key, v, nullptr);
        if (flag) ++success_counter;
      } break;

      case op_type::Insert: {
        Value v = rand() % 1000000;
        tree->insert(key, v);
      } break;

      case op_type::Update: {
        Value v = rand() % 1000000;
        tree->update(key, v);
      } break;
      case op_type::Range: {
        std::map<Key, Value> ret;
        tree->range_query(key, key + scan_num, ret);
      } break;

      default:
        std::cout << "OP Type NOT MATCH!" << std::endl;
    }
    if (counter % 20 == 19) {
      auto us_10 = thread_timer.end() / 100;
      if (us_10 >= LATENCY_WINDOWS) {
        us_10 = LATENCY_WINDOWS - 1;
      }
      xmd_latency[id][us_10]++;
    }

    tp[id][0]++;
    ++counter;
    // if (counter % 1000000 == 0) {
    //   std::cout << "Thread ID = " << id << "--------------------------------"
    //             << std::endl;
    //   cachepush::decision.show_statistic();
    //   std::cout << "-------------------------------------" << std::endl;
    // }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  total_time[id] = static_cast<uint64_t>(duration);
  // The one who first finish should terminate all threads
  // To avoid the straggling thread
  if (early_stop && !one_finish.load()) {
    one_finish.store(true);
    per_thread_op_num = 0;
  }

  worker.fetch_sub(1);

  // uint64_t throughput =
  //     counter / (static_cast<double>(duration) / std::pow(10, 6));
  // total_tp[id] = throughput;  // (ops/s)
  // total_time[id] = static_cast<uint64_t>(duration);
  // std::cout << "Success ratio = "
  //           << success_counter / static_cast<double>(counter) << std::endl;
  execute_op.fetch_add(counter);
  // if (cachepush::total_sample_times != 0) {
  //   std::cout << "Node search time(ns) = "
  //             << cachepush::total_nanoseconds / cachepush::total_sample_times
  //             << std::endl;
  //   std::cout << "Sample times = " << cachepush::total_sample_times
  //             << std::endl;
  // }
  // std::cout << "Real rpc ratio = " << tree->get_rpc_ratio() << std::endl;
}

void parse_args(int argc, char *argv[]) {
  if (argc != 16) {
    printf("argc = %d\n", argc);
    printf(
        "Usage: ./benchmark kNodeCount kReadRatio kInsertRatio kUpdateRatio "
        "kDeleteRatio kRangeRatio "
        "totalThreadCount memThreadCount "
        "cacheSize(MB) uniform_workload zipfian_theta bulk_load_num "
        "warmup_num op_num "
        "check_correctness(0=no, 1=yes) time_based(0=no, "
        "1=yes) early_stop(0=no, 1=yes) "
        " \n");
    exit(-1);
  }

  kNodeCount = atoi(argv[1]);
  kReadRatio = atoi(argv[2]);
  kInsertRatio = atoi(argv[3]);
  kUpdateRatio = atoi(argv[4]);
  kDeleteRatio = atoi(argv[5]);
  kRangeRatio = atoi(argv[6]);
  assert((kReadRatio + kInsertRatio + kUpdateRatio + kDeleteRatio +
          kRangeRatio) == 100);

  totalThreadCount = atoi(argv[7]);  // Here is total thread count
  memThreadCount = atoi(argv[8]);    // #threads in memory node

  cache_mb = atoi(argv[9]);
  workload_type = static_cast<WorkLoadType>(atoi(argv[10]));
  zipfian = atof(argv[11]);
  bulk_load_num = atoi(argv[12]) * 1000 * 1000;
  warmup_num = atoi(argv[13]) * 1000 * 1000;
  op_num = atoi(argv[14]) * 1000 * 1000;  // Here is total op_num => need to be
                                          // distributed across the bechmark
  kMaxThread = atoi(argv[15]);
  // How to make insert ready?
  kKeySpace = bulk_load_num +
              ceil((op_num + warmup_num) * (kInsertRatio / 100.0)) + 1000;

  // Get thread_key_space
  threadKSpace = kKeySpace / totalThreadCount;

  CNodeCount = (totalThreadCount % kMaxThread == 0)
                   ? (totalThreadCount / kMaxThread)
                   : (totalThreadCount / kMaxThread + 1);
  std::cout << "Compute node count = " << CNodeCount << std::endl;
  printf(
      "kNodeCount %d, kReadRatio %d, kInsertRatio %d, kUpdateRatio %d, "
      "kDeleteRatio %d, kRangeRatio %d, "
      "totalThreadCount %d, memThreadCount %d "
      "cache_size %lu, workload_type %u, zipfian %lf, bulk_load_num %lu, "
      "warmup_num %lu, "
      "op_num "
      "%lu, check_correctness %d, time_based %d, early_stop %d\n",
      kNodeCount, kReadRatio, kInsertRatio, kUpdateRatio, kDeleteRatio,
      kRangeRatio, totalThreadCount, memThreadCount, cache_mb, workload_type,
      zipfian, bulk_load_num, warmup_num, op_num, check_correctness, time_based,
      early_stop);
  std::cout << "kMaxThread = " << kMaxThread << std::endl;
  std::cout << "KeySpace = " << kKeySpace << std::endl;
}

void cal_latency() {
  uint64_t all_lat = 0;
  for (int i = 0; i < LATENCY_WINDOWS; ++i) {
    latency_th_all[i] = 0;
    for (int k = 0; k < MAX_APP_THREAD; ++k) {
      latency_th_all[i] += xmd_latency[k][i];
    }
    all_lat += latency_th_all[i];
  }

  uint64_t th50 = all_lat / 2;
  uint64_t th90 = all_lat * 9 / 10;
  uint64_t th95 = all_lat * 95 / 100;
  uint64_t th99 = all_lat * 99 / 100;
  uint64_t th999 = all_lat * 999 / 1000;

  uint64_t cum = 0;
  for (int i = 0; i < LATENCY_WINDOWS; ++i) {
    cum += latency_th_all[i];

    if (cum >= th50) {
      printf("p50 %f\t", i / 10.0);
      th50 = -1;
    }
    if (cum >= th90) {
      printf("p90 %f\t", i / 10.0);
      th90 = -1;
    }
    if (cum >= th95) {
      printf("p95 %f\t", i / 10.0);
      th95 = -1;
    }
    if (cum >= th99) {
      printf("p99 %f\t", i / 10.0);
      th99 = -1;
    }
    if (cum >= th999) {
      printf("p999 %f\n", i / 10.0);
      th999 = -1;
      return;
    }
  }
}

int main(int argc, char *argv[]) {
  numa_set_preferred(1);
  parse_args(argc, argv);

  // std::thread overhead_th[NR_DIRECTORY];
  // for (int i = 0; i < memThreadCount; i++) {
  //   overhead_th[i] = std::thread( dirthread_run, i);
  // }

  DSMConfig config;
  config.machineNR = kNodeCount;
  config.computeNR = CNodeCount;
  dsm = DSM::getInstance(config);
  // XMD::global_dsm_ = dsm;
  // #Worker-threads in this CNode
  node_id = dsm->getMyNodeID();
  if (node_id == (CNodeCount - 1)) {
    kThreadCount = totalThreadCount - ((CNodeCount - 1) * kMaxThread);
  } else {
    kThreadCount = kMaxThread;
  }

  double collect_throughput = 0;
  uint64_t total_throughput = 0;
  double total_max_time = 0;
  double total_cluster_max_time = 0;
  uint64_t total_cluster_tp = 0;
  uint64_t straggler_cluster_tp = 0;
  uint64_t collect_times = 0;

  if (node_id < CNodeCount) {
    dsm->registerThread();
    tree = new Tree(dsm);

    if (dsm->getMyNodeID() == 0) {
      for (uint64_t i = 1; i < 1024000; ++i) {
        tree->insert(int2key(i), i * 2);
      }
    }

    dsm->barrier("init__benchmark", CNodeCount);
    generate_workload();
    bulk_load();

    dsm->barrier("bulkload--finish", CNodeCount);

    std::cout << node_id << " is ready for the benchmark" << std::endl;

    dsm->resetThread();
    reset_all_params();

    std::thread ths[MAX_APP_THREAD];

    // thread_run(0);
    for (int i = 0; i < kThreadCount; i++) {
      ths[i] = std::thread(thread_run, i);
    }

    // Warmup
    auto start = std::chrono::high_resolution_clock::now();
    while (!ready.load()) {
      sleep(2);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
      if (time_based && duration >= 30) {
        per_thread_warmup_num = 0;
      }
    }

    // Main thread is used to collect the statistics
    timespec s, e;
    uint64_t pre_tp = 0;
    uint64_t pre_ths[MAX_APP_THREAD];
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      pre_ths[i] = 0;
    }

    ready_to_report.store(true);
    clock_gettime(CLOCK_REALTIME, &s);
    bool start_generate_throughput = false;

    std::cout << "Start collecting the statistic" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    // Create a monitor thread to exit if waiting too long
    std::atomic<bool> monitor_exit{false};
    std::thread monitor_thread([&monitor_exit]() {
      // Wait for 5 minutes (300 seconds)
      std::this_thread::sleep_for(std::chrono::seconds(100));
      if (!monitor_exit.load()) {
        std::cerr << "Monitor thread: Program has been running for too long. "
                     "Forcing exit."
                  << std::endl;
        exit(1);
      }
    });
    monitor_thread.detach();

    // System::profile("dex-test", [&]() {
    int iter = 0;
    while (true) {
      sleep(2);
      clock_gettime(CLOCK_REALTIME, &e);
      int microseconds = (e.tv_sec - s.tv_sec) * 1000000 +
                         (double)(e.tv_nsec - s.tv_nsec) / 1000;

      uint64_t all_tp = 0;
      for (int i = 0; i < kThreadCount; ++i) {
        all_tp += tp[i][0];
      }

      // Throughput in current phase (for very two seconds)
      uint64_t cap = all_tp - pre_tp;
      pre_tp = all_tp;

      for (int i = 0; i < kThreadCount; ++i) {
        auto val = tp[i][0];
        pre_ths[i] = val;
      }

      clock_gettime(CLOCK_REALTIME, &s);
      double per_node_tp = cap * 1.0 / microseconds;

      // FIXME(BT): use static counter for increment, need fix
      // uint64_t cluster_tp =
      //     dsm->sum((uint64_t)(per_node_tp * 1000), CNodeCount);
      uint64_t cluster_tp = dsm->sum_with_prefix(
          std::string("sum-") + std::string("-") + std::to_string(iter),
          (uint64_t)(per_node_tp * 1000), CNodeCount);

      monitor_exit.store(true);  // Signal monitor thread to exit

      // uint64_t cluster_tp = 0;
      printf("%d, throughput %.4f\n", dsm->getMyNodeID(), per_node_tp);
      // save_latency(iter);

      if (dsm->getMyNodeID() == 0) {
        printf("cluster throughput %.3f\n", cluster_tp / 1000.0);

        if (cluster_tp != 0) {
          start_generate_throughput = true;
        }

        // Means this Cnode already finish the workload
        if (start_generate_throughput && cluster_tp == 0) {
          auto end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::seconds>(end - start)
                  .count();
          std::cout << "The time duration = " << duration << " seconds"
                    << std::endl;
          monitor_exit.store(true);  // Signal monitor thread to exit
          break;
        }

        if (start_generate_throughput) {
          ++collect_times;
          collect_throughput += cluster_tp / 1000.0;
          auto end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::seconds>(end - start)
                  .count();
          if (time_based && duration > 60) {
            std::cout << "Running time is larger than " << 60 << "seconds"
                      << std::endl;
            per_thread_op_num = 0;
            monitor_exit.store(true);  // Signal monitor thread to exit
            break;
          }
        }
      } else {
        if (cluster_tp != 0) {
          start_generate_throughput = true;
        }

        if (start_generate_throughput && per_node_tp == 0) {
          monitor_exit.store(true);  // Signal monitor thread to exit
          break;
        }

        if (start_generate_throughput) {
          ++collect_times;
          auto end = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::seconds>(end - start)
                  .count();
          if (time_based && duration > 60) {
            per_thread_op_num = 0;
            monitor_exit.store(true);  // Signal monitor thread to exit
            break;
          }
        }
      }
      ++iter;
    }  // while(true) loop
       //});
    std::chrono::high_resolution_clock::time_point sleep_start =
        std::chrono::high_resolution_clock::now();
    sleep(2);
    while (worker.load() != 0) {
      sleep(2);
      auto sleep_end = std::chrono::high_resolution_clock::now();
      auto sleep_duration = std::chrono::duration_cast<std::chrono::seconds>(
                                sleep_end - sleep_start)
                                .count();
      if (sleep_duration > 30) {
        exit(0);
      }
    }

    for (int i = 0; i < kThreadCount; i++) {
      ths[i].join();
    }
    // for (int i = 0; i < memThreadCount; i ++) {
    //   overhead_th[i].join();
    // }

    // for (int i = 0; i < kThreadCount; ++i) {
    //   total_max_time = std::max_element(total_time, total_time + k)
    // }
    total_max_time = *std::max_element(total_time, total_time + kThreadCount);
    if (workload_type == WorkLoadType::uniform ||
        workload_type == WorkLoadType::zipf) {
      total_cluster_max_time = total_max_time;
    } else if (workload_type == WorkLoadType::gaussian_01 ||
               workload_type == WorkLoadType::gaussian_001) {
      total_cluster_max_time = dsm->max_total(total_max_time, CNodeCount);
    } else {
      assert(false);
    }
    // total_cluster_max_time = dsm->max_total(total_max_time, CNodeCount);
    std::cout << "XMD node max time: " << total_max_time;
    std::cout << "XMD cluster max time: " << total_cluster_max_time;

    uint64_t XMDsetting_node_throughput =
        execute_op.load() /
        (static_cast<double>(total_cluster_max_time) / std::pow(10, 6));
    uint64_t XMDsetting_cluster_throughput =
        dsm->sum_total(XMDsetting_node_throughput, CNodeCount, false);

    std::cout << "XMD node throughput: "
              << (static_cast<double>(XMDsetting_node_throughput) /
                  std::pow(10, 6))
              << " MOPS" << std::endl;
    std::cout << "XMD cluster throughput: "
              << (static_cast<double>(XMDsetting_cluster_throughput) /
                  std::pow(10, 6))
              << " MOPS" << std::endl;
    std::cout << "XMD cluster latency: node " << node_id << " " << std::endl;
    cal_latency();

    std::cout << "XMD RDMA info: " << node_id << " " << std::endl;

    uint64_t rdma_read_num = dsm->get_rdma_read_num();
    uint64_t rdma_write_num = dsm->get_rdma_write_num();
    uint64_t rdma_read_time = dsm->get_rdma_read_time();
    uint64_t rdma_write_time = dsm->get_rdma_write_time();
    int64_t rdma_read_size = dsm->get_rdma_read_size();
    uint64_t rdma_write_size = dsm->get_rdma_write_size();
    uint64_t rdma_cas_num = dsm->get_rdma_cas_num();
    uint64_t rdma_rpc_num = dsm->get_rdma_rpc_num();
    std::cout << "Avg. rdma read time(ms) = "
              << static_cast<double>(rdma_read_time) / 1000 / rdma_read_num
              << std::endl;
    std::cout << "Avg. rdma write time(ms) = "
              << static_cast<double>(rdma_write_time) / 1000 / rdma_write_num
              << std::endl;
    std::cout << "Avg. rdma read / op = "
              << static_cast<double>(rdma_read_num) / execute_op.load()
              << std::endl;
    std::cout << "Avg. rdma write / op = "
              << static_cast<double>(rdma_write_num) / execute_op.load()
              << std::endl;
    std::cout << "Avg. rdma cas / op = "
              << static_cast<double>(rdma_cas_num) / execute_op.load()
              << std::endl;
    std::cout << "Avg. rdma rpc / op = "
              << static_cast<double>(rdma_rpc_num) / execute_op.load()
              << std::endl;
    std::cout << "Avg. all rdma / op = "
              << static_cast<double>(rdma_read_num + rdma_write_num +
                                     rdma_cas_num + rdma_rpc_num) /
                     execute_op.load()
              << std::endl;
    std::cout << "Avg. rdma read size/ op = "
              << static_cast<double>(rdma_read_size) / execute_op.load()
              << std::endl;
    std::cout << "Avg. rdma write size / op = "
              << static_cast<double>(rdma_write_size) / execute_op.load()
              << std::endl;
    std::cout << "Avg. rdma RW size / op = "
              << static_cast<double>(rdma_read_size + rdma_write_size) /
                     execute_op.load()
              << std::endl;

    // uint64_t max_time = 0;
    // for (int i = 0; i < kThreadCount; ++i) {
    //   max_time = std::max<uint64_t>(max_time, total_time[i]);
    // }

    total_cluster_tp = dsm->sum_total(total_throughput, CNodeCount, false);
    straggler_cluster_tp =
        dsm->min_total(total_throughput / kThreadCount, CNodeCount);
    straggler_cluster_tp = straggler_cluster_tp * totalThreadCount;
    // op_num /
    // (static_cast<double>(straggler_cluster_tp) / std::pow(10, 6));

    throughput_vec.push_back(total_cluster_tp);
    straggler_throughput_vec.push_back(straggler_cluster_tp);
    std::cout << "Round "
              << " (max_throughput): " << total_cluster_tp / std::pow(10, 6)
              << " Mops/s" << std::endl;
    std::cout << "Round " << " (straggler_throughput): "
              << straggler_cluster_tp / std::pow(10, 6) << " Mops/s"
              << std::endl;
  }

  std::cout << "Before barrier finish" << std::endl;
  dsm->barrier("finish");
}