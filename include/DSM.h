#ifndef __DSM_H__
#define __DSM_H__

#include <atomic>

#include "Common.h"
#include "Config.h"
#include "Connection.h"
#include "DSMKeeper.h"
#include "GlobalAddress.h"
#include "LocalAllocator.h"
#include "RdmaBuffer.h"
#include "RdmaCache.h"

#define COUNT_RDMA 1

class DSMKeeper;
class Directory;

class DSM {
 public:
  void registerThread();
  void resetThread() { appID.store(0); }
  void loadKeySpace(const std::string &load_workloads_path, bool is_str);
  Key getRandomKey();
  Key getNoComflictKey(uint64_t key_hash, uint64_t global_thread_id,
                       uint64_t global_thread_num);
  static DSM *getInstance(const DSMConfig &conf);

  uint16_t getMyNodeID() { return myNodeID; }
  uint16_t getMemoryNodeID() { return conf.machineNR - 1; }
  uint16_t getMyThreadID() { return thread_id; }
  uint16_t getClusterSize() { return conf.machineNR; }
  uint64_t getThreadTag() { return thread_tag; }
  uint64_t getMyGlobalThreadID() {
    return conf.threadNR * myNodeID + (thread_id - 1);
  }
  uint32_t getComputeNum() { return conf.computeNR; }

  // RDMA operations
  // buffer is registered memory
  void read(char *buffer, GlobalAddress gaddr, size_t size, bool signal = true,
            CoroPull *sink = nullptr);
  void read_sync(char *buffer, GlobalAddress gaddr, size_t size,
                 CoroPull *sink = nullptr);
  void read_sync_without_sink(char *buffer, GlobalAddress gaddr, size_t size,
                              CoroPull *sink, CoroQueue *waiting_queue);

  void write(const char *buffer, GlobalAddress gaddr, size_t size,
             bool signal = true, CoroPull *sink = nullptr);
  void write_without_sink(const char *buffer, GlobalAddress gaddr, size_t size,
                          bool signal, CoroPull *sink,
                          CoroQueue *waiting_queue);
  void write_sync(const char *buffer, GlobalAddress gaddr, size_t size,
                  CoroPull *sink = nullptr);
  void write_sync_without_sink(const char *buffer, GlobalAddress gaddr,
                               size_t size, CoroPull *sink,
                               CoroQueue *waiting_queue);

  void read_batch(RdmaOpRegion *rs, int k, bool signal = true,
                  CoroPull *sink = nullptr);
  void read_batch_sync(RdmaOpRegion *rs, int k, CoroPull *sink = nullptr);
  void read_batch_sync_without_sink(RdmaOpRegion *rs, int k, CoroPull *sink,
                                    CoroQueue *waiting_queue);
  void read_batches_sync(const std::vector<RdmaOpRegion> &rs,
                         CoroPull *sink = nullptr);

  void write_batch(RdmaOpRegion *rs, int k, bool signal = true,
                   CoroPull *sink = nullptr);
  void write_batch_without_sink(RdmaOpRegion *rs, int k, bool signal,
                                CoroPull *sink, CoroQueue *waiting_queue);
  void write_batch_sync(RdmaOpRegion *rs, int k, CoroPull *sink = nullptr);
  void write_batch_sync_without_sink(RdmaOpRegion *rs, int k, CoroPull *sink,
                                     CoroQueue *waiting_queue);
  void write_batches_sync(const std::vector<RdmaOpRegion> &rs,
                          CoroPull *sink = nullptr);

  void write_faa(RdmaOpRegion &write_ror, RdmaOpRegion &faa_ror,
                 uint64_t add_val, bool signal = true,
                 CoroPull *sink = nullptr);
  void write_faa_sync(RdmaOpRegion &write_ror, RdmaOpRegion &faa_ror,
                      uint64_t add_val, CoroPull *sink = nullptr);

  void write_cas(RdmaOpRegion &write_ror, RdmaOpRegion &cas_ror, uint64_t equal,
                 uint64_t val, bool signal = true, CoroPull *sink = nullptr);
  void write_cas_sync(RdmaOpRegion &write_ror, RdmaOpRegion &cas_ror,
                      uint64_t equal, uint64_t val, CoroPull *sink = nullptr);

  void cas(GlobalAddress gaddr, uint64_t equal, uint64_t val,
           uint64_t *rdma_buffer, bool signal = true, CoroPull *sink = nullptr);
  bool cas_sync(GlobalAddress gaddr, uint64_t equal, uint64_t val,
                uint64_t *rdma_buffer, CoroPull *sink = nullptr);

  void cas_read(RdmaOpRegion &cas_ror, RdmaOpRegion &read_ror, uint64_t equal,
                uint64_t val, bool signal = true, CoroPull *sink = nullptr);
  bool cas_read_sync(RdmaOpRegion &cas_ror, RdmaOpRegion &read_ror,
                     uint64_t equal, uint64_t val, CoroPull *sink = nullptr);

  void read_cas(RdmaOpRegion &read_ror, RdmaOpRegion &cas_ror, uint64_t equal,
                uint64_t val, bool signal = true, CoroPull *sink = nullptr);
  bool read_cas_sync(RdmaOpRegion &read_ror, RdmaOpRegion &cas_ror,
                     uint64_t equal, uint64_t val, CoroPull *sink = nullptr);

  void cas_write(RdmaOpRegion &cas_ror, RdmaOpRegion &write_ror, uint64_t equal,
                 uint64_t val, bool signal = true, CoroPull *sink = nullptr);
  bool cas_write_sync(RdmaOpRegion &cas_ror, RdmaOpRegion &write_ror,
                      uint64_t equal, uint64_t val, CoroPull *sink = nullptr);

  // for on-chip device memory
  void read_dm(char *buffer, GlobalAddress gaddr, size_t size,
               bool signal = true, CoroPull *sink = nullptr);
  void read_dm_sync(char *buffer, GlobalAddress gaddr, size_t size,
                    CoroPull *sink = nullptr);

  void write_dm(const char *buffer, GlobalAddress gaddr, size_t size,
                bool signal = true, CoroPull *sink = nullptr);
  void write_dm_sync(const char *buffer, GlobalAddress gaddr, size_t size,
                     CoroPull *sink = nullptr);

  void cas_dm(GlobalAddress gaddr, uint64_t equal, uint64_t val,
              uint64_t *rdma_buffer, bool signal = true,
              CoroPull *sink = nullptr);
  bool cas_dm_sync(GlobalAddress gaddr, uint64_t equal, uint64_t val,
                   uint64_t *rdma_buffer, CoroPull *sink = nullptr);

  uint64_t poll_rdma_cq(int count = 1);
  bool poll_rdma_cq_once(uint64_t &wr_id);
  int poll_rdma_cq_batch_once(uint64_t *wr_ids, int count);

  uint64_t sum(uint64_t value) {
    static uint64_t count = 0;
    return keeper->sum(std::string("sum-") + std::to_string(count++), value);
  }

  uint64_t sum(uint64_t value, int node_num) {
    static uint64_t count = 0;
    return keeper->sum(std::string("sum-") + std::to_string(count++), value,
                       node_num, true);
  }

  uint64_t sum_with_prefix(std::string prefix, uint64_t value, int node_num) {
    // std::cout << "Node num = " << node_num << std::endl;
    return keeper->sum(prefix, value, node_num, true);
  }

  uint64_t sum_total(uint64_t value, bool time_out) {
    static uint64_t count = 0;
    return keeper->sum(std::string("sum-total") + std::to_string(count++),
                       value, time_out);
  }

  uint64_t sum_total(uint64_t value, int node_num, bool time_out) {
    static uint64_t count = 0;
    return keeper->sum(std::string("sum-total") + std::to_string(count++),
                       value, node_num, time_out);
  }

  uint64_t min_total(uint64_t value, int node_num) {
    static uint64_t count = 0;
    return keeper->min(std::string("min-total") + std::to_string(count++),
                       value, node_num);
  }

  uint64_t max_total(uint64_t value, int node_num) {
    static uint64_t count = 0;
    return keeper->max(std::string("max-total") + std::to_string(count++),
                       value, node_num);
  }

  // Memcached operations for sync
  size_t Put(uint64_t key, const void *value, size_t count) {
    std::string k = std::string("gam-") + std::to_string(key);
    keeper->memSet(k.c_str(), k.size(), (char *)value, count);
    return count;
  }

  size_t Get(uint64_t key, void *value) {
    std::string k = std::string("gam-") + std::to_string(key);
    size_t size;
    char *ret = keeper->memGet(k.c_str(), k.size(), &size);
    memcpy(value, ret, size);

    return size;
  }

#ifdef COUNT_RDMA

  uint64_t get_rdma_read_num() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += num_rdma_read[i][0];
    }
    return total;
  }

  uint64_t get_rdma_write_num() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += num_rdma_write[i][0];
    }
    return total;
  }

  uint64_t get_rdma_read_size() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += size_rdma_read[i][0];
    }
    return total;
  }

  uint64_t get_rdma_write_size() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += size_rdma_write[i][0];
    }
    return total;
  }

  uint64_t get_rdma_write_time() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += time_rdma_write[i][0];
    }
    return total;
  }

  uint64_t get_rdma_read_time() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += time_rdma_read[i][0];
    }
    return total;
  }

  uint64_t get_rdma_cas_num() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += num_rdma_cas[i][0];
    }
    return total;
  }

  uint64_t get_rdma_rpc_num() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_APP_THREAD; ++i) {
      total += num_rdma_rpc[i][0];
    }
    return total;
  }

  void clear_rdma_statistic() {
    memset(reinterpret_cast<void *>(num_rdma_read), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
    memset(reinterpret_cast<void *>(time_rdma_read), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
    memset(reinterpret_cast<void *>(num_rdma_write), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
    memset(reinterpret_cast<void *>(time_rdma_write), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
    memset(reinterpret_cast<void *>(num_rdma_cas), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
    memset(reinterpret_cast<void *>(num_rdma_rpc), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
    memset(reinterpret_cast<void *>(size_rdma_read), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
    memset(reinterpret_cast<void *>(size_rdma_write), 0,
           sizeof(uint64_t) * MAX_APP_THREAD * 8);
  }

  // RDMA statistic
  uint64_t num_rdma_read[MAX_APP_THREAD][8];
  uint64_t time_rdma_read[MAX_APP_THREAD][8];
  uint64_t num_rdma_write[MAX_APP_THREAD][8];
  uint64_t time_rdma_write[MAX_APP_THREAD][8];
  uint64_t num_rdma_cas[MAX_APP_THREAD][8];
  uint64_t num_rdma_rpc[MAX_APP_THREAD][8];
  uint64_t size_rdma_read[MAX_APP_THREAD][8];
  uint64_t size_rdma_write[MAX_APP_THREAD][8];
#endif

 private:
  DSM(const DSMConfig &conf);
  ~DSM();

  void initRDMAConnection();
  void fill_keys_dest(RdmaOpRegion &ror, GlobalAddress addr, bool is_chip);

  DSMConfig conf;
  std::atomic_int appID;
  Cache cache;

  static thread_local int thread_id;
  static thread_local uint64_t thread_tag;
  static thread_local ThreadConnection *iCon;
  static thread_local char *rdma_buffer;
  static thread_local LocalAllocator local_allocators[MEMORY_NODE_NUM]
                                                     [NR_DIRECTORY];
  static thread_local RdmaBuffer rbuf[MAX_CORO_NUM];

  uint64_t baseAddr;
  uint32_t myNodeID;
  uint64_t keySpaceSize;

  RemoteConnection *remoteInfo;
  ThreadConnection *thCon[MAX_APP_THREAD];
  DirectoryConnection *dirCon[NR_DIRECTORY];
  DSMKeeper *keeper;

  Directory *dirAgent[NR_DIRECTORY];
  Key keyBuffer[MAX_KEY_SPACE_SIZE];

 public:
  bool is_register() { return thread_id != -1; }
  void barrier(const std::string &ss) { keeper->barrier(ss); }
  void barrier(const std::string &ss, int node_num) {
    keeper->barrier(ss, node_num);
  }

  char *get_rdma_buffer() { return rdma_buffer; }
  RdmaBuffer &get_rbuf(CoroPull *sink) { return rbuf[sink ? sink->get() : 0]; }

  GlobalAddress alloc(size_t size, uint8_t align_bit = CACHELINE_ALIGN_BIT);
  void free(const GlobalAddress &addr, int size);

  void rpc_call_dir(const RawMessage &m, uint16_t node_id,
                    uint16_t dir_id = 0) {
#ifdef COUNT_RDMA
    num_rdma_rpc[getMyThreadID()][0]++;
    size_rdma_read[getMyThreadID()][0] += sizeof(RawMessage);
#endif

    auto buffer = (RawMessage *)iCon->message->getSendPool();

    memcpy(buffer, &m, sizeof(RawMessage));
    buffer->node_id = myNodeID;
    buffer->app_id = thread_id;

    iCon->sendMessage2Dir(buffer, node_id, dir_id);
  }

  RawMessage *rpc_wait() {
    ibv_wc wc;

    pollWithCQ(iCon->rpc_cq, 1, &wc);
    return (RawMessage *)iCon->message->getMessage();
  }
};

inline GlobalAddress DSM::alloc(size_t size, uint8_t align_bit) {
  // thread_local int cur_target_node =
  //     (this->getMyThreadID() + this->getMyNodeID()) % MEMORY_NODE_NUM;
  // thread_local int cur_target_dir_id =
  //     (this->getMyThreadID() + this->getMyNodeID()) % NR_DIRECTORY;
  thread_local int cur_target_node = conf.machineNR - 1;
  thread_local int cur_target_dir_id =
      (this->getMyThreadID() + this->getMyNodeID()) % NR_DIRECTORY;
  if (++cur_target_dir_id == NR_DIRECTORY) {
    // cur_target_node = (cur_target_node + 1) % MEMORY_NODE_NUM;
    // cur_target_dir_id = 0;
    cur_target_node = conf.machineNR - 1;
    cur_target_dir_id = 0;
  }

  auto &local_allocator = local_allocators[cur_target_node][cur_target_dir_id];

  // alloc from the target node
  bool need_chunk = true;
  GlobalAddress addr = local_allocator.malloc(size, need_chunk, align_bit);
  if (need_chunk) {
    RawMessage m;
    m.type = RpcType::MALLOC;

    this->rpc_call_dir(m, cur_target_node, cur_target_dir_id);
    local_allocator.set_chunck(rpc_wait()->addr);

    // retry
    addr = local_allocator.malloc(size, need_chunk, align_bit);
  }
  return addr;
}

inline void DSM::free(const GlobalAddress &addr, int size) {
  local_allocators[addr.nodeID][0].free(addr, size);
}

#endif /* __DSM_H__ */
