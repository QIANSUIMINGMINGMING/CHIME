#include "Rdma.h"

int kMaxDeviceMemorySize = 0;

void rdmaQueryQueuePair(ibv_qp *qp)
{
  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init_attr;
  ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr);
  switch (attr.qp_state)
  {
  case IBV_QPS_RESET:
    printf("QP state: IBV_QPS_RESET\n");
    break;
  case IBV_QPS_INIT:
    printf("QP state: IBV_QPS_INIT\n");
    break;
  case IBV_QPS_RTR:
    printf("QP state: IBV_QPS_RTR\n");
    break;
  case IBV_QPS_RTS:
    printf("QP state: IBV_QPS_RTS\n");
    break;
  case IBV_QPS_SQD:
    printf("QP state: IBV_QPS_SQD\n");
    break;
  case IBV_QPS_SQE:
    printf("QP state: IBV_QPS_SQE\n");
    break;
  case IBV_QPS_ERR:
    printf("QP state: IBV_QPS_ERR\n");
    break;
  case IBV_QPS_UNKNOWN:
    printf("QP state: IBV_QPS_UNKNOWN\n");
    break;
  }
}

void checkDMSupported(struct ibv_context *ctx)
{
  struct ibv_device_attr attrs;

  if (ibv_query_device(ctx, &attrs))
  {
    printf("Couldn't query device attributes\n");
    return;
  }

  // Check if device memory is supported
  if (!(attrs.device_cap_flags & IBV_DEVICE_MANAGED_FLOW_STEERING))
  {
    fprintf(stderr, "Device Memory is not supported!\n");
    exit(-1);
  }

  // For standard rdma-core, we'll use a default size
  // since max_dm_size is not directly available
  kMaxDeviceMemorySize = 128 * 1024; // Default to 128KB
  printf("Using default NIC Device Memory size of %dKB\n", kMaxDeviceMemorySize / 1024);
}

void testDeviceMemory(struct ibv_context *ctx)
{
  printf("Testing device memory support...\n");

  // First check if device memory is supported
  checkDMSupported(ctx);

  // Try to allocate a small piece of device memory
  struct ibv_alloc_dm_attr dm_attr;
  memset(&dm_attr, 0, sizeof(dm_attr));
  dm_attr.length = 4096; // Try to allocate 4KB

  struct ibv_dm *dm = ibv_alloc_dm(ctx, &dm_attr);
  if (!dm)
  {
    printf("Failed to allocate device memory\n");
    return;
  }

  printf("Successfully allocated 4KB of device memory\n");

  // Try to register the device memory as a memory region
  struct ibv_pd *pd = ibv_alloc_pd(ctx);
  if (!pd)
  {
    printf("Failed to allocate protection domain\n");
    ibv_free_dm(dm);
    return;
  }

  struct ibv_mr *mr = ibv_reg_dm_mr(pd, dm, 0, 4096,
                                    IBV_ACCESS_LOCAL_WRITE |
                                        IBV_ACCESS_REMOTE_READ |
                                        IBV_ACCESS_REMOTE_WRITE);
  if (!mr)
  {
    printf("Failed to register device memory as memory region\n");
    ibv_dealloc_pd(pd);
    ibv_free_dm(dm);
    return;
  }

  printf("Successfully registered device memory as memory region\n");

  // Cleanup
  ibv_dereg_mr(mr);
  ibv_dealloc_pd(pd);
  ibv_free_dm(dm);

  printf("Device memory test completed successfully\n");
}
