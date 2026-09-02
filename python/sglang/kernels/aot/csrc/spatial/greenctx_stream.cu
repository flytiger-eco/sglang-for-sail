// Documentation: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
#include <torch/all.h>

#include <cstdlib>

#include "cuda_utils.h"
#include "greenctx_stream.h"

static int CUDA_DRIVER_VERSION;

using PFN_cuGreenCtxStreamCreate = HGresult(HGGCAPI*)(HGstream*, HGgreenCtx, unsigned int, int);

auto probe_cuGreenCtxStreamCreate() -> PFN_cuGreenCtxStreamCreate {
  static PFN_cuGreenCtxStreamCreate pfn = nullptr;
  CUDA_DRV(hgGetProcAddress("cuGreenCtxStreamCreate", reinterpret_cast<void**>(&pfn), CUDA_DRIVER_VERSION, 0, nullptr));
  return pfn;
}

static std::vector<int64_t> create_greenctx_stream_fallback(HGgreenCtx gctx[2]) {
  HGstream streamA, streamB;
  HGcontext ctx;

  CUDA_DRV(hgCtxFromGreenCtx(&ctx, gctx[0]));
  CUDA_DRV(hgCtxPushCurrent(ctx));
  CUDA_DRV(hgStreamCreate(&streamA, HG_STREAM_NON_BLOCKING));
  CUDA_DRV(hgCtxPopCurrent(nullptr));

  CUDA_DRV(hgCtxFromGreenCtx(&ctx, gctx[1]));
  CUDA_DRV(hgCtxPushCurrent(ctx));
  CUDA_DRV(hgStreamCreate(&streamB, HG_STREAM_NON_BLOCKING));
  CUDA_DRV(hgCtxPopCurrent(nullptr));

  return {(int64_t)streamA, (int64_t)streamB};
}

inline void destroy_green_context(HGgreenCtx gctx) {
  if (!gctx) return;
  CUDA_DRV(hgGreenCtxDestroy(gctx));
}

static std::vector<int64_t> create_greenctx_stream_direct_dynamic(HGgreenCtx gctx[2]) {
  // This symbol is introduced in CUDA 12.5
  const static auto pfn = probe_cuGreenCtxStreamCreate();
  if (!pfn) {
    TORCH_WARN("cuGreenCtxStreamCreate(cuda>=12.5) is not available, using fallback");
    return create_greenctx_stream_fallback(gctx);
  }

  HGstream streamA, streamB;
  CUDA_DRV(pfn(&streamA, gctx[0], HG_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(pfn(&streamB, gctx[1], HG_STREAM_NON_BLOCKING, 0));

  return {(int64_t)streamA, (int64_t)streamB};
}

std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device) {
  CUDA_DRV(compatibleDriverGetVersion(&CUDA_DRIVER_VERSION));

  HGgreenCtx gctx[3];
  HGdevResourceDesc desc[3];
  HGdevResource input;
  HGdevResource resources[4];

  TORCH_CHECK(smA > 0 && smB > 0, "SM counts must be positive");

  CUDA_DRV(hgDeviceGetDevResource((HGdevice)device, &input, HG_DEV_RESOURCE_TYPE_SM));

  const unsigned minCount = static_cast<unsigned>(smA + smB);
  const unsigned minCountA = static_cast<unsigned>(smA);
  TORCH_CHECK(minCount <= input.sm.smCount, "Not enough SMs available for the requested configuration");

  unsigned nbGroups = 1;
  CUDA_DRV(hgDevSmResourceSplitByCount(&resources[2], &nbGroups, &input, &resources[3], 0, minCount));
  CUDA_DRV(hgDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(hgGreenCtxCreate(&gctx[2], desc[2], (HGdevice)device, HG_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(hgGreenCtxGetDevResource(gctx[2], &input, HG_DEV_RESOURCE_TYPE_SM));
  nbGroups = 1;
  CUDA_DRV(hgDevSmResourceSplitByCount(&resources[0], &nbGroups, &input, &resources[1], 0, minCountA));
  CUDA_DRV(hgDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(hgGreenCtxCreate(&gctx[0], desc[0], (HGdevice)device, HG_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(hgDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(hgGreenCtxCreate(&gctx[1], desc[1], (HGdevice)device, HG_GREEN_CTX_DEFAULT_STREAM));

  const int smCountA = resources[0].sm.smCount;
  const int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> streams = create_greenctx_stream_direct_dynamic(gctx);

  destroy_green_context(gctx[2]);

  std::vector<int64_t> vec = {
      streams[0],  // streamA
      streams[1],  // streamB
      (int64_t)smCountA,
      (int64_t)smCountB};

  return vec;
}
