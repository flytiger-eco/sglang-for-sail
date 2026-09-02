/// \file runtime.cuh
/// \brief Host-side CUDA runtime query helpers.
///
/// Thin wrappers around CUDA occupancy and device-property APIs with
/// automatic error checking via `RuntimeDeviceCheck`.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#ifndef USE_ROCM
#include <hggc_runtime.h>
#else
#include <hip/hip_runtime.h>
#ifndef hggcOccupancyMaxActiveBlocksPerMultiprocessor
#define hggcOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef hggcDeviceGetAttribute
#define hggcDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef hggcDevAttrMultiProcessorCount
#define hggcDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#endif
#ifndef hggcDevAttrComputeCapabilityMajor
#define hggcDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#endif
#ifndef hggcDevAttrComputeCapabilityMinor
#define hggcDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif
#ifndef compatibleRuntimeGetVersion
#define compatibleRuntimeGetVersion hipRuntimeGetVersion
#endif
#ifndef hggcOccupancyAvailableDynamicSMemPerBlock
inline hipError_t
hggcOccupancyAvailableDynamicSMemPerBlock(std::size_t* smem, const void* func, int num_blocks, int block_size) {
  // HIP does not expose this directly; return max shared mem as conservative estimate
  hipDeviceProp_t prop;
  int device;
  hipGetDevice(&device);
  hipGetDeviceProperties(&prop, device);
  *smem = prop.sharedMemPerBlock;
  return hipSuccess;
}
#endif
#endif

namespace sglang {

namespace host::runtime {

// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  RuntimeDeviceCheck(
      hggcOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id) -> uint32_t {
  int sm_count;
  RuntimeDeviceCheck(hggcDeviceGetAttribute(&sm_count, hggcDevAttrMultiProcessorCount, device_id));
  return static_cast<uint32_t>(sm_count);
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id) -> int {
  int cc_major;
  RuntimeDeviceCheck(hggcDeviceGetAttribute(&cc_major, hggcDevAttrComputeCapabilityMajor, device_id));
  return cc_major;
}

// Return the Minor compute capability for the given device
inline auto get_cc_minor(int device_id) -> int {
  int cc_minor;
  RuntimeDeviceCheck(hggcDeviceGetAttribute(&cc_minor, hggcDevAttrComputeCapabilityMinor, device_id));
  return cc_minor;
}

// Return the SM version (major * 10 + minor) for the given device
inline auto get_sm_version(int device_id) -> int {
  return get_cc_major(device_id) * 10 + get_cc_minor(device_id);
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
  RuntimeDeviceCheck(compatibleRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  RuntimeDeviceCheck(hggcOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}

}  // namespace host::runtime

}  // namespace sglang
