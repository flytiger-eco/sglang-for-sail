#include <hggc.h>
#include <hggc_runtime.h>
// hggc_runtime.h does not pull in the host runtime API; hggcGetErrorString
// below comes from hggc_runtime_api.h. See include/utils.h for details.
#include <hggc_runtime_api.h>

#include <iostream>

#define CUDA_RT(call)                                                                                        \
  do {                                                                                                       \
    hggcError_t _status = (call);                                                                            \
    if (_status != hggcSuccess) {                                                                            \
      std::cerr << "ERROR: CUDA RT call \"" << #call << "\" in line " << __LINE__ << " of file " << __FILE__ \
                << " failed with " << hggcGetErrorString(_status) << std::endl;                              \
      TORCH_CHECK(                                                                                           \
          false,                                                                                             \
          c10::str(                                                                                          \
              "ERROR: CUDA RT call \"",                                                                      \
              #call,                                                                                         \
              "\" in line ",                                                                                 \
              __LINE__,                                                                                      \
              " of file ",                                                                                   \
              __FILE__,                                                                                      \
              " failed with ",                                                                               \
              hggcGetErrorString(_status)));                                                                 \
    }                                                                                                        \
  } while (0)

#define CUDA_DRV(call)                                                                                        \
  do {                                                                                                        \
    HGresult _status = (call);                                                                                \
    if (_status != HGGC_SUCCESS) {                                                                            \
      const char* err_str;                                                                                    \
      hgGetErrorString(_status, &err_str);                                                                    \
      std::cerr << "ERROR: CUDA DRV call \"" << #call << "\" in line " << __LINE__ << " of file " << __FILE__ \
                << " failed with " << err_str << std::endl;                                                   \
      TORCH_CHECK(                                                                                            \
          false,                                                                                              \
          c10::str(                                                                                           \
              "ERROR: CUDA DRV call \"",                                                                      \
              #call,                                                                                          \
              "\" in line ",                                                                                  \
              __LINE__,                                                                                       \
              " of file ",                                                                                    \
              __FILE__,                                                                                       \
              " failed with ",                                                                                \
              err_str));                                                                                      \
    }                                                                                                         \
  } while (0)
