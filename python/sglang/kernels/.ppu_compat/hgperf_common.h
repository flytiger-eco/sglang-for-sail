#ifndef HGPERF_COMMON_H
#define HGPERF_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__GNUC__) && defined(HGPA_SHARED_LIB)
#pragma GCC visibility push(default)
#if !defined(HGPW_LOCAL)
#define HGPW_LOCAL __attribute__((visibility("hidden")))
#endif
#else
#if !defined(HGPW_LOCAL)
#define HGPW_LOCAL
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HGPERF_HGPA_STATUS_DEFINED
#define HGPERF_HGPA_STATUS_DEFINED

typedef enum HGPA_Status {
  HGPA_STATUS_SUCCESS = 0,
  HGPA_STATUS_ERROR = 1,
  HGPA_STATUS_INTERNAL_ERROR = 2,
  HGPA_STATUS_NOT_INITIALIZED = 3,
  HGPA_STATUS_NOT_LOADED = 4,
  HGPA_STATUS_FUNCTION_NOT_FOUND = 5,
  HGPA_STATUS_NOT_SUPPORTED = 6,
  HGPA_STATUS_NOT_IMPLEMENTED = 7,
  HGPA_STATUS_INVALID_ARGUMENT = 8,
  HGPA_STATUS_INVALID_METRIC_ID = 9,
  HGPA_STATUS_DRIVER_NOT_LOADED = 10,
  HGPA_STATUS_OUT_OF_MEMORY = 11,
  HGPA_STATUS_INVALID_THREAD_STATE = 12,
  HGPA_STATUS_FAILED_CONTEXT_ALLOC = 13,
  HGPA_STATUS_UNSUPPORTED_GPU = 14,
  HGPA_STATUS_INSUFFICIENT_DRIVER_VERSION = 15,
  HGPA_STATUS_OBJECT_NOT_REGISTERED = 16,
  HGPA_STATUS_INSUFFICIENT_PRIVILEGE = 17,
  HGPA_STATUS_INVALID_CONTEXT_STATE = 18,
  HGPA_STATUS_INVALID_OBJECT_STATE = 19,
  HGPA_STATUS_RESOURCE_UNAVAILABLE = 20,
  HGPA_STATUS_DRIVER_LOADED_TOO_LATE = 21,
  HGPA_STATUS_INSUFFICIENT_SPACE = 22,
  HGPA_STATUS_OBJECT_MISMATCH = 23,
  HGPA_STATUS_VIRTUALIZED_DEVICE_NOT_SUPPORTED = 24,
  HGPA_STATUS_PROFILING_NOT_ALLOWED = 25,
  HGPA_STATUS__COUNT
} HGPA_Status;

static inline void HGPW_HGPAStatusToString(HGPA_Status status,
                                           const char **ppStatusStr,
                                           const char **ppCommentStr) {
  (void)status;
  (void)ppStatusStr;
  (void)ppCommentStr;
  fprintf(stderr, "HGPW_HGPAStatusToString is not supported.\n");
  exit(1);
}

#endif // HGPERF_HGPA_STATUS_DEFINED

#ifndef HGPERF_HGPA_ACTIVITY_KIND_DEFINED
#define HGPERF_HGPA_ACTIVITY_KIND_DEFINED

typedef enum HGPA_ActivityKind {
  HGPA_ACTIVITY_KIND_INVALID = 0,
  HGPA_ACTIVITY_KIND_PROFILER,
  HGPA_ACTIVITY_KIND_REALTIME_SAMPLED,
  HGPA_ACTIVITY_KIND_REALTIME_PROFILER,
  HGPA_ACTIVITY_KIND__COUNT
} HGPA_ActivityKind;

#endif // HGPERF_HGPA_ACTIVITY_KIND_DEFINED

#ifndef HGPERF_HGPA_BOOL_DEFINED
#define HGPERF_HGPA_BOOL_DEFINED
typedef uint8_t HGPA_Bool;
#endif // HGPERF_HGPA_BOOL_DEFINED

#ifndef HGPA_STRUCT_SIZE
#define HGPA_STRUCT_SIZE(type_, lastfield_)                                    \
  (offsetof(type_, lastfield_) + sizeof(((type_ *)0)->lastfield_))
#endif // HGPA_STRUCT_SIZE

#ifndef HGPW_FIELD_EXISTS
#define HGPW_FIELD_EXISTS(pParams_, name_)                                     \
  ((pParams_)->structSize >=                                                   \
   (size_t)((const uint8_t *)(&(pParams_)->name_) + sizeof(pParams_)->name_ -  \
            (const uint8_t *)(pParams_)))
#endif // HGPW_FIELD_EXISTS

#ifndef HGPERF_HGPA_GETPROCADDRESS_DEFINED
#define HGPERF_HGPA_GETPROCADDRESS_DEFINED

typedef HGPA_Status (*HGPA_GenericFn)(void);

static inline HGPA_GenericFn HGPA_GetProcAddress(const char *pFunctionName) {
  (void)pFunctionName;
  fprintf(stderr, "HGPA_GetProcAddress is not supported.\n");
  exit(1);
  return (HGPA_GenericFn)0; /* unreachable */
}

#endif

#ifndef HGPERF_HGPW_SETLIBRARYLOADPATHS_DEFINED
#define HGPERF_HGPW_SETLIBRARYLOADPATHS_DEFINED

typedef struct HGPW_SetLibraryLoadPaths_Params {
  size_t structSize;
  void *pPriv;
  size_t numPaths;
  const char **ppPaths;
} HGPW_SetLibraryLoadPaths_Params;
#define HGPW_SetLibraryLoadPaths_Params_STRUCT_SIZE                            \
  HGPA_STRUCT_SIZE(HGPW_SetLibraryLoadPaths_Params, ppPaths)

static inline HGPA_Status
HGPW_SetLibraryLoadPaths(HGPW_SetLibraryLoadPaths_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_SetLibraryLoadPaths is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_SetLibraryLoadPathsW_Params {
  size_t structSize;
  void *pPriv;
  size_t numPaths;
  const wchar_t **ppwPaths;
} HGPW_SetLibraryLoadPathsW_Params;
#define HGPW_SetLibraryLoadPathsW_Params_STRUCT_SIZE                           \
  HGPA_STRUCT_SIZE(HGPW_SetLibraryLoadPathsW_Params, ppwPaths)

static inline HGPA_Status
HGPW_SetLibraryLoadPathsW(HGPW_SetLibraryLoadPathsW_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_SetLibraryLoadPathsW is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

#endif

#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(HGPA_SHARED_LIB)
#pragma GCC visibility pop
#endif

#endif // HGPERF_COMMON_H
