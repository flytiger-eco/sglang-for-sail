#ifndef HGPERF_HOST_H
#define HGPERF_HOST_H

#include "hgperf_common.h"
#include <stddef.h>
#include <stdint.h>

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

#ifndef HGPERF_HOST_API_DEFINED
#define HGPERF_HOST_API_DEFINED

typedef struct HGPW_InitializeHost_Params {
  size_t structSize;
  void *pPriv;
} HGPW_InitializeHost_Params;
#define HGPW_InitializeHost_Params_STRUCT_SIZE                                 \
  HGPA_STRUCT_SIZE(HGPW_InitializeHost_Params, pPriv)

static inline HGPA_Status
HGPW_InitializeHost(HGPW_InitializeHost_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_InitializeHost is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPA_RawMetricsConfig HGPA_RawMetricsConfig;

typedef struct HGPA_RawMetricRequest {
  size_t structSize;
  void *pPriv;
  const char *pMetricName;
  HGPA_Bool isolated;
  HGPA_Bool keepInstances;
} HGPA_RawMetricRequest;
#define HGPA_RAW_METRIC_REQUEST_STRUCT_SIZE                                    \
  HGPA_STRUCT_SIZE(HGPA_RawMetricRequest, keepInstances)

typedef struct HGPW_GetSupportedChipNames_Params {
  size_t structSize;
  void *pPriv;
  const char *const *ppChipNames;
  size_t numChipNames;
} HGPW_GetSupportedChipNames_Params;
#define HGPW_GetSupportedChipNames_Params_STRUCT_SIZE                          \
  HGPA_STRUCT_SIZE(HGPW_GetSupportedChipNames_Params, numChipNames)

static inline HGPA_Status
HGPW_GetSupportedChipNames(HGPW_GetSupportedChipNames_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_GetSupportedChipNames is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_Destroy_Params {
  size_t structSize;
  void *pPriv;
  HGPA_RawMetricsConfig *pRawMetricsConfig;
} HGPW_RawMetricsConfig_Destroy_Params;
#define HGPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE                       \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_Destroy_Params, pRawMetricsConfig)

static inline HGPA_Status
HGPW_RawMetricsConfig_Destroy(HGPW_RawMetricsConfig_Destroy_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_Destroy is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_SetCounterAvailability_Params {
  size_t structSize;
  void *pPriv;
  HGPA_RawMetricsConfig *pRawMetricsConfig;
  const uint8_t *pCounterAvailabilityImage;
} HGPW_RawMetricsConfig_SetCounterAvailability_Params;
#define HGPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE        \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_SetCounterAvailability_Params,        \
                   pCounterAvailabilityImage)

static inline HGPA_Status HGPW_RawMetricsConfig_SetCounterAvailability(
    HGPW_RawMetricsConfig_SetCounterAvailability_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawMetricsConfig_SetCounterAvailability is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_BeginPassGroup_Params {
  size_t structSize;
  void *pPriv;
  HGPA_RawMetricsConfig *pRawMetricsConfig;
  size_t maxPassCount;
} HGPW_RawMetricsConfig_BeginPassGroup_Params;
#define HGPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_BeginPassGroup_Params, maxPassCount)

static inline HGPA_Status HGPW_RawMetricsConfig_BeginPassGroup(
    HGPW_RawMetricsConfig_BeginPassGroup_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_BeginPassGroup is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_EndPassGroup_Params {
  size_t structSize;
  void *pPriv;
  HGPA_RawMetricsConfig *pRawMetricsConfig;
} HGPW_RawMetricsConfig_EndPassGroup_Params;
#define HGPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE                  \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_EndPassGroup_Params, pRawMetricsConfig)

static inline HGPA_Status HGPW_RawMetricsConfig_EndPassGroup(
    HGPW_RawMetricsConfig_EndPassGroup_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_EndPassGroup is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_GetNumMetrics_Params {
  size_t structSize;
  void *pPriv;
  const HGPA_RawMetricsConfig *pRawMetricsConfig;
  size_t numMetrics;
} HGPW_RawMetricsConfig_GetNumMetrics_Params;
#define HGPW_RawMetricsConfig_GetNumMetrics_Params_STRUCT_SIZE                 \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_GetNumMetrics_Params, numMetrics)

static inline HGPA_Status HGPW_RawMetricsConfig_GetNumMetrics(
    HGPW_RawMetricsConfig_GetNumMetrics_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_GetNumMetrics is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_GetMetricProperties_Params {
  size_t structSize;
  void *pPriv;
  const HGPA_RawMetricsConfig *pRawMetricsConfig;
  size_t metricIndex;
  const char *pMetricName;
  HGPA_Bool supportsPipelined;
  HGPA_Bool supportsIsolated;
} HGPW_RawMetricsConfig_GetMetricProperties_Params;
#define HGPW_RawMetricsConfig_GetMetricProperties_Params_STRUCT_SIZE           \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_GetMetricProperties_Params,           \
                   supportsIsolated)

static inline HGPA_Status HGPW_RawMetricsConfig_GetMetricProperties(
    HGPW_RawMetricsConfig_GetMetricProperties_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawMetricsConfig_GetMetricProperties is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_GetMetricProperties_V2_Params {
  size_t structSize;
  void *pPriv;
  const HGPA_RawMetricsConfig *pRawMetricsConfig;
  size_t metricIndex;
  const char *pMetricName;
} HGPW_RawMetricsConfig_GetMetricProperties_V2_Params;
#define HGPW_RawMetricsConfig_GetMetricProperties_V2_Params_STRUCT_SIZE        \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_GetMetricProperties_V2_Params,        \
                   pMetricName)

static inline HGPA_Status HGPW_RawMetricsConfig_GetMetricProperties_V2(
    HGPW_RawMetricsConfig_GetMetricProperties_V2_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawMetricsConfig_GetMetricProperties_V2 is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_AddMetrics_Params {
  size_t structSize;
  void *pPriv;
  HGPA_RawMetricsConfig *pRawMetricsConfig;
  const HGPA_RawMetricRequest *pRawMetricRequests;
  size_t numMetricRequests;
} HGPW_RawMetricsConfig_AddMetrics_Params;
#define HGPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE                    \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_AddMetrics_Params, numMetricRequests)

static inline HGPA_Status HGPW_RawMetricsConfig_AddMetrics(
    HGPW_RawMetricsConfig_AddMetrics_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_AddMetrics is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_IsAddMetricsPossible_Params {
  size_t structSize;
  void *pPriv;
  const HGPA_RawMetricsConfig *pRawMetricsConfig;
  const HGPA_RawMetricRequest *pRawMetricRequests;
  size_t numMetricRequests;
  HGPA_Bool isPossible;
} HGPW_RawMetricsConfig_IsAddMetricsPossible_Params;
#define HGPW_RawMetricsConfig_IsAddMetricsPossible_Params_STRUCT_SIZE          \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_IsAddMetricsPossible_Params,          \
                   isPossible)

static inline HGPA_Status HGPW_RawMetricsConfig_IsAddMetricsPossible(
    HGPW_RawMetricsConfig_IsAddMetricsPossible_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawMetricsConfig_IsAddMetricsPossible is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_GenerateConfigImage_Params {
  size_t structSize;
  void *pPriv;
  HGPA_RawMetricsConfig *pRawMetricsConfig;
  HGPA_Bool mergeAllPassGroups;
} HGPW_RawMetricsConfig_GenerateConfigImage_Params;
#define HGPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE           \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_GenerateConfigImage_Params,           \
                   mergeAllPassGroups)

static inline HGPA_Status HGPW_RawMetricsConfig_GenerateConfigImage(
    HGPW_RawMetricsConfig_GenerateConfigImage_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawMetricsConfig_GenerateConfigImage is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_GetConfigImage_Params {
  size_t structSize;
  void *pPriv;
  const HGPA_RawMetricsConfig *pRawMetricsConfig;
  size_t bytesAllocated;
  uint8_t *pBuffer;
  size_t bytesCopied;
} HGPW_RawMetricsConfig_GetConfigImage_Params;
#define HGPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_GetConfigImage_Params, bytesCopied)

static inline HGPA_Status HGPW_RawMetricsConfig_GetConfigImage(
    HGPW_RawMetricsConfig_GetConfigImage_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_GetConfigImage is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_GetNumPasses_Params {
  size_t structSize;
  void *pPriv;
  const HGPA_RawMetricsConfig *pRawMetricsConfig;
  size_t numPipelinedPasses;
  size_t numIsolatedPasses;
} HGPW_RawMetricsConfig_GetNumPasses_Params;
#define HGPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE                  \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_GetNumPasses_Params, numIsolatedPasses)

static inline HGPA_Status HGPW_RawMetricsConfig_GetNumPasses(
    HGPW_RawMetricsConfig_GetNumPasses_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_GetNumPasses is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawMetricsConfig_GetNumPasses_V2_Params {
  size_t structSize;
  void *pPriv;
  const HGPA_RawMetricsConfig *pRawMetricsConfig;
  size_t numPasses;
} HGPW_RawMetricsConfig_GetNumPasses_V2_Params;
#define HGPW_RawMetricsConfig_GetNumPasses_V2_Params_STRUCT_SIZE               \
  HGPA_STRUCT_SIZE(HGPW_RawMetricsConfig_GetNumPasses_V2_Params, numPasses)

static inline HGPA_Status HGPW_RawMetricsConfig_GetNumPasses_V2(
    HGPW_RawMetricsConfig_GetNumPasses_V2_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawMetricsConfig_GetNumPasses_V2 is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig HGPW_RawCounterConfig;

typedef enum HGPW_RawCounterDomain {
  HGPW_RAW_COUNTER_DOMAIN_INVALID = 0,
  HGPW_RAW_COUNTER_DOMAIN_TRACE = 1,
  HGPW_RAW_COUNTER_DOMAIN_GPU_SASS = 2,
  HGPW_RAW_COUNTER_DOMAIN_GPU_SM_B = 3,
  HGPW_RAW_COUNTER_DOMAIN_GPU_SM_C = 4,
  HGPW_RAW_COUNTER_DOMAIN_GPU_CTC = 6,
  HGPW_RAW_COUNTER_DOMAIN_GPU_FBPA = 8,
  HGPW_RAW_COUNTER_DOMAIN_GPU_FBSP = 9,
  HGPW_RAW_COUNTER_DOMAIN_GPU_FE_A = 10,
  HGPW_RAW_COUNTER_DOMAIN_GPU_FE_B = 11,
  HGPW_RAW_COUNTER_DOMAIN_GPU_FE_C = 12,
  HGPW_RAW_COUNTER_DOMAIN_GPU_GPC_A = 13,
  HGPW_RAW_COUNTER_DOMAIN_GPU_GPC_B = 14,
  HGPW_RAW_COUNTER_DOMAIN_GPU_GPC_C = 15,
  HGPW_RAW_COUNTER_DOMAIN_GPU_HOST = 16,
  HGPW_RAW_COUNTER_DOMAIN_GPU_HUB = 17,
  HGPW_RAW_COUNTER_DOMAIN_GPU_HUB_A = 18,
  HGPW_RAW_COUNTER_DOMAIN_GPU_HUB_B = 19,
  HGPW_RAW_COUNTER_DOMAIN_GPU_HUB_C = 20,
  HGPW_RAW_COUNTER_DOMAIN_GPU_LTS = 23,
  HGPW_RAW_COUNTER_DOMAIN_GPU_HGLRX = 26,
  HGPW_RAW_COUNTER_DOMAIN_GPU_HGLTX = 28,
  HGPW_RAW_COUNTER_DOMAIN_GPU_PCI = 29,
  HGPW_RAW_COUNTER_DOMAIN_GPU_PWR = 30,
  HGPW_RAW_COUNTER_DOMAIN_GPU_ROP = 31,
  HGPW_RAW_COUNTER_DOMAIN_GPU_SM_A = 32,
  HGPW_RAW_COUNTER_DOMAIN_GPU_TPC = 33,
  HGPW_RAW_COUNTER_DOMAIN_SOC_MCC = 38,
  HGPW_RAW_COUNTER_DOMAIN_SOC_NVENC = 48,
  HGPW_RAW_COUNTER_DOMAIN_SOC_OFA = 50,
  HGPW_RAW_COUNTER_DOMAIN_SOC_VIC = 53,
  HGPW_RAW_COUNTER_DOMAIN_SOC_DLA = 57,
  HGPW_RAW_COUNTER_DOMAIN_SOC_PVA_A = 58,
  HGPW_RAW_COUNTER_DOMAIN_SOC_PVA_B = 59,
  HGPW_RAW_COUNTER_DOMAIN_GPU_CTC_A = 62,
  HGPW_RAW_COUNTER_DOMAIN_GPU_CTC_B = 63,
  HGPW_RAW_COUNTER_DOMAIN_GPU_SYSLTS = 74,
  HGPW_RAW_COUNTER_DOMAIN_GPU_FE_D = 75,
  HGPW_RAW_COUNTER_DOMAIN_GPU_FBP_B = 78,
  HGPW_RAW_COUNTER_DOMAIN_GPU_SYSLTS_B = 79,
  HGPW_RAW_COUNTER_DOMAIN_GPU_TPC_B = 80
} HGPW_RawCounterDomain;

typedef enum HGPW_RawCounterDomain_CooperativeDomainGroup {
  HGPW_RAW_COUNTER_DOMAIN_CDG_GPU_GPC_A_GPC_B = 1048576,
  HGPW_RAW_COUNTER_DOMAIN_CDG_GPU_TPC_SM_A = 1048577,
  HGPW_RAW_COUNTER_DOMAIN_CDG_GPU_SM_B_SM_A = 1048578,
  HGPW_RAW_COUNTER_DOMAIN_CDG_GPU_SM_B_TPC = 1048579,
  HGPW_RAW_COUNTER_DOMAIN_CDG_GPU_SM_C_SM_A = 1048580
} HGPW_RawCounterDomain_CooperativeDomainGroup;

typedef struct HGPW_RawCounterConfig_RawCounterDomainToString_Params {
  size_t structSize;
  void *pPriv;
  uint32_t domain;
  const char *pDomainName;
} HGPW_RawCounterConfig_RawCounterDomainToString_Params;
#define HGPW_RawCounterConfig_RawCounterDomainToString_Params_STRUCT_SIZE      \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_RawCounterDomainToString_Params,      \
                   pDomainName)

static inline HGPA_Status HGPW_RawCounterConfig_RawCounterDomainToString(
    HGPW_RawCounterConfig_RawCounterDomainToString_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_RawCounterDomainToString is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_StringToRawCounterDomain_Params {
  size_t structSize;
  void *pPriv;
  const char *pDomainName;
  uint32_t domain;
} HGPW_RawCounterConfig_StringToRawCounterDomain_Params;
#define HGPW_RawCounterConfig_StringToRawCounterDomain_Params_STRUCT_SIZE      \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_StringToRawCounterDomain_Params,      \
                   domain)

static inline HGPA_Status HGPW_RawCounterConfig_StringToRawCounterDomain(
    HGPW_RawCounterConfig_StringToRawCounterDomain_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_StringToRawCounterDomain is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GetSupportedChipNames_Params {
  size_t structSize;
  void *pPriv;
  const char *const *ppChipNames;
  size_t numChipNames;
} HGPW_RawCounterConfig_GetSupportedChipNames_Params;
#define HGPW_RawCounterConfig_GetSupportedChipNames_Params_STRUCT_SIZE         \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_GetSupportedChipNames_Params,         \
                   numChipNames)

static inline HGPA_Status HGPW_RawCounterConfig_GetSupportedChipNames(
    HGPW_RawCounterConfig_GetSupportedChipNames_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_GetSupportedChipNames is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_Destroy_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_RawCounterConfig *pRawCounterConfig;
} HGPW_RawCounterConfig_Destroy_Params;
#define HGPW_RawCounterConfig_Destroy_Params_STRUCT_SIZE                       \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_Destroy_Params, pRawCounterConfig)

static inline HGPA_Status
HGPW_RawCounterConfig_Destroy(HGPW_RawCounterConfig_Destroy_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_Destroy is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_SetCounterAvailability_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_RawCounterConfig *pRawCounterConfig;
  const uint8_t *pCounterAvailabilityImage;
} HGPW_RawCounterConfig_SetCounterAvailability_Params;
#define HGPW_RawCounterConfig_SetCounterAvailability_Params_STRUCT_SIZE        \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_SetCounterAvailability_Params,        \
                   pCounterAvailabilityImage)

static inline HGPA_Status HGPW_RawCounterConfig_SetCounterAvailability(
    HGPW_RawCounterConfig_SetCounterAvailability_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_SetCounterAvailability is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t numAvailableDomains;
  HGPW_RawCounterDomain *pAvailableDomains;
} HGPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params;
#define HGPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params_STRUCT_SIZE \
  HGPA_STRUCT_SIZE(                                                               \
      HGPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params,              \
      pAvailableDomains)

static inline HGPA_Status
HGPW_RawCounterConfig_GetAllAvailableRawCounterDomains(
    HGPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_GetAllAvailableRawCounterDomains is "
                  "not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct
    HGPW_RawCounterConfig_GetAllAvailableCooperativeDomainGroups_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t numAvailableDomains;
  uint32_t *pAvailableDomains;
} HGPW_RawCounterConfig_GetAllAvailableCooperativeDomainGroups_Params;
#define HGPW_RawCounterConfig_GetAllAvailableCooperativeDomainGroups_Params_STRUCT_SIZE \
  HGPA_STRUCT_SIZE(                                                                     \
      HGPW_RawCounterConfig_GetAllAvailableCooperativeDomainGroups_Params,              \
      pAvailableDomains)

static inline HGPA_Status
HGPW_RawCounterConfig_GetAllAvailableCooperativeDomainGroups(
    HGPW_RawCounterConfig_GetAllAvailableCooperativeDomainGroups_Params
        *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_"
                  "GetAllAvailableCooperativeDomainGroups is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_IsCooperativeDomainGroup_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  uint32_t domain;
  HGPA_Bool isCdg;
} HGPW_RawCounterConfig_IsCooperativeDomainGroup_Params;
#define HGPW_RawCounterConfig_IsCooperativeDomainGroup_Params_STRUCT_SIZE      \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_IsCooperativeDomainGroup_Params, isCdg)

static inline HGPA_Status HGPW_RawCounterConfig_IsCooperativeDomainGroup(
    HGPW_RawCounterConfig_IsCooperativeDomainGroup_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_IsCooperativeDomainGroup is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct
    HGPW_RawCounterConfig_CooperativeDomainGroup_GetMemberDomains_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  uint32_t domain;
  size_t numMemberDomains;
  HGPW_RawCounterDomain *pMemberDomains;
} HGPW_RawCounterConfig_CooperativeDomainGroup_GetMemberDomains_Params;
#define HGPW_RawCounterConfig_CooperativeDomainGroup_GetMemberDomains_Params_STRUCT_SIZE \
  HGPA_STRUCT_SIZE(                                                                      \
      HGPW_RawCounterConfig_CooperativeDomainGroup_GetMemberDomains_Params,              \
      pMemberDomains)

static inline HGPA_Status
HGPW_RawCounterConfig_CooperativeDomainGroup_GetMemberDomains(
    HGPW_RawCounterConfig_CooperativeDomainGroup_GetMemberDomains_Params
        *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_CooperativeDomainGroup_"
                  "GetMemberDomains is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_BeginPassGroup_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t numDomains;
  const HGPW_RawCounterDomain *pDomains;
} HGPW_RawCounterConfig_BeginPassGroup_Params;
#define HGPW_RawCounterConfig_BeginPassGroup_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_BeginPassGroup_Params, pDomains)

static inline HGPA_Status HGPW_RawCounterConfig_BeginPassGroup(
    HGPW_RawCounterConfig_BeginPassGroup_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_BeginPassGroup is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_EndPassGroup_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t numDomains;
  const HGPW_RawCounterDomain *pDomains;
} HGPW_RawCounterConfig_EndPassGroup_Params;
#define HGPW_RawCounterConfig_EndPassGroup_Params_STRUCT_SIZE                  \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_EndPassGroup_Params, pDomains)

static inline HGPA_Status HGPW_RawCounterConfig_EndPassGroup(
    HGPW_RawCounterConfig_EndPassGroup_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_EndPassGroup is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GetNumRawCounters_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  uint32_t domain;
  size_t numRawCounters;
} HGPW_RawCounterConfig_GetNumRawCounters_Params;
#define HGPW_RawCounterConfig_GetNumRawCounters_Params_STRUCT_SIZE             \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_GetNumRawCounters_Params,             \
                   numRawCounters)

static inline HGPA_Status HGPW_RawCounterConfig_GetNumRawCounters(
    HGPW_RawCounterConfig_GetNumRawCounters_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_GetNumRawCounters is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GetRawCounterName_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  uint32_t domain;
  size_t rawCounterIndex;
  const char *pRawCounterName;
} HGPW_RawCounterConfig_GetRawCounterName_Params;
#define HGPW_RawCounterConfig_GetRawCounterName_Params_STRUCT_SIZE             \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_GetRawCounterName_Params,             \
                   pRawCounterName)

static inline HGPA_Status HGPW_RawCounterConfig_GetRawCounterName(
    HGPW_RawCounterConfig_GetRawCounterName_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_GetRawCounterName is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GetRawCounterProperties_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  const char *pRawCounterName;
  size_t numSupportedDomains;
  HGPW_RawCounterDomain *pSupportedDomains;
  size_t numSupportedCdgDomains;
  uint32_t *pSupportedCdgDomains;
} HGPW_RawCounterConfig_GetRawCounterProperties_Params;
#define HGPW_RawCounterConfig_GetRawCounterProperties_Params_STRUCT_SIZE       \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_GetRawCounterProperties_Params,       \
                   pSupportedCdgDomains)

static inline HGPA_Status HGPW_RawCounterConfig_GetRawCounterProperties(
    HGPW_RawCounterConfig_GetRawCounterProperties_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_GetRawCounterProperties is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterRequest {
  void *pPriv;
  const char *pRawCounterName;
  uint32_t domain;
  HGPA_Bool keepInstances;
} HGPW_RawCounterRequest;
#define HGPW_RAW_COUNTER_REQUEST_STRUCT_SIZE                                   \
  HGPA_STRUCT_SIZE(HGPW_RawCounterRequest, keepInstances)

typedef struct HGPW_RawCounterConfig_AddRawCounters_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t rawCounterRequestStructSize;
  size_t numRawCounterRequests;
  const HGPW_RawCounterRequest *pRawCounterRequests;
} HGPW_RawCounterConfig_AddRawCounters_Params;
#define HGPW_RawCounterConfig_AddRawCounters_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_AddRawCounters_Params,                \
                   pRawCounterRequests)

static inline HGPA_Status HGPW_RawCounterConfig_AddRawCounters(
    HGPW_RawCounterConfig_AddRawCounters_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_AddRawCounters is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_AreRawCountersSchedulable_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t rawCounterRequestStructSize;
  size_t numRawCounterRequests;
  const HGPW_RawCounterRequest *pRawCounterRequests;
  HGPA_Bool disallowAddingExtraPasses;
  HGPA_Bool schedulable;
} HGPW_RawCounterConfig_AreRawCountersSchedulable_Params;
#define HGPW_RawCounterConfig_AreRawCountersSchedulable_Params_STRUCT_SIZE     \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_AreRawCountersSchedulable_Params,     \
                   schedulable)

static inline HGPA_Status HGPW_RawCounterConfig_AreRawCountersSchedulable(
    HGPW_RawCounterConfig_AreRawCountersSchedulable_Params *pParams) {
  (void)pParams;
  fprintf(
      stderr,
      "HGPW_RawCounterConfig_AreRawCountersSchedulable is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_MergePassGroups_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_RawCounterConfig *pRawCounterConfig;
} HGPW_RawCounterConfig_MergePassGroups_Params;
#define HGPW_RawCounterConfig_MergePassGroups_Params_STRUCT_SIZE               \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_MergePassGroups_Params,               \
                   pRawCounterConfig)

static inline HGPA_Status HGPW_RawCounterConfig_MergePassGroups(
    HGPW_RawCounterConfig_MergePassGroups_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_MergePassGroups is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GenerateConfigImage_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_RawCounterConfig *pRawCounterConfig;
} HGPW_RawCounterConfig_GenerateConfigImage_Params;
#define HGPW_RawCounterConfig_GenerateConfigImage_Params_STRUCT_SIZE           \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_GenerateConfigImage_Params,           \
                   pRawCounterConfig)

static inline HGPA_Status HGPW_RawCounterConfig_GenerateConfigImage(
    HGPW_RawCounterConfig_GenerateConfigImage_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_RawCounterConfig_GenerateConfigImage is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GetConfigImage_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t bytesAllocated;
  uint8_t *pBuffer;
  size_t bytesCopied;
} HGPW_RawCounterConfig_GetConfigImage_Params;
#define HGPW_RawCounterConfig_GetConfigImage_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_GetConfigImage_Params, bytesCopied)

static inline HGPA_Status HGPW_RawCounterConfig_GetConfigImage(
    HGPW_RawCounterConfig_GetConfigImage_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_GetConfigImage is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_RawCounterConfig_GetNumPasses_Params {
  size_t structSize;
  void *pPriv;
  const struct HGPW_RawCounterConfig *pRawCounterConfig;
  size_t numPasses;
} HGPW_RawCounterConfig_GetNumPasses_Params;
#define HGPW_RawCounterConfig_GetNumPasses_Params_STRUCT_SIZE                  \
  HGPA_STRUCT_SIZE(HGPW_RawCounterConfig_GetNumPasses_Params, numPasses)

static inline HGPA_Status HGPW_RawCounterConfig_GetNumPasses(
    HGPW_RawCounterConfig_GetNumPasses_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_RawCounterConfig_GetNumPasses is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_Config_GetRawCounterInfo_Params {
  size_t structSize;
  void *pPriv;
  const uint8_t *pConfig;
  size_t configSize;
  const char *pRawCounterName;
  size_t *pPassIndices;
  size_t numPassIndices;
} HGPW_Config_GetRawCounterInfo_Params;
#define HGPW_Config_GetRawCounterInfo_Params_STRUCT_SIZE                       \
  HGPA_STRUCT_SIZE(HGPW_Config_GetRawCounterInfo_Params, numPassIndices)

static inline HGPA_Status
HGPW_Config_GetRawCounterInfo(HGPW_Config_GetRawCounterInfo_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_Config_GetRawCounterInfo is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_Config_GetRawCounters_Params {
  size_t structSize;
  void *pPriv;
  const uint8_t *pConfig;
  size_t configSize;
  size_t passIndex;
  const char **ppRawCounterNames;
  size_t numRawCounters;
} HGPW_Config_GetRawCounters_Params;
#define HGPW_Config_GetRawCounters_Params_STRUCT_SIZE                          \
  HGPA_STRUCT_SIZE(HGPW_Config_GetRawCounters_Params, numRawCounters)

static inline HGPA_Status
HGPW_Config_GetRawCounters(HGPW_Config_GetRawCounters_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_Config_GetRawCounters is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params {
  size_t structSize;
  void *pPriv;
  const uint8_t *pConfig;
  size_t configSize;
  size_t sampleSize;
} HGPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params;
#define HGPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params_STRUCT_SIZE \
  HGPA_STRUCT_SIZE(                                                              \
      HGPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params,              \
      sampleSize)

static inline HGPA_Status HGPW_PeriodicSampler_Config_GetSocEstimatedSampleSize(
    HGPW_PeriodicSampler_Config_GetSocEstimatedSampleSize_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_PeriodicSampler_Config_GetSocEstimatedSampleSize is "
                  "not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params {
  size_t structSize;
  void *pPriv;
  const uint8_t *pConfig;
  size_t configSize;
  size_t sampleSize;
} HGPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params;
#define HGPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params_STRUCT_SIZE \
  HGPA_STRUCT_SIZE(                                                              \
      HGPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params,              \
      sampleSize)

static inline HGPA_Status HGPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize(
    HGPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_PeriodicSampler_Config_GetGpuEstimatedSampleSize is "
                  "not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator HGPW_MetricsEvaluator;

#ifndef HGPW_DIM_UNIT_DEFINED
#define HGPW_DIM_UNIT_DEFINED
typedef enum HGPW_DimUnitName {
  HGPW_DIM_UNIT_INVALID = 3518299157,
  HGPW_DIM_UNIT_UNITLESS = 2126137902,
  HGPW_DIM_UNIT_ATTRIBUTES = 3776338729,
  HGPW_DIM_UNIT_BRANCH_TARGETS = 3746602690,
  HGPW_DIM_UNIT_BYTES = 3797850191,
  HGPW_DIM_UNIT_CGAS = 2067066462,
  HGPW_DIM_UNIT_CROP_SUBPACKETS = 175697354,
  HGPW_DIM_UNIT_CTAS = 1960564139,
  HGPW_DIM_UNIT_CTC_CYCLES = 2224883873,
  HGPW_DIM_UNIT_DRAM_CYCLES = 2650981327,
  HGPW_DIM_UNIT_FBP_CYCLES = 1785238957,
  HGPW_DIM_UNIT_FE_OPS = 2919159083,
  HGPW_DIM_UNIT_GCC_REQUESTS = 2855402624,
  HGPW_DIM_UNIT_GPC_CYCLES = 1222631184,
  HGPW_DIM_UNIT_IDC_REQUESTS = 2012649669,
  HGPW_DIM_UNIT_INSTRUCTIONS = 1418625543,
  HGPW_DIM_UNIT_KILOBYTES = 1335980302,
  HGPW_DIM_UNIT_L1DATA_BANK_ACCESSES = 1479493682,
  HGPW_DIM_UNIT_L1DATA_BANK_CONFLICTS = 3433170787,
  HGPW_DIM_UNIT_L1TEX_LINES = 1899735838,
  HGPW_DIM_UNIT_L1TEX_REQUESTS = 1306473767,
  HGPW_DIM_UNIT_L1TEX_TAGS = 26573010,
  HGPW_DIM_UNIT_L1TEX_WAVEFRONTS = 129373765,
  HGPW_DIM_UNIT_L2_REQUESTS = 1143695106,
  HGPW_DIM_UNIT_L2_SECTORS = 3424101564,
  HGPW_DIM_UNIT_L2_TAGS = 3755612781,
  HGPW_DIM_UNIT_LRC_REQUESTS = 2280914327,
  HGPW_DIM_UNIT_LRC_SECTORS = 7212034,
  HGPW_DIM_UNIT_MATH_OPS = 653103099,
  HGPW_DIM_UNIT_MCC_CYCLES = 1826685787,
  HGPW_DIM_UNIT_NANOSECONDS = 3047500672,
  HGPW_DIM_UNIT_NVDEC_CYCLES = 292063189,
  HGPW_DIM_UNIT_NVDLA_CYCLES = 3374059789,
  HGPW_DIM_UNIT_NVENC_CYCLES = 2267185244,
  HGPW_DIM_UNIT_HGLRX_CYCLES = 4059934930,
  HGPW_DIM_UNIT_HGLTX_CYCLES = 1814350488,
  HGPW_DIM_UNIT_OFA_CYCLES = 4290210307,
  HGPW_DIM_UNIT_PCIE_CYCLES = 1230450943,
  HGPW_DIM_UNIT_PCIE_REQUESTS = 4131188227,
  HGPW_DIM_UNIT_PERCENT = 1284354694,
  HGPW_DIM_UNIT_PIXELS = 4227616663,
  HGPW_DIM_UNIT_PIXEL_SHADER_BARRIERS = 3705502518,
  HGPW_DIM_UNIT_PLANE_EQUATIONS = 1945793602,
  HGPW_DIM_UNIT_PRIMITIVES = 2373084002,
  HGPW_DIM_UNIT_PVAVPU_CYCLES = 2238259366,
  HGPW_DIM_UNIT_PVA_CYCLES = 202044173,
  HGPW_DIM_UNIT_QUADS = 1539753497,
  HGPW_DIM_UNIT_RAYS = 2989263609,
  HGPW_DIM_UNIT_REGISTERS = 2837260947,
  HGPW_DIM_UNIT_RF_QUANTA = 2083886015,
  HGPW_DIM_UNIT_SAMPLES = 746046551,
  HGPW_DIM_UNIT_SECONDS = 1164825258,
  HGPW_DIM_UNIT_SYSL2_REQUESTS = 2165109286,
  HGPW_DIM_UNIT_SYSL2_SECTORS = 2268734175,
  HGPW_DIM_UNIT_SYSL2_TAGS = 3308651352,
  HGPW_DIM_UNIT_SYSLRC_REQUESTS = 3328245480,
  HGPW_DIM_UNIT_SYSLRC_SECTORS = 1190477493,
  HGPW_DIM_UNIT_SYS_CYCLES = 3310821688,
  HGPW_DIM_UNIT_TEXELS = 1293214069,
  HGPW_DIM_UNIT_THREADS = 164261907,
  HGPW_DIM_UNIT_TMEM_ACCESSES = 3742902067,
  HGPW_DIM_UNIT_VERTICES = 1873662209,
  HGPW_DIM_UNIT_VIC_CYCLES = 103143588,
  HGPW_DIM_UNIT_WARPS = 97951949,
  HGPW_DIM_UNIT_WORKIDS = 1971113483,
  HGPW_DIM_UNIT_WORKLOADS = 1728142656,
  HGPW_DIM_UNIT_ZROP_SUBPACKETS = 1657246246,
  HGPW_DIM_UNIT_Z_OCCLUDERS = 2718134770
} HGPW_DimUnitName;
#endif // HGPW_DIM_UNIT_DEFINED

#ifndef HGPW_HW_UNIT_DEFINED
#define HGPW_HW_UNIT_DEFINED
typedef enum HGPW_HwUnit {
  HGPW_HW_UNIT_INVALID = 3498035701,
  HGPW_HW_UNIT_CROP = 2872137846,
  HGPW_HW_UNIT_CTC = 4123164475,
  HGPW_HW_UNIT_DRAM = 1662616918,
  HGPW_HW_UNIT_DRAMC = 1401232876,
  HGPW_HW_UNIT_FBP = 2947194306,
  HGPW_HW_UNIT_FBPA = 690045803,
  HGPW_HW_UNIT_FE = 2204924321,
  HGPW_HW_UNIT_GPC = 1911735839,
  HGPW_HW_UNIT_GPU = 1014363534,
  HGPW_HW_UNIT_GR = 2933618517,
  HGPW_HW_UNIT_IDC = 842765289,
  HGPW_HW_UNIT_L1TEX = 893940957,
  HGPW_HW_UNIT_LRC = 4004756136,
  HGPW_HW_UNIT_LTS = 2333266697,
  HGPW_HW_UNIT_MCC = 3980130194,
  HGPW_HW_UNIT_NVDLA = 4201167892,
  HGPW_HW_UNIT_NVENC = 207708260,
  HGPW_HW_UNIT_HGLRX = 3091684901,
  HGPW_HW_UNIT_HGLTX = 869679659,
  HGPW_HW_UNIT_OFA = 70307371,
  HGPW_HW_UNIT_PCIE = 3433264174,
  HGPW_HW_UNIT_PDA = 345193251,
  HGPW_HW_UNIT_PES = 804128425,
  HGPW_HW_UNIT_PROP = 3339255507,
  HGPW_HW_UNIT_PVA = 2565499490,
  HGPW_HW_UNIT_PVAVPU = 1656645655,
  HGPW_HW_UNIT_RASTER = 187932504,
  HGPW_HW_UNIT_SM = 724224710,
  HGPW_HW_UNIT_SMSP = 2837616917,
  HGPW_HW_UNIT_SYS = 768990063,
  HGPW_HW_UNIT_SYSLRC = 3247626950,
  HGPW_HW_UNIT_SYSLTS = 4137740217,
  HGPW_HW_UNIT_TPC = 1889024613,
  HGPW_HW_UNIT_VAF = 753670509,
  HGPW_HW_UNIT_VIC = 322439594,
  HGPW_HW_UNIT_VPC = 275561583,
  HGPW_HW_UNIT_ZCULL = 2401248356,
  HGPW_HW_UNIT_ZROP = 979500456
} HGPW_HwUnit;
#endif // HGPW_HW_UNIT_DEFINED

typedef enum HGPW_RollupOp {
  HGPW_ROLLUP_OP_AVG = 0,
  HGPW_ROLLUP_OP_MAX,
  HGPW_ROLLUP_OP_MIN,
  HGPW_ROLLUP_OP_SUM,
  HGPW_ROLLUP_OP__COUNT
} HGPW_RollupOp;

typedef enum HGPW_MetricType {
  HGPW_METRIC_TYPE_COUNTER = 0,
  HGPW_METRIC_TYPE_RATIO,
  HGPW_METRIC_TYPE_THROUGHPUT,
  HGPW_METRIC_TYPE__COUNT
} HGPW_MetricType;

typedef enum HGPW_Submetric {
  HGPW_SUBMETRIC_NONE = 0,
  HGPW_SUBMETRIC_PEAK_SUSTAINED = 1,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE = 2,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE_PER_SECOND = 3,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED = 4,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED_PER_SECOND = 5,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_FRAME = 6,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_FRAME_PER_SECOND = 7,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_REGION = 8,
  HGPW_SUBMETRIC_PEAK_SUSTAINED_REGION_PER_SECOND = 9,
  HGPW_SUBMETRIC_PER_CYCLE_ACTIVE = 10,
  HGPW_SUBMETRIC_PER_CYCLE_ELAPSED = 11,
  HGPW_SUBMETRIC_PER_CYCLE_IN_FRAME = 12,
  HGPW_SUBMETRIC_PER_CYCLE_IN_REGION = 13,
  HGPW_SUBMETRIC_PER_SECOND = 14,
  HGPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ACTIVE = 15,
  HGPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED = 16,
  HGPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_FRAME = 17,
  HGPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_REGION = 18,
  HGPW_SUBMETRIC_MAX_RATE = 19,
  HGPW_SUBMETRIC_PCT = 20,
  HGPW_SUBMETRIC_RATIO = 21,
  HGPW_SUBMETRIC__COUNT
} HGPW_Submetric;

typedef struct HGPW_MetricEvalRequest {
  size_t metricIndex;
  uint8_t metricType;
  uint8_t rollupOp;
  uint16_t submetric;
} HGPW_MetricEvalRequest;
#define HGPW_MetricEvalRequest_STRUCT_SIZE                                     \
  HGPA_STRUCT_SIZE(HGPW_MetricEvalRequest, submetric)

typedef struct HGPW_DimUnitFactor {
  uint32_t dimUnit;
  int8_t exponent;
} HGPW_DimUnitFactor;
#define HGPW_DimUnitFactor_STRUCT_SIZE                                         \
  HGPA_STRUCT_SIZE(HGPW_DimUnitFactor, exponent)

typedef struct HGPW_MetricsEvaluator_Destroy_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
} HGPW_MetricsEvaluator_Destroy_Params;
#define HGPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE                       \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_Destroy_Params, pMetricsEvaluator)

static inline HGPA_Status
HGPW_MetricsEvaluator_Destroy(HGPW_MetricsEvaluator_Destroy_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_MetricsEvaluator_Destroy is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetMetricNames_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  uint8_t metricType;
  const char *pMetricNames;
  const size_t *pMetricNameBeginIndices;
  size_t numMetrics;
} HGPW_MetricsEvaluator_GetMetricNames_Params;
#define HGPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetMetricNames_Params, numMetrics)

static inline HGPA_Status HGPW_MetricsEvaluator_GetMetricNames(
    HGPW_MetricsEvaluator_GetMetricNames_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_MetricsEvaluator_GetMetricNames is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetMetricTypeAndIndex_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  const char *pMetricName;
  uint8_t metricType;
  size_t metricIndex;
} HGPW_MetricsEvaluator_GetMetricTypeAndIndex_Params;
#define HGPW_MetricsEvaluator_GetMetricTypeAndIndex_Params_STRUCT_SIZE         \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetMetricTypeAndIndex_Params,         \
                   metricIndex)

static inline HGPA_Status HGPW_MetricsEvaluator_GetMetricTypeAndIndex(
    HGPW_MetricsEvaluator_GetMetricTypeAndIndex_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_GetMetricTypeAndIndex is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct
    HGPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  const char *pMetricName;
  struct HGPW_MetricEvalRequest *pMetricEvalRequest;
  size_t metricEvalRequestStructSize;
} HGPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params;
#define HGPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE \
  HGPA_STRUCT_SIZE(                                                                   \
      HGPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params,              \
      metricEvalRequestStructSize)

static inline HGPA_Status
HGPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
    HGPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
        *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest "
                  "is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_HwUnitToString_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  uint32_t hwUnit;
  const char *pHwUnitName;
} HGPW_MetricsEvaluator_HwUnitToString_Params;
#define HGPW_MetricsEvaluator_HwUnitToString_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_HwUnitToString_Params, pHwUnitName)

static inline HGPA_Status HGPW_MetricsEvaluator_HwUnitToString(
    HGPW_MetricsEvaluator_HwUnitToString_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_MetricsEvaluator_HwUnitToString is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetCounterProperties_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  size_t counterIndex;
  const char *pDescription;
  uint32_t hwUnit;
} HGPW_MetricsEvaluator_GetCounterProperties_Params;
#define HGPW_MetricsEvaluator_GetCounterProperties_Params_STRUCT_SIZE          \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetCounterProperties_Params, hwUnit)

static inline HGPA_Status HGPW_MetricsEvaluator_GetCounterProperties(
    HGPW_MetricsEvaluator_GetCounterProperties_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_GetCounterProperties is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetRatioMetricProperties_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  size_t ratioMetricIndex;
  const char *pDescription;
  uint64_t hwUnit;
} HGPW_MetricsEvaluator_GetRatioMetricProperties_Params;
#define HGPW_MetricsEvaluator_GetRatioMetricProperties_Params_STRUCT_SIZE      \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetRatioMetricProperties_Params,      \
                   hwUnit)

static inline HGPA_Status HGPW_MetricsEvaluator_GetRatioMetricProperties(
    HGPW_MetricsEvaluator_GetRatioMetricProperties_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_GetRatioMetricProperties is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetThroughputMetricProperties_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  size_t throughputMetricIndex;
  const char *pDescription;
  uint32_t hwUnit;
  size_t numCounters;
  const size_t *pCounterIndices;
  size_t numSubThroughputs;
  const size_t *pSubThroughputIndices;
} HGPW_MetricsEvaluator_GetThroughputMetricProperties_Params;
#define HGPW_MetricsEvaluator_GetThroughputMetricProperties_Params_STRUCT_SIZE \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetThroughputMetricProperties_Params, \
                   pSubThroughputIndices)

static inline HGPA_Status HGPW_MetricsEvaluator_GetThroughputMetricProperties(
    HGPW_MetricsEvaluator_GetThroughputMetricProperties_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_MetricsEvaluator_GetThroughputMetricProperties is not "
                  "supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetSupportedSubmetrics_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  uint8_t metricType;
  const uint16_t *pSupportedSubmetrics;
  size_t numSupportedSubmetrics;
} HGPW_MetricsEvaluator_GetSupportedSubmetrics_Params;
#define HGPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE        \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetSupportedSubmetrics_Params,        \
                   numSupportedSubmetrics)

static inline HGPA_Status HGPW_MetricsEvaluator_GetSupportedSubmetrics(
    HGPW_MetricsEvaluator_GetSupportedSubmetrics_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_GetSupportedSubmetrics is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetMetricRawDependencies_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  const struct HGPW_MetricEvalRequest *pMetricEvalRequests;
  size_t numMetricEvalRequests;
  size_t metricEvalRequestStructSize;
  size_t metricEvalRequestStrideSize;
  const char **ppRawDependencies;
  size_t numRawDependencies;
  const char **ppOptionalRawDependencies;
  size_t numOptionalRawDependencies;
} HGPW_MetricsEvaluator_GetMetricRawDependencies_Params;
#define HGPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE      \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetMetricRawDependencies_Params,      \
                   numOptionalRawDependencies)

static inline HGPA_Status HGPW_MetricsEvaluator_GetMetricRawDependencies(
    HGPW_MetricsEvaluator_GetMetricRawDependencies_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_GetMetricRawDependencies is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_DimUnitToString_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  uint32_t dimUnit;
  const char *pSingularName;
  const char *pPluralName;
} HGPW_MetricsEvaluator_DimUnitToString_Params;
#define HGPW_MetricsEvaluator_DimUnitToString_Params_STRUCT_SIZE               \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_DimUnitToString_Params, pPluralName)

static inline HGPA_Status HGPW_MetricsEvaluator_DimUnitToString(
    HGPW_MetricsEvaluator_DimUnitToString_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_MetricsEvaluator_DimUnitToString is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_GetMetricDimUnits_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  const struct HGPW_MetricEvalRequest *pMetricEvalRequest;
  size_t metricEvalRequestStructSize;
  HGPW_DimUnitFactor *pDimUnits;
  size_t numDimUnits;
  size_t dimUnitFactorStructSize;
} HGPW_MetricsEvaluator_GetMetricDimUnits_Params;
#define HGPW_MetricsEvaluator_GetMetricDimUnits_Params_STRUCT_SIZE             \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_GetMetricDimUnits_Params,             \
                   dimUnitFactorStructSize)

static inline HGPA_Status HGPW_MetricsEvaluator_GetMetricDimUnits(
    HGPW_MetricsEvaluator_GetMetricDimUnits_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_GetMetricDimUnits is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_SetUserData_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  double frameDuration;
  double regionDuration;
  HGPA_Bool isolated;
} HGPW_MetricsEvaluator_SetUserData_Params;
#define HGPW_MetricsEvaluator_SetUserData_Params_STRUCT_SIZE                   \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_SetUserData_Params, isolated)

static inline HGPA_Status HGPW_MetricsEvaluator_SetUserData(
    HGPW_MetricsEvaluator_SetUserData_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_MetricsEvaluator_SetUserData is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_EvaluateToGpuValues_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  const struct HGPW_MetricEvalRequest *pMetricEvalRequests;
  size_t numMetricEvalRequests;
  size_t metricEvalRequestStructSize;
  size_t metricEvalRequestStrideSize;
  const uint8_t *pCounterDataImage;
  size_t counterDataImageSize;
  size_t rangeIndex;
  HGPA_Bool isolated;
  double *pMetricValues;
} HGPW_MetricsEvaluator_EvaluateToGpuValues_Params;
#define HGPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE           \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_EvaluateToGpuValues_Params,           \
                   pMetricValues)

static inline HGPA_Status HGPW_MetricsEvaluator_EvaluateToGpuValues(
    HGPW_MetricsEvaluator_EvaluateToGpuValues_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_EvaluateToGpuValues is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_MetricsEvaluator_SetDeviceAttributes_Params {
  size_t structSize;
  void *pPriv;
  struct HGPW_MetricsEvaluator *pMetricsEvaluator;
  const uint8_t *pCounterDataImage;
  size_t counterDataImageSize;
} HGPW_MetricsEvaluator_SetDeviceAttributes_Params;
#define HGPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE           \
  HGPA_STRUCT_SIZE(HGPW_MetricsEvaluator_SetDeviceAttributes_Params,           \
                   counterDataImageSize)

static inline HGPA_Status HGPW_MetricsEvaluator_SetDeviceAttributes(
    HGPW_MetricsEvaluator_SetDeviceAttributes_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_MetricsEvaluator_SetDeviceAttributes is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterData_CalculateCounterDataImageCopySize_Params {
  size_t structSize;
  void *pPriv;
  const uint8_t *pCounterDataPrefix;
  size_t counterDataPrefixSize;
  uint32_t maxNumRanges;
  uint32_t maxNumRangeTreeNodes;
  uint32_t maxRangeNameLength;
  const uint8_t *pCounterDataSrc;
  size_t copyDataImageCounterSize;
} HGPW_CounterData_CalculateCounterDataImageCopySize_Params;
#define HGPW_CounterData_CalculateCounterDataImageCopySize_Params_STRUCT_SIZE  \
  HGPA_STRUCT_SIZE(HGPW_CounterData_CalculateCounterDataImageCopySize_Params,  \
                   copyDataImageCounterSize)

static inline HGPA_Status HGPW_CounterData_CalculateCounterDataImageCopySize(
    HGPW_CounterData_CalculateCounterDataImageCopySize_Params *pParams) {
  (void)pParams;
  fprintf(
      stderr,
      "HGPW_CounterData_CalculateCounterDataImageCopySize is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterData_InitializeCounterDataImageCopy_Params {
  size_t structSize;
  void *pPriv;
  const uint8_t *pCounterDataPrefix;
  size_t counterDataPrefixSize;
  uint32_t maxNumRanges;
  uint32_t maxNumRangeTreeNodes;
  uint32_t maxRangeNameLength;
  const uint8_t *pCounterDataSrc;
  uint8_t *pCounterDataDst;
} HGPW_CounterData_InitializeCounterDataImageCopy_Params;
#define HGPW_CounterData_InitializeCounterDataImageCopy_Params_STRUCT_SIZE     \
  HGPA_STRUCT_SIZE(HGPW_CounterData_InitializeCounterDataImageCopy_Params,     \
                   pCounterDataDst)

static inline HGPA_Status HGPW_CounterData_InitializeCounterDataImageCopy(
    HGPW_CounterData_InitializeCounterDataImageCopy_Params *pParams) {
  (void)pParams;
  fprintf(
      stderr,
      "HGPW_CounterData_InitializeCounterDataImageCopy is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterData_ExtractCounterDataPrefix_Params {
  size_t structSize;
  void *pPriv;
  const uint8_t *pCounterDataSrc;
  size_t counterDataSrcSize;
  uint8_t *pCounterDataPrefix;
  size_t counterDataPrefixSize;
} HGPW_CounterData_ExtractCounterDataPrefix_Params;
#define HGPW_CounterData_ExtractCounterDataPrefix_Params_STRUCT_SIZE           \
  HGPA_STRUCT_SIZE(HGPW_CounterData_ExtractCounterDataPrefix_Params,           \
                   counterDataPrefixSize)

static inline HGPA_Status HGPW_CounterData_ExtractCounterDataPrefix(
    HGPW_CounterData_ExtractCounterDataPrefix_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_CounterData_ExtractCounterDataPrefix is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPA_CounterDataCombiner HGPA_CounterDataCombiner;

typedef struct HGPW_CounterDataCombiner_Create_Params {
  size_t structSize;
  void *pPriv;
  uint8_t *pCounterDataDst;
  HGPA_CounterDataCombiner *pCounterDataCombiner;
} HGPW_CounterDataCombiner_Create_Params;
#define HGPW_CounterDataCombiner_Create_Params_STRUCT_SIZE                     \
  HGPA_STRUCT_SIZE(HGPW_CounterDataCombiner_Create_Params, pCounterDataCombiner)

static inline HGPA_Status HGPW_CounterDataCombiner_Create(
    HGPW_CounterDataCombiner_Create_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataCombiner_Create is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataCombiner_Destroy_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataCombiner *pCounterDataCombiner;
} HGPW_CounterDataCombiner_Destroy_Params;
#define HGPW_CounterDataCombiner_Destroy_Params_STRUCT_SIZE                    \
  HGPA_STRUCT_SIZE(HGPW_CounterDataCombiner_Destroy_Params,                    \
                   pCounterDataCombiner)

static inline HGPA_Status HGPW_CounterDataCombiner_Destroy(
    HGPW_CounterDataCombiner_Destroy_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataCombiner_Destroy is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataCombiner_CreateRange_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataCombiner *pCounterDataCombiner;
  size_t numDescriptions;
  const char *const *ppDescriptions;
  size_t rangeIndexDst;
} HGPW_CounterDataCombiner_CreateRange_Params;
#define HGPW_CounterDataCombiner_CreateRange_Params_STRUCT_SIZE                \
  HGPA_STRUCT_SIZE(HGPW_CounterDataCombiner_CreateRange_Params, rangeIndexDst)

static inline HGPA_Status HGPW_CounterDataCombiner_CreateRange(
    HGPW_CounterDataCombiner_CreateRange_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataCombiner_CreateRange is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataCombiner_CopyIntoRange_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataCombiner *pCounterDataCombiner;
  size_t rangeIndexDst;
  const uint8_t *pCounterDataSrc;
  size_t rangeIndexSrc;
} HGPW_CounterDataCombiner_CopyIntoRange_Params;
#define HGPW_CounterDataCombiner_CopyIntoRange_Params_STRUCT_SIZE              \
  HGPA_STRUCT_SIZE(HGPW_CounterDataCombiner_CopyIntoRange_Params, rangeIndexSrc)

static inline HGPA_Status HGPW_CounterDataCombiner_CopyIntoRange(
    HGPW_CounterDataCombiner_CopyIntoRange_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataCombiner_CopyIntoRange is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataCombiner_AccumulateIntoRange_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataCombiner *pCounterDataCombiner;
  size_t rangeIndexDst;
  uint32_t dstMultiplier;
  const uint8_t *pCounterDataSrc;
  size_t rangeIndexSrc;
  uint32_t srcMultiplier;
} HGPW_CounterDataCombiner_AccumulateIntoRange_Params;
#define HGPW_CounterDataCombiner_AccumulateIntoRange_Params_STRUCT_SIZE        \
  HGPA_STRUCT_SIZE(HGPW_CounterDataCombiner_AccumulateIntoRange_Params,        \
                   srcMultiplier)

static inline HGPA_Status HGPW_CounterDataCombiner_AccumulateIntoRange(
    HGPW_CounterDataCombiner_AccumulateIntoRange_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_CounterDataCombiner_AccumulateIntoRange is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataCombiner_SumIntoRange_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataCombiner *pCounterDataCombiner;
  size_t rangeIndexDst;
  const uint8_t *pCounterDataSrc;
  size_t rangeIndexSrc;
} HGPW_CounterDataCombiner_SumIntoRange_Params;
#define HGPW_CounterDataCombiner_SumIntoRange_Params_STRUCT_SIZE               \
  HGPA_STRUCT_SIZE(HGPW_CounterDataCombiner_SumIntoRange_Params, rangeIndexSrc)

static inline HGPA_Status HGPW_CounterDataCombiner_SumIntoRange(
    HGPW_CounterDataCombiner_SumIntoRange_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataCombiner_SumIntoRange is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataCombiner_WeightedSumIntoRange_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataCombiner *pCounterDataCombiner;
  size_t rangeIndexDst;
  double dstMultiplier;
  const uint8_t *pCounterDataSrc;
  size_t rangeIndexSrc;
  double srcMultiplier;
} HGPW_CounterDataCombiner_WeightedSumIntoRange_Params;
#define HGPW_CounterDataCombiner_WeightedSumIntoRange_Params_STRUCT_SIZE       \
  HGPA_STRUCT_SIZE(HGPW_CounterDataCombiner_WeightedSumIntoRange_Params,       \
                   srcMultiplier)

static inline HGPA_Status HGPW_CounterDataCombiner_WeightedSumIntoRange(
    HGPW_CounterDataCombiner_WeightedSumIntoRange_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_CounterDataCombiner_WeightedSumIntoRange is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPA_CounterDataBuilder HGPA_CounterDataBuilder;

typedef struct HGPW_CounterDataBuilder_Create_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataBuilder *pCounterDataBuilder;
  const char *pChipName;
} HGPW_CounterDataBuilder_Create_Params;
#define HGPW_CounterDataBuilder_Create_Params_STRUCT_SIZE                      \
  HGPA_STRUCT_SIZE(HGPW_CounterDataBuilder_Create_Params, pChipName)

static inline HGPA_Status
HGPW_CounterDataBuilder_Create(HGPW_CounterDataBuilder_Create_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataBuilder_Create is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataBuilder_Destroy_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataBuilder *pCounterDataBuilder;
} HGPW_CounterDataBuilder_Destroy_Params;
#define HGPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE                     \
  HGPA_STRUCT_SIZE(HGPW_CounterDataBuilder_Destroy_Params, pCounterDataBuilder)

static inline HGPA_Status HGPW_CounterDataBuilder_Destroy(
    HGPW_CounterDataBuilder_Destroy_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataBuilder_Destroy is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataBuilder_AddMetrics_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataBuilder *pCounterDataBuilder;
  const HGPA_RawMetricRequest *pRawMetricRequests;
  size_t numMetricRequests;
} HGPW_CounterDataBuilder_AddMetrics_Params;
#define HGPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE                  \
  HGPA_STRUCT_SIZE(HGPW_CounterDataBuilder_AddMetrics_Params, numMetricRequests)

static inline HGPA_Status HGPW_CounterDataBuilder_AddMetrics(
    HGPW_CounterDataBuilder_AddMetrics_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataBuilder_AddMetrics is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataBuilder_AddRawCounters_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataBuilder *pCounterDataBuilder;
  size_t rawCounterRequestStructSize;
  size_t numRawCounterRequests;
  const HGPW_RawCounterRequest *pRawCounterRequests;
} HGPW_CounterDataBuilder_AddRawCounters_Params;
#define HGPW_CounterDataBuilder_AddRawCounters_Params_STRUCT_SIZE              \
  HGPA_STRUCT_SIZE(HGPW_CounterDataBuilder_AddRawCounters_Params,              \
                   pRawCounterRequests)

static inline HGPA_Status HGPW_CounterDataBuilder_AddRawCounters(
    HGPW_CounterDataBuilder_AddRawCounters_Params *pParams) {
  (void)pParams;
  fprintf(stderr, "HGPW_CounterDataBuilder_AddRawCounters is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

typedef struct HGPW_CounterDataBuilder_GetCounterDataPrefix_Params {
  size_t structSize;
  void *pPriv;
  HGPA_CounterDataBuilder *pCounterDataBuilder;
  size_t bytesAllocated;
  uint8_t *pBuffer;
  size_t bytesCopied;
} HGPW_CounterDataBuilder_GetCounterDataPrefix_Params;
#define HGPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE        \
  HGPA_STRUCT_SIZE(HGPW_CounterDataBuilder_GetCounterDataPrefix_Params,        \
                   bytesCopied)

static inline HGPA_Status HGPW_CounterDataBuilder_GetCounterDataPrefix(
    HGPW_CounterDataBuilder_GetCounterDataPrefix_Params *pParams) {
  (void)pParams;
  fprintf(stderr,
          "HGPW_CounterDataBuilder_GetCounterDataPrefix is not supported.\n");
  exit(1);
  return (HGPA_Status)0; /* unreachable */
}

#endif // HGPERF_HOST_API_DEFINED

#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(HGPA_SHARED_LIB)
#pragma GCC visibility pop
#endif

#endif // HGPERF_HOST_H
