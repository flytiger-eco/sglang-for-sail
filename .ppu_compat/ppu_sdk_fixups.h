#ifndef PPU_SDK_FIXUPS_H
#define PPU_SDK_FIXUPS_H

/*
 * ppu_sdk_fixups.h — PPU SDK fixups
 *
 * Included from compatible_wrapper.h in two phases:
 * 1. Pre-include  : before PPU SDK headers
 * 2. Post-include : after PPU SDK headers (activated by PPU_SDK_FIXUPS_POST)
 */

/* ═══════════════════════════════════════════════════════════════════════
 * Phase 1: Pre-include fixes
 * ═══════════════════════════════════════════════════════════════════════ */

/* hgcc does not implicitly include <functional> in C++ mode. */
#if defined(__cplusplus)
#include <functional>
#endif

#endif /* PPU_SDK_FIXUPS_H */

/* ═══════════════════════════════════════════════════════════════════════
 * Phase 2: Post-include fixes
 *
 * This section is outside the include guard. It is processed only when
 * PPU_SDK_FIXUPS_POST is defined before including this header.
 * ═══════════════════════════════════════════════════════════════════════ */
#ifdef PPU_SDK_FIXUPS_POST

/* PPU SDK has no cufile.h. */
#undef USE_CUFILE

/* Missing acdnnFraction_t type. */
typedef struct acdnnFractionStruct {
  int64_t numerator;
  int64_t denominator;
} acdnnFraction_t;

/* Missing acdnn descriptor types (commented out in PPU SDK). */
typedef struct acdnnAlgorithmStruct *acdnnAlgorithmDescriptor_t;
typedef struct acdnnAlgorithmPerformanceStruct *acdnnAlgorithmPerformance_t;
typedef struct acdnnPersistentRNNPlan *acdnnPersistentRNNPlan_t;

/* Missing ACDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H. C++ needs explicit enum cast.
 */
#if defined(__cplusplus)
#define ACDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H ((acdnnRNNAlgo_t)3)
#else
#define ACDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H 3
#endif
#undef ACDNN_RNN_ALGO_COUNT
#define ACDNN_RNN_ALGO_COUNT 4

/* ACDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT value mismatch. */
#undef ACDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT
#define ACDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT 6

/* ACDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT value mismatch. */
#undef ACDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT
#define ACDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT 7

/* Missing hggcDataType fp8/fp6/fp4 enum values. C++ needs explicit enum cast.
 */
#define HGGC_R_8F_UE4M3 HGGC_R_8F_E4M3
#if defined(__cplusplus)
#define HGGC_R_8F_UE8M0 ((hggcDataType)30)
#define HGGC_R_6F_E2M3 ((hggcDataType)31)
#define HGGC_R_6F_E3M2 ((hggcDataType)32)
#define HGGC_R_4F_E2M1 ((hggcDataType)33)
#else
#define HGGC_R_8F_UE8M0 30
#define HGGC_R_6F_E2M3 31
#define HGGC_R_6F_E3M2 32
#define HGGC_R_4F_E2M1 33
#endif

/* HGML_DEVICE_PCI_BUS_ID_FMT format string mismatch. */
#undef HGML_DEVICE_PCI_BUS_ID_FMT
#define HGML_DEVICE_PCI_BUS_ID_FMT "%08X:%02X:%02X.0"
#undef HGML_DEVICE_PCI_BUS_ID_LEGACY_FMT
#define HGML_DEVICE_PCI_BUS_ID_LEGACY_FMT "%04X:%02X:%02X.0"

/* Missing ACDNN_KNOB_TYPE_* enum values. C++ needs explicit enum cast. */
#if defined(__cplusplus)
#define ACDNN_KNOB_TYPE_WORKSPACE ((acdnnBackendKnobType_t)25)
#define ACDNN_KNOB_TYPE_TILE_CGA ((acdnnBackendKnobType_t)26)
#define ACDNN_KNOB_TYPE_TILE_CGA_M ((acdnnBackendKnobType_t)27)
#define ACDNN_KNOB_TYPE_TILE_CGA_N ((acdnnBackendKnobType_t)28)
#define ACDNN_KNOB_TYPE_BLOCK_SIZE ((acdnnBackendKnobType_t)29)
#define ACDNN_KNOB_TYPE_OCCUPANCY ((acdnnBackendKnobType_t)30)
#define ACDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD ((acdnnBackendKnobType_t)31)
#define ACDNN_KNOB_TYPE_NUM_C_PER_BLOCK ((acdnnBackendKnobType_t)32)
#define ACDNN_KNOB_TYPE_SPLIT_COLS ((acdnnBackendKnobType_t)33)
#define ACDNN_KNOB_TYPE_TILE_ROWS ((acdnnBackendKnobType_t)34)
#define ACDNN_KNOB_TYPE_TILE_COLS ((acdnnBackendKnobType_t)35)
#define ACDNN_KNOB_TYPE_LOAD_SIZE ((acdnnBackendKnobType_t)36)
#define ACDNN_KNOB_TYPE_CTA_COUNT ((acdnnBackendKnobType_t)38)
#define ACDNN_KNOB_TYPE_STREAM_K ((acdnnBackendKnobType_t)39)
#define ACDNN_KNOB_TYPE_SPLIT_P_SLC ((acdnnBackendKnobType_t)40)
#define ACDNN_KNOB_TYPE_TILE_M ((acdnnBackendKnobType_t)41)
#define ACDNN_KNOB_TYPE_TILE_N ((acdnnBackendKnobType_t)42)
#define ACDNN_KNOB_TYPE_WARP_SPEC_CFG ((acdnnBackendKnobType_t)43)
#endif /* __cplusplus */

/* Missing ACDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION. */
#if defined(__cplusplus)
#define ACDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION                          \
  ((acdnnBackendAttributeName_t)405)
#endif /* __cplusplus */

/* Missing ACDNN_TYPE_* enum values. C++ needs explicit enum cast. */
#if defined(__cplusplus)
#define ACDNN_TYPE_CHAR ((acdnnBackendAttributeType_t)24)
#define ACDNN_TYPE_SIGNAL_MODE ((acdnnBackendAttributeType_t)25)
#define ACDNN_TYPE_FRACTION ((acdnnBackendAttributeType_t)26)
#define ACDNN_TYPE_NORM_MODE ((acdnnBackendAttributeType_t)27)
#define ACDNN_TYPE_NORM_FWD_PHASE ((acdnnBackendAttributeType_t)28)
#define ACDNN_TYPE_RNG_DISTRIBUTION ((acdnnBackendAttributeType_t)29)
#endif /* __cplusplus */

/* Missing acdnnSignalMode_t enum. */
typedef enum {
  ACDNN_SIGNAL_SET = 0,
  ACDNN_SIGNAL_WAIT = 1,
} acdnnSignalMode_t;

/* Missing ACDNN_ATTR_* attribute name values. C++ needs explicit enum cast. */
#if defined(__cplusplus)
#define ACDNN_ATTR_POINTWISE_AXIS ((acdnnBackendAttributeName_t)9)
#define ACDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC ((acdnnBackendAttributeName_t)913)
#define ACDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET ((acdnnBackendAttributeName_t)203)
#define ACDNN_ATTR_ENGINE_SM_COUNT_TARGET ((acdnnBackendAttributeName_t)1306)
#define ACDNN_ATTR_MATMUL_PADDING_VALUE ((acdnnBackendAttributeName_t)1503)
#define ACDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC                       \
  ((acdnnBackendAttributeName_t)1525)
#define ACDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC                       \
  ((acdnnBackendAttributeName_t)1526)
#define ACDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC                       \
  ((acdnnBackendAttributeName_t)1527)
#define ACDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC                                \
  ((acdnnBackendAttributeName_t)1726)
#define ACDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC                                \
  ((acdnnBackendAttributeName_t)1727)
#define ACDNN_ATTR_OPERATION_CONCAT_AXIS ((acdnnBackendAttributeName_t)1800)
#define ACDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS                                \
  ((acdnnBackendAttributeName_t)1801)
#define ACDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX                              \
  ((acdnnBackendAttributeName_t)1802)
#define ACDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC                                \
  ((acdnnBackendAttributeName_t)1803)
#define ACDNN_ATTR_OPERATION_SIGNAL_MODE ((acdnnBackendAttributeName_t)1900)
#define ACDNN_ATTR_OPERATION_SIGNAL_FLAGDESC ((acdnnBackendAttributeName_t)1901)
#define ACDNN_ATTR_OPERATION_SIGNAL_VALUE ((acdnnBackendAttributeName_t)1902)
#define ACDNN_ATTR_OPERATION_SIGNAL_XDESC ((acdnnBackendAttributeName_t)1903)
#define ACDNN_ATTR_OPERATION_SIGNAL_YDESC ((acdnnBackendAttributeName_t)1904)
#define ACDNN_ATTR_OPERATION_NORM_FWD_MODE ((acdnnBackendAttributeName_t)2000)
#define ACDNN_ATTR_OPERATION_NORM_FWD_PHASE ((acdnnBackendAttributeName_t)2001)
#define ACDNN_ATTR_OPERATION_NORM_FWD_XDESC ((acdnnBackendAttributeName_t)2002)
#define ACDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC                                \
  ((acdnnBackendAttributeName_t)2003)
#define ACDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC                        \
  ((acdnnBackendAttributeName_t)2004)
#define ACDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC                               \
  ((acdnnBackendAttributeName_t)2005)
#define ACDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC                                \
  ((acdnnBackendAttributeName_t)2006)
#define ACDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC                             \
  ((acdnnBackendAttributeName_t)2007)
#define ACDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC                      \
  ((acdnnBackendAttributeName_t)2008)
#define ACDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC                  \
  ((acdnnBackendAttributeName_t)2009)
#define ACDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC                   \
  ((acdnnBackendAttributeName_t)2010)
#define ACDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC                 \
  ((acdnnBackendAttributeName_t)2011)
#define ACDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC                  \
  ((acdnnBackendAttributeName_t)2012)
#define ACDNN_ATTR_OPERATION_NORM_FWD_YDESC ((acdnnBackendAttributeName_t)2013)
#define ACDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS                          \
  ((acdnnBackendAttributeName_t)2014)
#define ACDNN_ATTR_OPERATION_NORM_BWD_MODE ((acdnnBackendAttributeName_t)2100)
#define ACDNN_ATTR_OPERATION_NORM_BWD_XDESC ((acdnnBackendAttributeName_t)2101)
#define ACDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC                                \
  ((acdnnBackendAttributeName_t)2102)
#define ACDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC                        \
  ((acdnnBackendAttributeName_t)2103)
#define ACDNN_ATTR_OPERATION_NORM_BWD_DYDESC ((acdnnBackendAttributeName_t)2104)
#define ACDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC                               \
  ((acdnnBackendAttributeName_t)2105)
#define ACDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC                             \
  ((acdnnBackendAttributeName_t)2106)
#define ACDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC                              \
  ((acdnnBackendAttributeName_t)2107)
#define ACDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC                               \
  ((acdnnBackendAttributeName_t)2108)
#define ACDNN_ATTR_OPERATION_NORM_BWD_DXDESC ((acdnnBackendAttributeName_t)2109)
#define ACDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS                          \
  ((acdnnBackendAttributeName_t)2110)
#define ACDNN_ATTR_OPERATION_RESHAPE_XDESC ((acdnnBackendAttributeName_t)2200)
#define ACDNN_ATTR_OPERATION_RESHAPE_YDESC ((acdnnBackendAttributeName_t)2201)
#define ACDNN_ATTR_RNG_DISTRIBUTION ((acdnnBackendAttributeName_t)2300)
#define ACDNN_ATTR_RNG_NORMAL_DIST_MEAN ((acdnnBackendAttributeName_t)2301)
#define ACDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION                          \
  ((acdnnBackendAttributeName_t)2302)
#define ACDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM ((acdnnBackendAttributeName_t)2303)
#define ACDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM ((acdnnBackendAttributeName_t)2304)
#define ACDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY                              \
  ((acdnnBackendAttributeName_t)2305)
#define ACDNN_ATTR_OPERATION_RNG_YDESC ((acdnnBackendAttributeName_t)2310)
#define ACDNN_ATTR_OPERATION_RNG_SEED ((acdnnBackendAttributeName_t)2311)
#define ACDNN_ATTR_OPERATION_RNG_DESC ((acdnnBackendAttributeName_t)2312)
#define ACDNN_ATTR_OPERATION_RNG_OFFSET_DESC ((acdnnBackendAttributeName_t)2313)
#endif /* __cplusplus */

/* Missing acdnnReorderType_t enum. */
typedef enum {
  ACDNN_DEFAULT_REORDER = 0,
  ACDNN_NO_REORDER = 1,
} acdnnReorderType_t;

/* Missing acdnnRngDistribution_t enum. */
typedef enum {
  ACDNN_RNG_DISTRIBUTION_BERNOULLI = 0,
  ACDNN_RNG_DISTRIBUTION_UNIFORM = 1,
  ACDNN_RNG_DISTRIBUTION_NORMAL = 2,
} acdnnRngDistribution_t;

/* Missing ACDNN_DATA_FP8_E4M3 and ACDNN_DATA_FP8_E5M2. C++ needs explicit enum
 * cast. */
#if defined(__cplusplus)
#define ACDNN_DATA_FP8_E4M3 ((acdnnDataType_t)13)
#define ACDNN_DATA_FP8_E5M2 ((acdnnDataType_t)14)
#else
#define ACDNN_DATA_FP8_E4M3 13
#define ACDNN_DATA_FP8_E5M2 14
#endif

/* Missing ACDNN_DATA_FAST_FLOAT_FOR_FP8. */
#if defined(__cplusplus)
#define ACDNN_DATA_FAST_FLOAT_FOR_FP8 ((acdnnDataType_t)15)
#else
#define ACDNN_DATA_FAST_FLOAT_FOR_FP8 15
#endif

/* Missing acdnnPointwiseMode_t enum values. C++ needs explicit enum cast. */
#if defined(__cplusplus)
#define ACDNN_POINTWISE_ERF ((acdnnPointwiseMode_t)20)
#define ACDNN_POINTWISE_IDENTITY ((acdnnPointwiseMode_t)21)
#define ACDNN_POINTWISE_RECIPROCAL ((acdnnPointwiseMode_t)22)
#define ACDNN_POINTWISE_GEN_INDEX ((acdnnPointwiseMode_t)501)
#define ACDNN_POINTWISE_BINARY_SELECT ((acdnnPointwiseMode_t)601)
#else
#define ACDNN_POINTWISE_ERF 20
#define ACDNN_POINTWISE_IDENTITY 21
#define ACDNN_POINTWISE_RECIPROCAL 22
#define ACDNN_POINTWISE_GEN_INDEX 501
#define ACDNN_POINTWISE_BINARY_SELECT 601
#endif

/* Missing acdnnBackendNormMode_t enum. */
typedef enum {
  ACDNN_LAYER_NORM = 0,
  ACDNN_INSTANCE_NORM = 1,
  ACDNN_BATCH_NORM = 2,
  ACDNN_GROUP_NORM = 3,
} acdnnBackendNormMode_t;

/* Missing acdnnBackendNormFwdPhase_t enum. */
typedef enum {
  ACDNN_NORM_FWD_INFERENCE = 0,
  ACDNN_NORM_FWD_TRAINING = 1,
} acdnnBackendNormFwdPhase_t;

/* Missing ACDNN_TENSOR_REORDERING_F16x16. C++ needs explicit enum cast. */
#if defined(__cplusplus)
#define ACDNN_TENSOR_REORDERING_F16x16 ((acdnnBackendTensorReordering_t)2)
#else
#define ACDNN_TENSOR_REORDERING_F16x16 2
#endif

/* Missing ACDNN_RESAMPLE_AVGPOOL_INCLUDE/EXCLUDE_PADDING. C++ needs explicit
 * enum cast. */
#if defined(__cplusplus)
#define ACDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING ((acdnnResampleMode_t)2)
#define ACDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING ((acdnnResampleMode_t)4)
#else
#define ACDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING 2
#define ACDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING 4
#endif

/* Missing acdnnBackendDescriptorType_t enum values. C++ needs explicit enum
 * cast. */
#if defined(__cplusplus)
#define ACDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR                              \
  ((acdnnBackendDescriptorType_t)27)
#define ACDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR                              \
  ((acdnnBackendDescriptorType_t)28)
#define ACDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR                        \
  ((acdnnBackendDescriptorType_t)29)
#define ACDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR                       \
  ((acdnnBackendDescriptorType_t)30)
#define ACDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR                             \
  ((acdnnBackendDescriptorType_t)31)
#define ACDNN_BACKEND_RNG_DESCRIPTOR ((acdnnBackendDescriptorType_t)32)
#define ACDNN_BACKEND_OPERATION_RNG_DESCRIPTOR                                 \
  ((acdnnBackendDescriptorType_t)33)
#else
#define ACDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR 27
#define ACDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR 28
#define ACDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR 29
#define ACDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR 30
#define ACDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR 31
#define ACDNN_BACKEND_RNG_DESCRIPTOR 32
#define ACDNN_BACKEND_OPERATION_RNG_DESCRIPTOR 33
#endif

/* Missing ACDNN_STATUS_VERSION_MISMATCH. C++ needs explicit enum cast. */
#if defined(__cplusplus)
#define ACDNN_STATUS_VERSION_MISMATCH ((acdnnStatus_t)14)
#else
#define ACDNN_STATUS_VERSION_MISMATCH 14
#endif

/* C++ compatibility wrappers: provide overloads that accept hggcDataType
 * for computeType, migrating to acblasComputeType_t before calling the real
 * API. */
#if defined(__cplusplus)

/* Convert hggcDataType to acblasComputeType_t. */
static inline acblasStatus_t
acblasMigrateComputeType(acblasHandle_t handle, hggcDataType dataType,
                         acblasComputeType_t *computeType) {
  acblasMath_t mathMode = ACBLAS_DEFAULT_MATH;
  acblasStatus_t status = acblasGetMathMode(handle, &mathMode);
  if (status != ACBLAS_STATUS_SUCCESS) {
    return status;
  }

  bool isPedantic = ((mathMode & 0xf) == ACBLAS_PEDANTIC_MATH);

  switch (dataType) {
  case HGGC_R_32F:
  case HGGC_C_32F:
    *computeType =
        isPedantic ? ACBLAS_COMPUTE_32F_PEDANTIC : ACBLAS_COMPUTE_32F;
    return ACBLAS_STATUS_SUCCESS;
  case HGGC_R_64F:
  case HGGC_C_64F:
    *computeType =
        isPedantic ? ACBLAS_COMPUTE_64F_PEDANTIC : ACBLAS_COMPUTE_64F;
    return ACBLAS_STATUS_SUCCESS;
  case HGGC_R_16F:
    *computeType =
        isPedantic ? ACBLAS_COMPUTE_16F_PEDANTIC : ACBLAS_COMPUTE_16F;
    return ACBLAS_STATUS_SUCCESS;
  case HGGC_R_32I:
    *computeType =
        isPedantic ? ACBLAS_COMPUTE_32I_PEDANTIC : ACBLAS_COMPUTE_32I;
    return ACBLAS_STATUS_SUCCESS;
  default:
    return ACBLAS_STATUS_NOT_SUPPORTED;
  }
}

/* Overload: accepts hggcDataType for computeType. */
static inline acblasStatus_t
acblasGemmEx(acblasHandle_t handle, acblasOperation_t transa,
             acblasOperation_t transb, int m, int n, int k, const void *alpha,
             const void *A, hggcDataType Atype, int lda, const void *B,
             hggcDataType Btype, int ldb, const void *beta, void *C,
             hggcDataType Ctype, int ldc, hggcDataType computeType,
             acblasGemmAlgo_t algo) {
  acblasComputeType_t migratedComputeType = ACBLAS_COMPUTE_32F;
  acblasStatus_t status =
      acblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != ACBLAS_STATUS_SUCCESS) {
    return status;
  }

  return acblasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
                      Btype, ldb, beta, C, Ctype, ldc, migratedComputeType,
                      algo);
}

/* Overload: accepts hggcDataType for computeType. */
static inline acblasStatus_t acblasGemmStridedBatchedEx(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int m, int n, int k, const void *alpha, const void *A, hggcDataType Atype,
    int lda, long long int strideA, const void *B, hggcDataType Btype, int ldb,
    long long int strideB, const void *beta, void *C, hggcDataType Ctype,
    int ldc, long long int strideC, int batchCount, hggcDataType computeType,
    acblasGemmAlgo_t algo) {
  acblasComputeType_t migratedComputeType = ACBLAS_COMPUTE_32F;
  acblasStatus_t status =
      acblasMigrateComputeType(handle, computeType, &migratedComputeType);
  if (status != ACBLAS_STATUS_SUCCESS) {
    return status;
  }

  return acblasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A,
                                    Atype, lda, strideA, B, Btype, ldb, strideB,
                                    beta, C, Ctype, ldc, strideC, batchCount,
                                    migratedComputeType, algo);
}
/* Missing CUPTI CBID enum values. Defined as _SIZE sentinel (no-op). */
#ifndef HGPTI_HGGC_DRIVER_CBID_hgFuncGetParamInfo
#define HGPTI_HGGC_DRIVER_CBID_hgFuncGetParamInfo HGPTI_HGGC_DRIVER_CBID_SIZE
#endif

#ifndef HGPTI_HGGC_RUNTIME_CBID_hggcLibraryLoadData_v12060
#define HGPTI_HGGC_RUNTIME_CBID_hggcLibraryLoadData_v12060                     \
  HGPTI_HGGC_RUNTIME_CBID_SIZE
#endif

#endif /* __cplusplus */

#undef PPU_SDK_FIXUPS_POST
#endif /* PPU_SDK_FIXUPS_POST */
