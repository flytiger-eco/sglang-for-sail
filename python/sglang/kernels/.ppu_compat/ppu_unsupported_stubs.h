#ifndef PPU_UNSUPPORTED_STUBS_H
#define PPU_UNSUPPORTED_STUBS_H

/* ── cublas: 426 unsupported APIs ── */

static inline acblasStatus_t acblasSetKernelStream(hggcStream_t stream) {
  (void)stream;
  fprintf(stderr, "acblasSetKernelStream is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasAsumEx_64(acblasHandle_t handle, int64_t n,
                                             const void *x, hggcDataType xType,
                                             int64_t incx, void *result,
                                             hggcDataType resultType,
                                             hggcDataType executiontype) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)result;
  (void)resultType;
  (void)executiontype;
  fprintf(stderr, "acblasAsumEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasAxpyEx_64(acblasHandle_t handle, int64_t n, const void *alpha,
                hggcDataType alphaType, const void *x, hggcDataType xType,
                int64_t incx, void *y, hggcDataType yType, int64_t incy,
                hggcDataType executiontype) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)alphaType;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  (void)executiontype;
  fprintf(stderr, "acblasAxpyEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCaxpy_v2(acblasHandle_t handle, int n,
                                            const acComplex *alpha,
                                            const acComplex *x, int incx,
                                            acComplex *y, int incy) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCaxpy_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCaxpy_v2_64(acblasHandle_t handle, int64_t n,
                                               const acComplex *alpha,
                                               const acComplex *x, int64_t incx,
                                               acComplex *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCaxpy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCcopy_v2(acblasHandle_t handle, int n,
                                            const acComplex *x, int incx,
                                            acComplex *y, int incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCcopy_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCcopy_v2_64(acblasHandle_t handle, int64_t n,
                                               const acComplex *x, int64_t incx,
                                               acComplex *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCcopy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCdgmm(acblasHandle_t handle,
                                         acblasSideMode_t mode, int m, int n,
                                         const acComplex *A, int lda,
                                         const acComplex *x, int incx,
                                         acComplex *C, int ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCdgmm is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCdgmm_64(acblasHandle_t handle, acblasSideMode_t mode, int64_t m,
               int64_t n, const acComplex *A, int64_t lda, const acComplex *x,
               int64_t incx, acComplex *C, int64_t ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCdgmm_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCdotc_v2(acblasHandle_t handle, int n,
                                            const acComplex *x, int incx,
                                            const acComplex *y, int incy,
                                            acComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasCdotc_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCdotc_v2_64(acblasHandle_t handle, int64_t n,
                                               const acComplex *x, int64_t incx,
                                               const acComplex *y, int64_t incy,
                                               acComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasCdotc_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCdotu_v2(acblasHandle_t handle, int n,
                                            const acComplex *x, int incx,
                                            const acComplex *y, int incy,
                                            acComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasCdotu_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCdotu_v2_64(acblasHandle_t handle, int64_t n,
                                               const acComplex *x, int64_t incx,
                                               const acComplex *y, int64_t incy,
                                               acComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasCdotu_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgbmv_v2(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
               int kl, int ku, const acComplex *alpha, const acComplex *A,
               int lda, const acComplex *x, int incx, const acComplex *beta,
               acComplex *y, int incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCgbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgbmv_v2_64(acblasHandle_t handle, acblasOperation_t trans, int64_t m,
                  int64_t n, int64_t kl, int64_t ku, const acComplex *alpha,
                  const acComplex *A, int64_t lda, const acComplex *x,
                  int64_t incx, const acComplex *beta, acComplex *y,
                  int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCgbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgeam(acblasHandle_t handle, acblasOperation_t transa,
            acblasOperation_t transb, int m, int n, const acComplex *alpha,
            const acComplex *A, int lda, const acComplex *beta,
            const acComplex *B, int ldb, acComplex *C, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCgeam is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgeam_64(acblasHandle_t handle, acblasOperation_t transa,
               acblasOperation_t transb, int64_t m, int64_t n,
               const acComplex *alpha, const acComplex *A, int64_t lda,
               const acComplex *beta, const acComplex *B, int64_t ldb,
               acComplex *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCgeam_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgelsBatched(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
                   int nrhs, acComplex *const Aarray[], int lda,
                   acComplex *const Carray[], int ldc, int *info,
                   int *devInfoArray, int batchSize) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)Aarray;
  (void)lda;
  (void)Carray;
  (void)ldc;
  (void)info;
  (void)devInfoArray;
  (void)batchSize;
  fprintf(stderr, "acblasCgelsBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemm3m(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int m, int n, int k, const acComplex *alpha, const acComplex *A, int lda,
    const acComplex *B, int ldb, const acComplex *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCgemm3m is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemm3mBatched(acblasHandle_t handle, acblasOperation_t transa,
                     acblasOperation_t transb, int m, int n, int k,
                     const acComplex *alpha, const acComplex *const Aarray[],
                     int lda, const acComplex *const Barray[], int ldb,
                     const acComplex *beta, acComplex *const Carray[], int ldc,
                     int batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasCgemm3mBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemm3mBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const acComplex *alpha,
    const acComplex *const Aarray[], int64_t lda,
    const acComplex *const Barray[], int64_t ldb, const acComplex *beta,
    acComplex *const Carray[], int64_t ldc, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasCgemm3mBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemm3mEx(acblasHandle_t handle, acblasOperation_t transa,
                acblasOperation_t transb, int m, int n, int k,
                const acComplex *alpha, const void *A, hggcDataType Atype,
                int lda, const void *B, hggcDataType Btype, int ldb,
                const acComplex *beta, void *C, hggcDataType Ctype, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)B;
  (void)Btype;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCgemm3mEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemm3mEx_64(acblasHandle_t handle, acblasOperation_t transa,
                   acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                   const acComplex *alpha, const void *A, hggcDataType Atype,
                   int64_t lda, const void *B, hggcDataType Btype, int64_t ldb,
                   const acComplex *beta, void *C, hggcDataType Ctype,
                   int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)B;
  (void)Btype;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCgemm3mEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemm3mStridedBatched(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int m, int n, int k, const acComplex *alpha, const acComplex *A, int lda,
    long long strideA, const acComplex *B, int ldb, long long strideB,
    const acComplex *beta, acComplex *C, int ldc, long long strideC,
    int batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasCgemm3mStridedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemm3mStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const acComplex *alpha, const acComplex *A,
    int64_t lda, long long strideA, const acComplex *B, int64_t ldb,
    long long strideB, const acComplex *beta, acComplex *C, int64_t ldc,
    long long strideC, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasCgemm3mStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemm3m_64(acblasHandle_t handle, acblasOperation_t transa,
                 acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                 const acComplex *alpha, const acComplex *A, int64_t lda,
                 const acComplex *B, int64_t ldb, const acComplex *beta,
                 acComplex *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCgemm3m_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemmBatched(acblasHandle_t handle, acblasOperation_t transa,
                   acblasOperation_t transb, int m, int n, int k,
                   const acComplex *alpha, const acComplex *const Aarray[],
                   int lda, const acComplex *const Barray[], int ldb,
                   const acComplex *beta, acComplex *const Carray[], int ldc,
                   int batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasCgemmBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemmBatched_64(acblasHandle_t handle, acblasOperation_t transa,
                      acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                      const acComplex *alpha, const acComplex *const Aarray[],
                      int64_t lda, const acComplex *const Barray[], int64_t ldb,
                      const acComplex *beta, acComplex *const Carray[],
                      int64_t ldc, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasCgemmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemmEx(acblasHandle_t handle, acblasOperation_t transa,
              acblasOperation_t transb, int m, int n, int k,
              const acComplex *alpha, const void *A, hggcDataType Atype,
              int lda, const void *B, hggcDataType Btype, int ldb,
              const acComplex *beta, void *C, hggcDataType Ctype, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)B;
  (void)Btype;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCgemmEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemmEx_64(acblasHandle_t handle, acblasOperation_t transa,
                 acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                 const acComplex *alpha, const void *A, hggcDataType Atype,
                 int64_t lda, const void *B, hggcDataType Btype, int64_t ldb,
                 const acComplex *beta, void *C, hggcDataType Ctype,
                 int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)B;
  (void)Btype;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCgemmEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemmStridedBatched(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int m, int n, int k, const acComplex *alpha, const acComplex *A, int lda,
    long long strideA, const acComplex *B, int ldb, long long strideB,
    const acComplex *beta, acComplex *C, int ldc, long long strideC,
    int batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasCgemmStridedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemmStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const acComplex *alpha, const acComplex *A,
    int64_t lda, long long strideA, const acComplex *B, int64_t ldb,
    long long strideB, const acComplex *beta, acComplex *C, int64_t ldc,
    long long strideC, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasCgemmStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemm_v2(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int m, int n, int k, const acComplex *alpha, const acComplex *A, int lda,
    const acComplex *B, int ldb, const acComplex *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCgemm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemm_v2_64(acblasHandle_t handle, acblasOperation_t transa,
                  acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  const acComplex *B, int64_t ldb, const acComplex *beta,
                  acComplex *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCgemm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemvBatched(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
                   const acComplex *alpha, const acComplex *const Aarray[],
                   int lda, const acComplex *const xarray[], int incx,
                   const acComplex *beta, acComplex *const yarray[], int incy,
                   int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasCgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const acComplex *alpha, const acComplex *const Aarray[], int64_t lda,
    const acComplex *const xarray[], int64_t incx, const acComplex *beta,
    acComplex *const yarray[], int64_t incy, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasCgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const acComplex *alpha, const acComplex *A, int64_t lda, long long strideA,
    const acComplex *x, int64_t incx, long long stridex, const acComplex *beta,
    acComplex *y, int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasCgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemv_v2(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
               const acComplex *alpha, const acComplex *A, int lda,
               const acComplex *x, int incx, const acComplex *beta,
               acComplex *y, int incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCgemv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgemv_v2_64(acblasHandle_t handle, acblasOperation_t trans, int64_t m,
                  int64_t n, const acComplex *alpha, const acComplex *A,
                  int64_t lda, const acComplex *x, int64_t incx,
                  const acComplex *beta, acComplex *y, int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCgemv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgeqrfBatched(acblasHandle_t handle, int m, int n,
                    acComplex *const Aarray[], int lda,
                    acComplex *const TauArray[], int *info, int batchSize) {
  (void)handle;
  (void)m;
  (void)n;
  (void)Aarray;
  (void)lda;
  (void)TauArray;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasCgeqrfBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgerc_v2(acblasHandle_t handle, int m, int n,
                                            const acComplex *alpha,
                                            const acComplex *x, int incx,
                                            const acComplex *y, int incy,
                                            acComplex *A, int lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCgerc_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgerc_v2_64(acblasHandle_t handle, int64_t m, int64_t n,
                  const acComplex *alpha, const acComplex *x, int64_t incx,
                  const acComplex *y, int64_t incy, acComplex *A, int64_t lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCgerc_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgeru_v2(acblasHandle_t handle, int m, int n,
                                            const acComplex *alpha,
                                            const acComplex *x, int incx,
                                            const acComplex *y, int incy,
                                            acComplex *A, int lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCgeru_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgeru_v2_64(acblasHandle_t handle, int64_t m, int64_t n,
                  const acComplex *alpha, const acComplex *x, int64_t incx,
                  const acComplex *y, int64_t incy, acComplex *A, int64_t lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCgeru_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgetrfBatched(acblasHandle_t handle, int n,
                                                 acComplex *const A[], int lda,
                                                 int *P, int *info,
                                                 int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)P;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasCgetrfBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCgetriBatched(acblasHandle_t handle, int n,
                                                 const acComplex *const A[],
                                                 int lda, const int *P,
                                                 acComplex *const C[], int ldc,
                                                 int *info, int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)P;
  (void)C;
  (void)ldc;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasCgetriBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCgetrsBatched(acblasHandle_t handle, acblasOperation_t trans, int n,
                    int nrhs, const acComplex *const Aarray[], int lda,
                    const int *devIpiv, acComplex *const Barray[], int ldb,
                    int *info, int batchSize) {
  (void)handle;
  (void)trans;
  (void)n;
  (void)nrhs;
  (void)Aarray;
  (void)lda;
  (void)devIpiv;
  (void)Barray;
  (void)ldb;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasCgetrsBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChbmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n, int k,
               const acComplex *alpha, const acComplex *A, int lda,
               const acComplex *x, int incx, const acComplex *beta,
               acComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasChbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChbmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  int64_t k, const acComplex *alpha, const acComplex *A,
                  int64_t lda, const acComplex *x, int64_t incx,
                  const acComplex *beta, acComplex *y, int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasChbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChemm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, int m, int n, const acComplex *alpha,
               const acComplex *A, int lda, const acComplex *B, int ldb,
               const acComplex *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasChemm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChemm_v2_64(acblasHandle_t handle, acblasSideMode_t side,
                  acblasFillMode_t uplo, int64_t m, int64_t n,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  const acComplex *B, int64_t ldb, const acComplex *beta,
                  acComplex *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasChemm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChemv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acComplex *alpha, const acComplex *A, int lda,
               const acComplex *x, int incx, const acComplex *beta,
               acComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasChemv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChemv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  const acComplex *x, int64_t incx, const acComplex *beta,
                  acComplex *y, int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasChemv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCher2_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acComplex *alpha, const acComplex *x, int incx,
               const acComplex *y, int incy, acComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCher2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCher2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acComplex *alpha, const acComplex *x, int64_t incx,
                  const acComplex *y, int64_t incy, acComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCher2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCher2k_v2(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k, const acComplex *alpha,
                const acComplex *A, int lda, const acComplex *B, int ldb,
                const float *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCher2k_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCher2k_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                   acblasOperation_t trans, int64_t n, int64_t k,
                   const acComplex *alpha, const acComplex *A, int64_t lda,
                   const acComplex *B, int64_t ldb, const float *beta,
                   acComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCher2k_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCher_v2(acblasHandle_t handle,
                                           acblasFillMode_t uplo, int n,
                                           const float *alpha,
                                           const acComplex *x, int incx,
                                           acComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCher_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCher_v2_64(acblasHandle_t handle,
                                              acblasFillMode_t uplo, int64_t n,
                                              const float *alpha,
                                              const acComplex *x, int64_t incx,
                                              acComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCher_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCherk3mEx(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k, const float *alpha,
                const void *A, hggcDataType Atype, int lda, const float *beta,
                void *C, hggcDataType Ctype, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCherk3mEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCherk3mEx_64(
    acblasHandle_t handle, acblasFillMode_t uplo, acblasOperation_t trans,
    int64_t n, int64_t k, const float *alpha, const void *A, hggcDataType Atype,
    int64_t lda, const float *beta, void *C, hggcDataType Ctype, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCherk3mEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCherkEx(acblasHandle_t handle, acblasFillMode_t uplo,
              acblasOperation_t trans, int n, int k, const float *alpha,
              const void *A, hggcDataType Atype, int lda, const float *beta,
              void *C, hggcDataType Ctype, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCherkEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCherkEx_64(
    acblasHandle_t handle, acblasFillMode_t uplo, acblasOperation_t trans,
    int64_t n, int64_t k, const float *alpha, const void *A, hggcDataType Atype,
    int64_t lda, const float *beta, void *C, hggcDataType Ctype, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCherkEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCherk_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, int n, int k, const float *alpha,
               const acComplex *A, int lda, const float *beta, acComplex *C,
               int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCherk_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCherk_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, int64_t n, int64_t k,
                  const float *alpha, const acComplex *A, int64_t lda,
                  const float *beta, acComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCherk_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCherkx(acblasHandle_t handle, acblasFillMode_t uplo,
             acblasOperation_t trans, int n, int k, const acComplex *alpha,
             const acComplex *A, int lda, const acComplex *B, int ldb,
             const float *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCherkx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCherkx_64(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int64_t n, int64_t k,
                const acComplex *alpha, const acComplex *A, int64_t lda,
                const acComplex *B, int64_t ldb, const float *beta,
                acComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCherkx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChpmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acComplex *alpha, const acComplex *AP, const acComplex *x,
               int incx, const acComplex *beta, acComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasChpmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChpmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acComplex *alpha, const acComplex *AP,
                  const acComplex *x, int64_t incx, const acComplex *beta,
                  acComplex *y, int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasChpmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChpr2_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acComplex *alpha, const acComplex *x, int incx,
               const acComplex *y, int incy, acComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)AP;
  fprintf(stderr, "acblasChpr2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChpr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acComplex *alpha, const acComplex *x, int64_t incx,
                  const acComplex *y, int64_t incy, acComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)AP;
  fprintf(stderr, "acblasChpr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasChpr_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
              const float *alpha, const acComplex *x, int incx, acComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)AP;
  fprintf(stderr, "acblasChpr_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasChpr_v2_64(acblasHandle_t handle,
                                              acblasFillMode_t uplo, int64_t n,
                                              const float *alpha,
                                              const acComplex *x, int64_t incx,
                                              acComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)AP;
  fprintf(stderr, "acblasChpr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCmatinvBatched(acblasHandle_t handle, int n, const acComplex *const A[],
                     int lda, acComplex *const Ainv[], int lda_inv, int *info,
                     int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)Ainv;
  (void)lda_inv;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasCmatinvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCopyEx_64(acblasHandle_t handle, int64_t n,
                                             const void *x, hggcDataType xType,
                                             int64_t incx, void *y,
                                             hggcDataType yType, int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  fprintf(stderr, "acblasCopyEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCrot_v2(acblasHandle_t handle, int n,
                                           acComplex *x, int incx, acComplex *y,
                                           int incy, const float *c,
                                           const acComplex *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasCrot_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCrot_v2_64(acblasHandle_t handle, int64_t n,
                                              acComplex *x, int64_t incx,
                                              acComplex *y, int64_t incy,
                                              const float *c,
                                              const acComplex *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasCrot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCrotg_v2(acblasHandle_t handle, acComplex *a,
                                            acComplex *b, float *c,
                                            acComplex *s) {
  (void)handle;
  (void)a;
  (void)b;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasCrotg_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCscal_v2(acblasHandle_t handle, int n,
                                            const acComplex *alpha,
                                            acComplex *x, int incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCscal_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCscal_v2_64(acblasHandle_t handle, int64_t n,
                                               const acComplex *alpha,
                                               acComplex *x, int64_t incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCscal_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCsrot_v2(acblasHandle_t handle, int n,
                                            acComplex *x, int incx,
                                            acComplex *y, int incy,
                                            const float *c, const float *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasCsrot_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCsrot_v2_64(acblasHandle_t handle, int64_t n,
                                               acComplex *x, int64_t incx,
                                               acComplex *y, int64_t incy,
                                               const float *c, const float *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasCsrot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCsscal_v2(acblasHandle_t handle, int n,
                                             const float *alpha, acComplex *x,
                                             int incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCsscal_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCsscal_v2_64(acblasHandle_t handle,
                                                int64_t n, const float *alpha,
                                                acComplex *x, int64_t incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCsscal_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCswap_v2(acblasHandle_t handle, int n,
                                            acComplex *x, int incx,
                                            acComplex *y, int incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCswap_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCswap_v2_64(acblasHandle_t handle, int64_t n,
                                               acComplex *x, int64_t incx,
                                               acComplex *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCswap_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsymm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, int m, int n, const acComplex *alpha,
               const acComplex *A, int lda, const acComplex *B, int ldb,
               const acComplex *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsymm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsymm_v2_64(acblasHandle_t handle, acblasSideMode_t side,
                  acblasFillMode_t uplo, int64_t m, int64_t n,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  const acComplex *B, int64_t ldb, const acComplex *beta,
                  acComplex *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsymm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsymv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acComplex *alpha, const acComplex *A, int lda,
               const acComplex *x, int incx, const acComplex *beta,
               acComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCsymv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsymv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  const acComplex *x, int64_t incx, const acComplex *beta,
                  acComplex *y, int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasCsymv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyr2_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acComplex *alpha, const acComplex *x, int incx,
               const acComplex *y, int incy, acComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCsyr2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acComplex *alpha, const acComplex *x, int64_t incx,
                  const acComplex *y, int64_t incy, acComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCsyr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyr2k_v2(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k, const acComplex *alpha,
                const acComplex *A, int lda, const acComplex *B, int ldb,
                const acComplex *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsyr2k_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyr2k_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                   acblasOperation_t trans, int64_t n, int64_t k,
                   const acComplex *alpha, const acComplex *A, int64_t lda,
                   const acComplex *B, int64_t ldb, const acComplex *beta,
                   acComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsyr2k_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCsyr_v2(acblasHandle_t handle,
                                           acblasFillMode_t uplo, int n,
                                           const acComplex *alpha,
                                           const acComplex *x, int incx,
                                           acComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCsyr_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCsyr_v2_64(acblasHandle_t handle,
                                              acblasFillMode_t uplo, int64_t n,
                                              const acComplex *alpha,
                                              const acComplex *x, int64_t incx,
                                              acComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCsyr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrk3mEx(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k, const acComplex *alpha,
                const void *A, hggcDataType Atype, int lda,
                const acComplex *beta, void *C, hggcDataType Ctype, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCsyrk3mEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrk3mEx_64(acblasHandle_t handle, acblasFillMode_t uplo,
                   acblasOperation_t trans, int64_t n, int64_t k,
                   const acComplex *alpha, const void *A, hggcDataType Atype,
                   int64_t lda, const acComplex *beta, void *C,
                   hggcDataType Ctype, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCsyrk3mEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrkEx(acblasHandle_t handle, acblasFillMode_t uplo,
              acblasOperation_t trans, int n, int k, const acComplex *alpha,
              const void *A, hggcDataType Atype, int lda, const acComplex *beta,
              void *C, hggcDataType Ctype, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCsyrkEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrkEx_64(acblasHandle_t handle, acblasFillMode_t uplo,
                 acblasOperation_t trans, int64_t n, int64_t k,
                 const acComplex *alpha, const void *A, hggcDataType Atype,
                 int64_t lda, const acComplex *beta, void *C,
                 hggcDataType Ctype, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasCsyrkEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrk_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, int n, int k, const acComplex *alpha,
               const acComplex *A, int lda, const acComplex *beta, acComplex *C,
               int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsyrk_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrk_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, int64_t n, int64_t k,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  const acComplex *beta, acComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsyrk_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrkx(acblasHandle_t handle, acblasFillMode_t uplo,
             acblasOperation_t trans, int n, int k, const acComplex *alpha,
             const acComplex *A, int lda, const acComplex *B, int ldb,
             const acComplex *beta, acComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsyrkx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCsyrkx_64(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int64_t n, int64_t k,
                const acComplex *alpha, const acComplex *A, int64_t lda,
                const acComplex *B, int64_t ldb, const acComplex *beta,
                acComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCsyrkx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtbmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n, int k,
               const acComplex *A, int lda, acComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtbmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const acComplex *A, int64_t lda, acComplex *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtbsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n, int k,
               const acComplex *A, int lda, acComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtbsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtbsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const acComplex *A, int64_t lda, acComplex *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtbsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtpmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const acComplex *AP, acComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtpmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtpmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acComplex *AP, acComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtpmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtpsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const acComplex *AP, acComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtpsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtpsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acComplex *AP, acComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtpsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCtpttr(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const acComplex *AP, acComplex *A,
                                          int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)AP;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasCtpttr is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrmm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, acblasOperation_t trans,
               acblasDiagType_t diag, int m, int n, const acComplex *alpha,
               const acComplex *A, int lda, const acComplex *B, int ldb,
               acComplex *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCtrmm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrmm_v2_64(acblasHandle_t handle, acblasSideMode_t side,
                  acblasFillMode_t uplo, acblasOperation_t trans,
                  acblasDiagType_t diag, int64_t m, int64_t n,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  const acComplex *B, int64_t ldb, acComplex *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasCtrmm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const acComplex *A, int lda, acComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtrmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acComplex *A, int64_t lda, acComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtrmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrsmBatched(acblasHandle_t handle, acblasSideMode_t side,
                   acblasFillMode_t uplo, acblasOperation_t trans,
                   acblasDiagType_t diag, int m, int n, const acComplex *alpha,
                   const acComplex *const A[], int lda, acComplex *const B[],
                   int ldb, int batchCount) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)batchCount;
  fprintf(stderr, "acblasCtrsmBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCtrsmBatched_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, acblasDiagType_t diag, int64_t m, int64_t n,
    const acComplex *alpha, const acComplex *const A[], int64_t lda,
    acComplex *const B[], int64_t ldb, int64_t batchCount) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)batchCount;
  fprintf(stderr, "acblasCtrsmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrsm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, acblasOperation_t trans,
               acblasDiagType_t diag, int m, int n, const acComplex *alpha,
               const acComplex *A, int lda, acComplex *B, int ldb) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasCtrsm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrsm_v2_64(acblasHandle_t handle, acblasSideMode_t side,
                  acblasFillMode_t uplo, acblasOperation_t trans,
                  acblasDiagType_t diag, int64_t m, int64_t n,
                  const acComplex *alpha, const acComplex *A, int64_t lda,
                  acComplex *B, int64_t ldb) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasCtrsm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const acComplex *A, int lda, acComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtrsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasCtrsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acComplex *A, int64_t lda, acComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasCtrsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasCtrttp(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const acComplex *A, int lda,
                                          acComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)AP;
  fprintf(stderr, "acblasCtrttp is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDasum_v2_64(acblasHandle_t handle, int64_t n,
                                               const double *x, int64_t incx,
                                               double *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasDasum_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDaxpy_v2_64(acblasHandle_t handle, int64_t n,
                                               const double *alpha,
                                               const double *x, int64_t incx,
                                               double *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDaxpy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDcopy_v2_64(acblasHandle_t handle, int64_t n,
                                               const double *x, int64_t incx,
                                               double *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDcopy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDdgmm(acblasHandle_t handle,
                                         acblasSideMode_t mode, int m, int n,
                                         const double *A, int lda,
                                         const double *x, int incx, double *C,
                                         int ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDdgmm is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDdgmm_64(acblasHandle_t handle, acblasSideMode_t mode, int64_t m,
               int64_t n, const double *A, int64_t lda, const double *x,
               int64_t incx, double *C, int64_t ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDdgmm_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDdot_v2_64(acblasHandle_t handle, int64_t n,
                                              const double *x, int64_t incx,
                                              const double *y, int64_t incy,
                                              double *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasDdot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDgbmv_v2(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
               int kl, int ku, const double *alpha, const double *A, int lda,
               const double *x, int incx, const double *beta, double *y,
               int incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDgbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDgbmv_v2_64(acblasHandle_t handle, acblasOperation_t trans, int64_t m,
                  int64_t n, int64_t kl, int64_t ku, const double *alpha,
                  const double *A, int64_t lda, const double *x, int64_t incx,
                  const double *beta, double *y, int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDgbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDgeam_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda,
    const double *beta, const double *B, int64_t ldb, double *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDgeam_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDgemmBatched_64(acblasHandle_t handle, acblasOperation_t transa,
                      acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                      const double *alpha, const double *const Aarray[],
                      int64_t lda, const double *const Barray[], int64_t ldb,
                      const double *beta, double *const Carray[], int64_t ldc,
                      int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasDgemmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDgemmGroupedBatched(
    acblasHandle_t handle, const acblasOperation_t transa_array[],
    const acblasOperation_t transb_array[], const int m_array[],
    const int n_array[], const int k_array[], const double alpha_array[],
    const double *const Aarray[], const int lda_array[],
    const double *const Barray[], const int ldb_array[],
    const double beta_array[], double *const Carray[], const int ldc_array[],
    int group_count, const int group_size[]) {
  (void)handle;
  (void)transa_array;
  (void)transb_array;
  (void)m_array;
  (void)n_array;
  (void)k_array;
  (void)alpha_array;
  (void)Aarray;
  (void)lda_array;
  (void)Barray;
  (void)ldb_array;
  (void)beta_array;
  (void)Carray;
  (void)ldc_array;
  (void)group_count;
  (void)group_size;
  fprintf(stderr, "acblasDgemmGroupedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDgemmGroupedBatched_64(
    acblasHandle_t handle, const acblasOperation_t transa_array[],
    const acblasOperation_t transb_array[], const int64_t m_array[],
    const int64_t n_array[], const int64_t k_array[],
    const double alpha_array[], const double *const Aarray[],
    const int64_t lda_array[], const double *const Barray[],
    const int64_t ldb_array[], const double beta_array[],
    double *const Carray[], const int64_t ldc_array[], int64_t group_count,
    const int64_t group_size[]) {
  (void)handle;
  (void)transa_array;
  (void)transb_array;
  (void)m_array;
  (void)n_array;
  (void)k_array;
  (void)alpha_array;
  (void)Aarray;
  (void)lda_array;
  (void)Barray;
  (void)ldb_array;
  (void)beta_array;
  (void)Carray;
  (void)ldc_array;
  (void)group_count;
  (void)group_size;
  fprintf(stderr, "acblasDgemmGroupedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDgemmStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const double *alpha, const double *A,
    int64_t lda, long long strideA, const double *B, int64_t ldb,
    long long strideB, const double *beta, double *C, int64_t ldc,
    long long strideC, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasDgemmStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDgemm_v2_64(acblasHandle_t handle, acblasOperation_t transa,
                  acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                  const double *alpha, const double *A, int64_t lda,
                  const double *B, int64_t ldb, const double *beta, double *C,
                  int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDgemm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDgemvBatched(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
                   const double *alpha, const double *const Aarray[], int lda,
                   const double *const xarray[], int incx, const double *beta,
                   double *const yarray[], int incy, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasDgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const double *alpha, const double *const Aarray[], int64_t lda,
    const double *const xarray[], int64_t incx, const double *beta,
    double *const yarray[], int64_t incy, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasDgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const double *alpha, const double *A, int64_t lda, long long strideA,
    const double *x, int64_t incx, long long stridex, const double *beta,
    double *y, int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasDgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDgemv_v2_64(acblasHandle_t handle, acblasOperation_t trans, int64_t m,
                  int64_t n, const double *alpha, const double *A, int64_t lda,
                  const double *x, int64_t incx, const double *beta, double *y,
                  int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDgemv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDger_v2_64(acblasHandle_t handle, int64_t m,
                                              int64_t n, const double *alpha,
                                              const double *x, int64_t incx,
                                              const double *y, int64_t incy,
                                              double *A, int64_t lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasDger_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDgetriBatched(acblasHandle_t handle, int n,
                                                 const double *const A[],
                                                 int lda, const int *P,
                                                 double *const C[], int ldc,
                                                 int *info, int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)P;
  (void)C;
  (void)ldc;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasDgetriBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDmatinvBatched(acblasHandle_t handle, int n,
                                                  const double *const A[],
                                                  int lda, double *const Ainv[],
                                                  int lda_inv, int *info,
                                                  int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)Ainv;
  (void)lda_inv;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasDmatinvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDnrm2_v2_64(acblasHandle_t handle, int64_t n,
                                               const double *x, int64_t incx,
                                               double *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasDnrm2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDotEx_64(acblasHandle_t handle, int64_t n, const void *x,
               hggcDataType xType, int64_t incx, const void *y,
               hggcDataType yType, int64_t incy, void *result,
               hggcDataType resultType, hggcDataType executionType) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  (void)result;
  (void)resultType;
  (void)executionType;
  fprintf(stderr, "acblasDotEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDotcEx(acblasHandle_t handle, int n,
                                          const void *x, hggcDataType xType,
                                          int incx, const void *y,
                                          hggcDataType yType, int incy,
                                          void *result, hggcDataType resultType,
                                          hggcDataType executionType) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  (void)result;
  (void)resultType;
  (void)executionType;
  fprintf(stderr, "acblasDotcEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDotcEx_64(acblasHandle_t handle, int64_t n, const void *x,
                hggcDataType xType, int64_t incx, const void *y,
                hggcDataType yType, int64_t incy, void *result,
                hggcDataType resultType, hggcDataType executionType) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  (void)result;
  (void)resultType;
  (void)executionType;
  fprintf(stderr, "acblasDotcEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDrot_v2_64(acblasHandle_t handle, int64_t n, double *x, int64_t incx,
                 double *y, int64_t incy, const double *c, const double *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasDrot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDrotm_v2_64(acblasHandle_t handle, int64_t n,
                                               double *x, int64_t incx,
                                               double *y, int64_t incy,
                                               const double *param) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)param;
  fprintf(stderr, "acblasDrotm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsbmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n, int k,
               const double *alpha, const double *A, int lda, const double *x,
               int incx, const double *beta, double *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDsbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsbmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  int64_t k, const double *alpha, const double *A, int64_t lda,
                  const double *x, int64_t incx, const double *beta, double *y,
                  int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDsbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDscal_v2_64(acblasHandle_t handle, int64_t n,
                                               const double *alpha, double *x,
                                               int64_t incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDscal_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDspmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const double *alpha, const double *AP, const double *x, int incx,
               const double *beta, double *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDspmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDspmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const double *alpha, const double *AP, const double *x,
                  int64_t incx, const double *beta, double *y, int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDspmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDspr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const double *alpha, const double *x, int64_t incx,
                  const double *y, int64_t incy, double *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)AP;
  fprintf(stderr, "acblasDspr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDspr_v2_64(acblasHandle_t handle,
                                              acblasFillMode_t uplo, int64_t n,
                                              const double *alpha,
                                              const double *x, int64_t incx,
                                              double *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)AP;
  fprintf(stderr, "acblasDspr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDswap_v2_64(acblasHandle_t handle, int64_t n,
                                               double *x, int64_t incx,
                                               double *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDswap_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsymm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, int m, int n, const double *alpha,
               const double *A, int lda, const double *B, int ldb,
               const double *beta, double *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsymm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDsymm_v2_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    int64_t m, int64_t n, const double *alpha, const double *A, int64_t lda,
    const double *B, int64_t ldb, const double *beta, double *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsymm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsymv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const double *alpha, const double *A, int lda, const double *x,
               int incx, const double *beta, double *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDsymv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsymv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const double *alpha, const double *A, int64_t lda,
                  const double *x, int64_t incx, const double *beta, double *y,
                  int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasDsymv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsyr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const double *alpha, const double *x, int64_t incx,
                  const double *y, int64_t incy, double *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasDsyr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsyr2k_v2(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k, const double *alpha,
                const double *A, int lda, const double *B, int ldb,
                const double *beta, double *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsyr2k_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDsyr2k_v2_64(
    acblasHandle_t handle, acblasFillMode_t uplo, acblasOperation_t trans,
    int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda,
    const double *B, int64_t ldb, const double *beta, double *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsyr2k_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDsyr_v2_64(acblasHandle_t handle,
                                              acblasFillMode_t uplo, int64_t n,
                                              const double *alpha,
                                              const double *x, int64_t incx,
                                              double *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasDsyr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsyrk_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, int n, int k, const double *alpha,
               const double *A, int lda, const double *beta, double *C,
               int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsyrk_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsyrk_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, int64_t n, int64_t k,
                  const double *alpha, const double *A, int64_t lda,
                  const double *beta, double *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsyrk_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDsyrkx(acblasHandle_t handle, acblasFillMode_t uplo,
             acblasOperation_t trans, int n, int k, const double *alpha,
             const double *A, int lda, const double *B, int ldb,
             const double *beta, double *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsyrkx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDsyrkx_64(
    acblasHandle_t handle, acblasFillMode_t uplo, acblasOperation_t trans,
    int64_t n, int64_t k, const double *alpha, const double *A, int64_t lda,
    const double *B, int64_t ldb, const double *beta, double *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDsyrkx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtbmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n, int k,
               const double *A, int lda, double *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtbmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const double *A, int64_t lda, double *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtbsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n, int k,
               const double *A, int lda, double *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtbsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtbsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const double *A, int64_t lda, double *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtbsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtpmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const double *AP, double *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtpmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtpmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const double *AP, double *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtpmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtpsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const double *AP, double *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtpsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtpsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const double *AP, double *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtpsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDtpttr(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const double *AP, double *A,
                                          int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)AP;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasDtpttr is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtrmm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, acblasOperation_t trans,
               acblasDiagType_t diag, int m, int n, const double *alpha,
               const double *A, int lda, const double *B, int ldb, double *C,
               int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDtrmm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtrmm_v2_64(acblasHandle_t handle, acblasSideMode_t side,
                  acblasFillMode_t uplo, acblasOperation_t trans,
                  acblasDiagType_t diag, int64_t m, int64_t n,
                  const double *alpha, const double *A, int64_t lda,
                  const double *B, int64_t ldb, double *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasDtrmm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtrmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const double *A, int lda, double *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtrmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtrmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const double *A, int64_t lda, double *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtrmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtrsmBatched_64(acblasHandle_t handle, acblasSideMode_t side,
                      acblasFillMode_t uplo, acblasOperation_t trans,
                      acblasDiagType_t diag, int64_t m, int64_t n,
                      const double *alpha, const double *const A[], int64_t lda,
                      double *const B[], int64_t ldb, int64_t batchCount) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)batchCount;
  fprintf(stderr, "acblasDtrsmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDtrsm_v2_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, acblasDiagType_t diag, int64_t m, int64_t n,
    const double *alpha, const double *A, int64_t lda, double *B, int64_t ldb) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasDtrsm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtrsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const double *A, int lda, double *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtrsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasDtrsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const double *A, int64_t lda, double *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasDtrsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDtrttp(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const double *A, int lda,
                                          double *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)AP;
  fprintf(stderr, "acblasDtrttp is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDzasum_v2(acblasHandle_t handle, int n,
                                             const acDoubleComplex *x, int incx,
                                             double *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasDzasum_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDzasum_v2_64(acblasHandle_t handle,
                                                int64_t n,
                                                const acDoubleComplex *x,
                                                int64_t incx, double *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasDzasum_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDznrm2_v2(acblasHandle_t handle, int n,
                                             const acDoubleComplex *x, int incx,
                                             double *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasDznrm2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasDznrm2_v2_64(acblasHandle_t handle,
                                                int64_t n,
                                                const acDoubleComplex *x,
                                                int64_t incx, double *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasDznrm2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasGemmBatchedEx_64(acblasHandle_t handle, acblasOperation_t transa,
                       acblasOperation_t transb, int64_t m, int64_t n,
                       int64_t k, const void *alpha, const void *const Aarray[],
                       hggcDataType Atype, int64_t lda,
                       const void *const Barray[], hggcDataType Btype,
                       int64_t ldb, const void *beta, void *const Carray[],
                       hggcDataType Ctype, int64_t ldc, int64_t batchCount,
                       acblasComputeType_t computeType, acblasGemmAlgo_t algo) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)Atype;
  (void)lda;
  (void)Barray;
  (void)Btype;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)Ctype;
  (void)ldc;
  (void)batchCount;
  (void)computeType;
  (void)algo;
  fprintf(stderr, "acblasGemmBatchedEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasGemmEx_64(acblasHandle_t handle, acblasOperation_t transa,
                acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                const void *alpha, const void *A, hggcDataType Atype,
                int64_t lda, const void *B, hggcDataType Btype, int64_t ldb,
                const void *beta, void *C, hggcDataType Ctype, int64_t ldc,
                acblasComputeType_t computeType, acblasGemmAlgo_t algo) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)B;
  (void)Btype;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  (void)computeType;
  (void)algo;
  fprintf(stderr, "acblasGemmEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGemmGroupedBatchedEx(
    acblasHandle_t handle, const acblasOperation_t transa_array[],
    const acblasOperation_t transb_array[], const int m_array[],
    const int n_array[], const int k_array[], const void *alpha_array,
    const void *const Aarray[], hggcDataType_t Atype, const int lda_array[],
    const void *const Barray[], hggcDataType_t Btype, const int ldb_array[],
    const void *beta_array, void *const Carray[], hggcDataType_t Ctype,
    const int ldc_array[], int group_count, const int group_size[],
    acblasComputeType_t computeType) {
  (void)handle;
  (void)transa_array;
  (void)transb_array;
  (void)m_array;
  (void)n_array;
  (void)k_array;
  (void)alpha_array;
  (void)Aarray;
  (void)Atype;
  (void)lda_array;
  (void)Barray;
  (void)Btype;
  (void)ldb_array;
  (void)beta_array;
  (void)Carray;
  (void)Ctype;
  (void)ldc_array;
  (void)group_count;
  (void)group_size;
  (void)computeType;
  fprintf(stderr, "acblasGemmGroupedBatchedEx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGemmGroupedBatchedEx_64(
    acblasHandle_t handle, const acblasOperation_t transa_array[],
    const acblasOperation_t transb_array[], const int64_t m_array[],
    const int64_t n_array[], const int64_t k_array[], const void *alpha_array,
    const void *const Aarray[], hggcDataType_t Atype, const int64_t lda_array[],
    const void *const Barray[], hggcDataType_t Btype, const int64_t ldb_array[],
    const void *beta_array, void *const Carray[], hggcDataType_t Ctype,
    const int64_t ldc_array[], int64_t group_count, const int64_t group_size[],
    acblasComputeType_t computeType) {
  (void)handle;
  (void)transa_array;
  (void)transb_array;
  (void)m_array;
  (void)n_array;
  (void)k_array;
  (void)alpha_array;
  (void)Aarray;
  (void)Atype;
  (void)lda_array;
  (void)Barray;
  (void)Btype;
  (void)ldb_array;
  (void)beta_array;
  (void)Carray;
  (void)Ctype;
  (void)ldc_array;
  (void)group_count;
  (void)group_size;
  (void)computeType;
  fprintf(stderr, "acblasGemmGroupedBatchedEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGemmStridedBatchedEx_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const void *alpha, const void *A,
    hggcDataType Atype, int64_t lda, long long strideA, const void *B,
    hggcDataType Btype, int64_t ldb, long long strideB, const void *beta,
    void *C, hggcDataType Ctype, int64_t ldc, long long strideC,
    int64_t batchCount, acblasComputeType_t computeType,
    acblasGemmAlgo_t algo) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)Btype;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  (void)computeType;
  (void)algo;
  fprintf(stderr, "acblasGemmStridedBatchedEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGetMatrixAsync(int rows, int cols,
                                                  int elemSize, const void *A,
                                                  int lda, void *B, int ldb,
                                                  hggcStream_t stream) {
  (void)rows;
  (void)cols;
  (void)elemSize;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)stream;
  fprintf(stderr, "acblasGetMatrixAsync is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGetMatrixAsync_64(int64_t rows, int64_t cols,
                                                     int64_t elemSize,
                                                     const void *A, int64_t lda,
                                                     void *B, int64_t ldb,
                                                     hggcStream_t stream) {
  (void)rows;
  (void)cols;
  (void)elemSize;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)stream;
  fprintf(stderr, "acblasGetMatrixAsync_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGetMatrix_64(int64_t rows, int64_t cols,
                                                int64_t elemSize, const void *A,
                                                int64_t lda, void *B,
                                                int64_t ldb) {
  (void)rows;
  (void)cols;
  (void)elemSize;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasGetMatrix_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGetSmCountTarget(acblasHandle_t handle,
                                                    int *smCountTarget) {
  (void)handle;
  (void)smCountTarget;
  fprintf(stderr, "acblasGetSmCountTarget is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline const char *acblasGetStatusName(acblasStatus_t status) {
  (void)status;
  fprintf(stderr, "acblasGetStatusName is not supported.\n");
  exit(1);
  return (const char *)0; /* unreachable */
}

static inline acblasStatus_t
acblasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx,
                     void *hostPtr, int incy, hggcStream_t stream) {
  (void)n;
  (void)elemSize;
  (void)devicePtr;
  (void)incx;
  (void)hostPtr;
  (void)incy;
  (void)stream;
  fprintf(stderr, "acblasGetVectorAsync is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasGetVectorAsync_64(int64_t n, int64_t elemSize, const void *devicePtr,
                        int64_t incx, void *hostPtr, int64_t incy,
                        hggcStream_t stream) {
  (void)n;
  (void)elemSize;
  (void)devicePtr;
  (void)incx;
  (void)hostPtr;
  (void)incy;
  (void)stream;
  fprintf(stderr, "acblasGetVectorAsync_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasGetVector_64(int64_t n, int64_t elemSize,
                                                const void *x, int64_t incx,
                                                void *y, int64_t incy) {
  (void)n;
  (void)elemSize;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasGetVector_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasHSHgemvBatched(acblasHandle_t handle, acblasOperation_t trans, int m,
                     int n, const float *alpha, const __half *const Aarray[],
                     int lda, const __half *const xarray[], int incx,
                     const float *beta, __half *const yarray[], int incy,
                     int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasHSHgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasHSHgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *const Aarray[], int64_t lda,
    const __half *const xarray[], int64_t incx, const float *beta,
    __half *const yarray[], int64_t incy, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasHSHgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasHSHgemvStridedBatched(
    acblasHandle_t handle, acblasOperation_t trans, int m, int n,
    const float *alpha, const __half *A, int lda, long long strideA,
    const __half *x, int incx, long long stridex, const float *beta, __half *y,
    int incy, long long stridey, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasHSHgemvStridedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasHSHgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *A, int64_t lda, long long strideA,
    const __half *x, int64_t incx, long long stridex, const float *beta,
    __half *y, int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasHSHgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasHSSgemvBatched(acblasHandle_t handle, acblasOperation_t trans, int m,
                     int n, const float *alpha, const __half *const Aarray[],
                     int lda, const __half *const xarray[], int incx,
                     const float *beta, float *const yarray[], int incy,
                     int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasHSSgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasHSSgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *const Aarray[], int64_t lda,
    const __half *const xarray[], int64_t incx, const float *beta,
    float *const yarray[], int64_t incy, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasHSSgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasHSSgemvStridedBatched(
    acblasHandle_t handle, acblasOperation_t trans, int m, int n,
    const float *alpha, const __half *A, int lda, long long strideA,
    const __half *x, int incx, long long stridex, const float *beta, float *y,
    int incy, long long stridey, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasHSSgemvStridedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasHSSgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __half *A, int64_t lda, long long strideA,
    const __half *x, int64_t incx, long long stridex, const float *beta,
    float *y, int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasHSSgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasHgemmBatched_64(acblasHandle_t handle, acblasOperation_t transa,
                      acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                      const __half *alpha, const __half *const Aarray[],
                      int64_t lda, const __half *const Barray[], int64_t ldb,
                      const __half *beta, __half *const Carray[], int64_t ldc,
                      int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasHgemmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasHgemmStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const __half *alpha, const __half *A,
    int64_t lda, long long strideA, const __half *B, int64_t ldb,
    long long strideB, const __half *beta, __half *C, int64_t ldc,
    long long strideC, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasHgemmStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasHgemm_64(acblasHandle_t handle, acblasOperation_t transa,
               acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
               const __half *alpha, const __half *A, int64_t lda,
               const __half *B, int64_t ldb, const __half *beta, __half *C,
               int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasHgemm_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIamaxEx_64(acblasHandle_t handle, int64_t n,
                                              const void *x, hggcDataType xType,
                                              int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIamaxEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIaminEx_64(acblasHandle_t handle, int64_t n,
                                              const void *x, hggcDataType xType,
                                              int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIaminEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIcamax_v2(acblasHandle_t handle, int n,
                                             const acComplex *x, int incx,
                                             int *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIcamax_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIcamax_v2_64(acblasHandle_t handle,
                                                int64_t n, const acComplex *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIcamax_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIcamin_v2(acblasHandle_t handle, int n,
                                             const acComplex *x, int incx,
                                             int *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIcamin_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIcamin_v2_64(acblasHandle_t handle,
                                                int64_t n, const acComplex *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIcamin_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIdamax_v2_64(acblasHandle_t handle,
                                                int64_t n, const double *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIdamax_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIdamin_v2_64(acblasHandle_t handle,
                                                int64_t n, const double *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIdamin_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIsamax_v2_64(acblasHandle_t handle,
                                                int64_t n, const float *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIsamax_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIsamin_v2_64(acblasHandle_t handle,
                                                int64_t n, const float *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIsamin_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIzamax_v2(acblasHandle_t handle, int n,
                                             const acDoubleComplex *x, int incx,
                                             int *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIzamax_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIzamax_v2_64(acblasHandle_t handle,
                                                int64_t n,
                                                const acDoubleComplex *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIzamax_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIzamin_v2(acblasHandle_t handle, int n,
                                             const acDoubleComplex *x, int incx,
                                             int *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIzamin_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasIzamin_v2_64(acblasHandle_t handle,
                                                int64_t n,
                                                const acDoubleComplex *x,
                                                int64_t incx, int64_t *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasIzamin_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasLoggerConfigure(int logIsOn, int logToStdOut,
                                                   int logToStdErr,
                                                   const char *logFileName) {
  (void)logIsOn;
  (void)logToStdOut;
  (void)logToStdErr;
  (void)logFileName;
  fprintf(stderr, "acblasLoggerConfigure is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasNrm2Ex_64(acblasHandle_t handle, int64_t n,
                                             const void *x, hggcDataType xType,
                                             int64_t incx, void *result,
                                             hggcDataType resultType,
                                             hggcDataType executionType) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)result;
  (void)resultType;
  (void)executionType;
  fprintf(stderr, "acblasNrm2Ex_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasRotEx_64(acblasHandle_t handle, int64_t n, void *x, hggcDataType xType,
               int64_t incx, void *y, hggcDataType yType, int64_t incy,
               const void *c, const void *s, hggcDataType csType,
               hggcDataType executiontype) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  (void)c;
  (void)s;
  (void)csType;
  (void)executiontype;
  fprintf(stderr, "acblasRotEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasRotmEx_64(acblasHandle_t handle, int64_t n, void *x, hggcDataType xType,
                int64_t incx, void *y, hggcDataType yType, int64_t incy,
                const void *param, hggcDataType paramType,
                hggcDataType executiontype) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  (void)param;
  (void)paramType;
  (void)executiontype;
  fprintf(stderr, "acblasRotmEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSasum_v2_64(acblasHandle_t handle, int64_t n,
                                               const float *x, int64_t incx,
                                               float *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasSasum_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSaxpy_v2_64(acblasHandle_t handle, int64_t n,
                                               const float *alpha,
                                               const float *x, int64_t incx,
                                               float *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSaxpy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasScalEx_64(acblasHandle_t handle, int64_t n,
                                             const void *alpha,
                                             hggcDataType alphaType, void *x,
                                             hggcDataType xType, int64_t incx,
                                             hggcDataType executionType) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)alphaType;
  (void)x;
  (void)xType;
  (void)incx;
  (void)executionType;
  fprintf(stderr, "acblasScalEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasScasum_v2(acblasHandle_t handle, int n,
                                             const acComplex *x, int incx,
                                             float *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasScasum_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasScasum_v2_64(acblasHandle_t handle,
                                                int64_t n, const acComplex *x,
                                                int64_t incx, float *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasScasum_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasScnrm2_v2(acblasHandle_t handle, int n,
                                             const acComplex *x, int incx,
                                             float *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasScnrm2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasScnrm2_v2_64(acblasHandle_t handle,
                                                int64_t n, const acComplex *x,
                                                int64_t incx, float *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasScnrm2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasScopy_v2_64(acblasHandle_t handle, int64_t n,
                                               const float *x, int64_t incx,
                                               float *y, int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasScopy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSdgmm(acblasHandle_t handle,
                                         acblasSideMode_t mode, int m, int n,
                                         const float *A, int lda,
                                         const float *x, int incx, float *C,
                                         int ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSdgmm is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSdgmm_64(acblasHandle_t handle, acblasSideMode_t mode, int64_t m,
               int64_t n, const float *A, int64_t lda, const float *x,
               int64_t incx, float *C, int64_t ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSdgmm_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSdot_v2_64(acblasHandle_t handle, int64_t n,
                                              const float *x, int64_t incx,
                                              const float *y, int64_t incy,
                                              float *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasSdot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSetMatrixAsync(int rows, int cols,
                                                  int elemSize, const void *A,
                                                  int lda, void *B, int ldb,
                                                  hggcStream_t stream) {
  (void)rows;
  (void)cols;
  (void)elemSize;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)stream;
  fprintf(stderr, "acblasSetMatrixAsync is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSetMatrixAsync_64(int64_t rows, int64_t cols,
                                                     int64_t elemSize,
                                                     const void *A, int64_t lda,
                                                     void *B, int64_t ldb,
                                                     hggcStream_t stream) {
  (void)rows;
  (void)cols;
  (void)elemSize;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)stream;
  fprintf(stderr, "acblasSetMatrixAsync_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSetMatrix_64(int64_t rows, int64_t cols,
                                                int64_t elemSize, const void *A,
                                                int64_t lda, void *B,
                                                int64_t ldb) {
  (void)rows;
  (void)cols;
  (void)elemSize;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasSetMatrix_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSetSmCountTarget(acblasHandle_t handle,
                                                    int smCountTarget) {
  (void)handle;
  (void)smCountTarget;
  fprintf(stderr, "acblasSetSmCountTarget is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSetVectorAsync(int n, int elemSize,
                                                  const void *hostPtr, int incx,
                                                  void *devicePtr, int incy,
                                                  hggcStream_t stream) {
  (void)n;
  (void)elemSize;
  (void)hostPtr;
  (void)incx;
  (void)devicePtr;
  (void)incy;
  (void)stream;
  fprintf(stderr, "acblasSetVectorAsync is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSetVectorAsync_64(int64_t n, int64_t elemSize, const void *hostPtr,
                        int64_t incx, void *devicePtr, int64_t incy,
                        hggcStream_t stream) {
  (void)n;
  (void)elemSize;
  (void)hostPtr;
  (void)incx;
  (void)devicePtr;
  (void)incy;
  (void)stream;
  fprintf(stderr, "acblasSetVectorAsync_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSetVector_64(int64_t n, int64_t elemSize,
                                                const void *x, int64_t incx,
                                                void *devicePtr, int64_t incy) {
  (void)n;
  (void)elemSize;
  (void)x;
  (void)incx;
  (void)devicePtr;
  (void)incy;
  fprintf(stderr, "acblasSetVector_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSgbmv_v2(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
               int kl, int ku, const float *alpha, const float *A, int lda,
               const float *x, int incx, const float *beta, float *y,
               int incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSgbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSgbmv_v2_64(acblasHandle_t handle, acblasOperation_t trans, int64_t m,
                  int64_t n, int64_t kl, int64_t ku, const float *alpha,
                  const float *A, int64_t lda, const float *x, int64_t incx,
                  const float *beta, float *y, int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSgbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSgeam_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda,
    const float *beta, const float *B, int64_t ldb, float *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSgeam_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSgemmBatched_64(acblasHandle_t handle, acblasOperation_t transa,
                      acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                      const float *alpha, const float *const Aarray[],
                      int64_t lda, const float *const Barray[], int64_t ldb,
                      const float *beta, float *const Carray[], int64_t ldc,
                      int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasSgemmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSgemmEx_64(acblasHandle_t handle, acblasOperation_t transa,
                 acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                 const float *alpha, const void *A, hggcDataType Atype,
                 int64_t lda, const void *B, hggcDataType Btype, int64_t ldb,
                 const float *beta, void *C, hggcDataType Ctype, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)Atype;
  (void)lda;
  (void)B;
  (void)Btype;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)Ctype;
  (void)ldc;
  fprintf(stderr, "acblasSgemmEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSgemmGroupedBatched(
    acblasHandle_t handle, const acblasOperation_t transa_array[],
    const acblasOperation_t transb_array[], const int m_array[],
    const int n_array[], const int k_array[], const float alpha_array[],
    const float *const Aarray[], const int lda_array[],
    const float *const Barray[], const int ldb_array[],
    const float beta_array[], float *const Carray[], const int ldc_array[],
    int group_count, const int group_size[]) {
  (void)handle;
  (void)transa_array;
  (void)transb_array;
  (void)m_array;
  (void)n_array;
  (void)k_array;
  (void)alpha_array;
  (void)Aarray;
  (void)lda_array;
  (void)Barray;
  (void)ldb_array;
  (void)beta_array;
  (void)Carray;
  (void)ldc_array;
  (void)group_count;
  (void)group_size;
  fprintf(stderr, "acblasSgemmGroupedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSgemmGroupedBatched_64(
    acblasHandle_t handle, const acblasOperation_t transa_array[],
    const acblasOperation_t transb_array[], const int64_t m_array[],
    const int64_t n_array[], const int64_t k_array[], const float alpha_array[],
    const float *const Aarray[], const int64_t lda_array[],
    const float *const Barray[], const int64_t ldb_array[],
    const float beta_array[], float *const Carray[], const int64_t ldc_array[],
    int64_t group_count, const int64_t group_size[]) {
  (void)handle;
  (void)transa_array;
  (void)transb_array;
  (void)m_array;
  (void)n_array;
  (void)k_array;
  (void)alpha_array;
  (void)Aarray;
  (void)lda_array;
  (void)Barray;
  (void)ldb_array;
  (void)beta_array;
  (void)Carray;
  (void)ldc_array;
  (void)group_count;
  (void)group_size;
  fprintf(stderr, "acblasSgemmGroupedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSgemmStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const float *alpha, const float *A,
    int64_t lda, long long strideA, const float *B, int64_t ldb,
    long long strideB, const float *beta, float *C, int64_t ldc,
    long long strideC, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasSgemmStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSgemm_v2_64(acblasHandle_t handle, acblasOperation_t transa,
                  acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                  const float *alpha, const float *A, int64_t lda,
                  const float *B, int64_t ldb, const float *beta, float *C,
                  int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSgemm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSgemvBatched(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
                   const float *alpha, const float *const Aarray[], int lda,
                   const float *const xarray[], int incx, const float *beta,
                   float *const yarray[], int incy, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasSgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const float *const Aarray[], int64_t lda,
    const float *const xarray[], int64_t incx, const float *beta,
    float *const yarray[], int64_t incy, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasSgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const float *A, int64_t lda, long long strideA,
    const float *x, int64_t incx, long long stridex, const float *beta,
    float *y, int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasSgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSgemv_v2_64(acblasHandle_t handle, acblasOperation_t trans, int64_t m,
                  int64_t n, const float *alpha, const float *A, int64_t lda,
                  const float *x, int64_t incx, const float *beta, float *y,
                  int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSgemv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSger_v2_64(acblasHandle_t handle, int64_t m,
                                              int64_t n, const float *alpha,
                                              const float *x, int64_t incx,
                                              const float *y, int64_t incy,
                                              float *A, int64_t lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasSger_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSmatinvBatched(acblasHandle_t handle, int n,
                                                  const float *const A[],
                                                  int lda, float *const Ainv[],
                                                  int lda_inv, int *info,
                                                  int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)Ainv;
  (void)lda_inv;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasSmatinvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSnrm2_v2_64(acblasHandle_t handle, int64_t n,
                                               const float *x, int64_t incx,
                                               float *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)result;
  fprintf(stderr, "acblasSnrm2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSrot_v2_64(acblasHandle_t handle, int64_t n,
                                              float *x, int64_t incx, float *y,
                                              int64_t incy, const float *c,
                                              const float *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasSrot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSrotm_v2_64(acblasHandle_t handle, int64_t n,
                                               float *x, int64_t incx, float *y,
                                               int64_t incy,
                                               const float *param) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)param;
  fprintf(stderr, "acblasSrotm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsbmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n, int k,
               const float *alpha, const float *A, int lda, const float *x,
               int incx, const float *beta, float *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSsbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsbmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  int64_t k, const float *alpha, const float *A, int64_t lda,
                  const float *x, int64_t incx, const float *beta, float *y,
                  int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSsbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSscal_v2_64(acblasHandle_t handle, int64_t n,
                                               const float *alpha, float *x,
                                               int64_t incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasSscal_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSspmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const float *alpha, const float *AP, const float *x, int incx,
               const float *beta, float *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSspmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSspmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const float *alpha, const float *AP, const float *x,
                  int64_t incx, const float *beta, float *y, int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSspmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSspr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const float *alpha, const float *x, int64_t incx,
                  const float *y, int64_t incy, float *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)AP;
  fprintf(stderr, "acblasSspr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSspr_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                 const float *alpha, const float *x, int64_t incx, float *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)AP;
  fprintf(stderr, "acblasSspr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSswap_v2_64(acblasHandle_t handle, int64_t n,
                                               float *x, int64_t incx, float *y,
                                               int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSswap_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsymm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, int m, int n, const float *alpha,
               const float *A, int lda, const float *B, int ldb,
               const float *beta, float *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsymm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSsymm_v2_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    int64_t m, int64_t n, const float *alpha, const float *A, int64_t lda,
    const float *B, int64_t ldb, const float *beta, float *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsymm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsymv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const float *alpha, const float *A, int lda, const float *x,
               int incx, const float *beta, float *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSsymv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsymv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const float *alpha, const float *A, int64_t lda,
                  const float *x, int64_t incx, const float *beta, float *y,
                  int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasSsymv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsyr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const float *alpha, const float *x, int64_t incx,
                  const float *y, int64_t incy, float *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasSsyr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsyr2k_v2(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k, const float *alpha,
                const float *A, int lda, const float *B, int ldb,
                const float *beta, float *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsyr2k_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSsyr2k_v2_64(
    acblasHandle_t handle, acblasFillMode_t uplo, acblasOperation_t trans,
    int64_t n, int64_t k, const float *alpha, const float *A, int64_t lda,
    const float *B, int64_t ldb, const float *beta, float *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsyr2k_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSsyr_v2_64(acblasHandle_t handle,
                                              acblasFillMode_t uplo, int64_t n,
                                              const float *alpha,
                                              const float *x, int64_t incx,
                                              float *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasSsyr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsyrk_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, int n, int k, const float *alpha,
               const float *A, int lda, const float *beta, float *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsyrk_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsyrk_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, int64_t n, int64_t k,
                  const float *alpha, const float *A, int64_t lda,
                  const float *beta, float *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsyrk_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsyrkx(acblasHandle_t handle, acblasFillMode_t uplo,
             acblasOperation_t trans, int n, int k, const float *alpha,
             const float *A, int lda, const float *B, int ldb,
             const float *beta, float *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsyrkx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasSsyrkx_64(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int64_t n, int64_t k,
                const float *alpha, const float *A, int64_t lda, const float *B,
                int64_t ldb, const float *beta, float *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasSsyrkx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStbmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n, int k,
               const float *A, int lda, float *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStbmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const float *A, int64_t lda, float *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStbsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n, int k,
               const float *A, int lda, float *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStbsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStbsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const float *A, int64_t lda, float *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStbsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStpmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const float *AP, float *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStpmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStpmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const float *AP, float *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStpmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStpsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const float *AP, float *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStpsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStpsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const float *AP, float *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStpsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasStpttr(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const float *AP, float *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)AP;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasStpttr is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStrmm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, acblasOperation_t trans,
               acblasDiagType_t diag, int m, int n, const float *alpha,
               const float *A, int lda, const float *B, int ldb, float *C,
               int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasStrmm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStrmm_v2_64(acblasHandle_t handle, acblasSideMode_t side,
                  acblasFillMode_t uplo, acblasOperation_t trans,
                  acblasDiagType_t diag, int64_t m, int64_t n,
                  const float *alpha, const float *A, int64_t lda,
                  const float *B, int64_t ldb, float *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasStrmm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStrmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const float *A, int lda, float *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStrmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStrmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const float *A, int64_t lda, float *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStrmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStrsmBatched_64(acblasHandle_t handle, acblasSideMode_t side,
                      acblasFillMode_t uplo, acblasOperation_t trans,
                      acblasDiagType_t diag, int64_t m, int64_t n,
                      const float *alpha, const float *const A[], int64_t lda,
                      float *const B[], int64_t ldb, int64_t batchCount) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)batchCount;
  fprintf(stderr, "acblasStrsmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasStrsm_v2_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, acblasDiagType_t diag, int64_t m, int64_t n,
    const float *alpha, const float *A, int64_t lda, float *B, int64_t ldb) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasStrsm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStrsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const float *A, int lda, float *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStrsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasStrsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const float *A, int64_t lda, float *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasStrsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasStrttp(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const float *A, int lda, float *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)AP;
  fprintf(stderr, "acblasStrttp is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasSwapEx_64(acblasHandle_t handle, int64_t n,
                                             void *x, hggcDataType xType,
                                             int64_t incx, void *y,
                                             hggcDataType yType, int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)xType;
  (void)incx;
  (void)y;
  (void)yType;
  (void)incy;
  fprintf(stderr, "acblasSwapEx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSSgemvBatched(
    acblasHandle_t handle, acblasOperation_t trans, int m, int n,
    const float *alpha, const __ppu_bfloat16 *const Aarray[], int lda,
    const __ppu_bfloat16 *const xarray[], int incx, const float *beta,
    float *const yarray[], int incy, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasTSSgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSSgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __ppu_bfloat16 *const Aarray[], int64_t lda,
    const __ppu_bfloat16 *const xarray[], int64_t incx, const float *beta,
    float *const yarray[], int64_t incy, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasTSSgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSSgemvStridedBatched(
    acblasHandle_t handle, acblasOperation_t trans, int m, int n,
    const float *alpha, const __ppu_bfloat16 *A, int lda, long long strideA,
    const __ppu_bfloat16 *x, int incx, long long stridex, const float *beta,
    float *y, int incy, long long stridey, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasTSSgemvStridedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSSgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __ppu_bfloat16 *A, int64_t lda, long long strideA,
    const __ppu_bfloat16 *x, int64_t incx, long long stridex, const float *beta,
    float *y, int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasTSSgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSTgemvBatched(
    acblasHandle_t handle, acblasOperation_t trans, int m, int n,
    const float *alpha, const __ppu_bfloat16 *const Aarray[], int lda,
    const __ppu_bfloat16 *const xarray[], int incx, const float *beta,
    __ppu_bfloat16 *const yarray[], int incy, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasTSTgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSTgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __ppu_bfloat16 *const Aarray[], int64_t lda,
    const __ppu_bfloat16 *const xarray[], int64_t incx, const float *beta,
    __ppu_bfloat16 *const yarray[], int64_t incy, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasTSTgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSTgemvStridedBatched(
    acblasHandle_t handle, acblasOperation_t trans, int m, int n,
    const float *alpha, const __ppu_bfloat16 *A, int lda, long long strideA,
    const __ppu_bfloat16 *x, int incx, long long stridex, const float *beta,
    __ppu_bfloat16 *y, int incy, long long stridey, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasTSTgemvStridedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasTSTgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const float *alpha, const __ppu_bfloat16 *A, int64_t lda, long long strideA,
    const __ppu_bfloat16 *x, int64_t incx, long long stridex, const float *beta,
    __ppu_bfloat16 *y, int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasTSTgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasUint8gemmBias(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    acblasOperation_t transc, int m, int n, int k, const unsigned char *A,
    int A_bias, int lda, const unsigned char *B, int B_bias, int ldb,
    unsigned char *C, int C_bias, int ldc, int C_mult, int C_shift) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)transc;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)A_bias;
  (void)lda;
  (void)B;
  (void)B_bias;
  (void)ldb;
  (void)C;
  (void)C_bias;
  (void)ldc;
  (void)C_mult;
  (void)C_shift;
  fprintf(stderr, "acblasUint8gemmBias is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline void acblasXerbla(const char *srName, int info) {
  (void)srName;
  (void)info;
  fprintf(stderr, "acblasXerbla is not supported.\n");
  exit(1);
}

static inline acblasStatus_t acblasXtGetNumBoards(int nbDevices, int deviceId[],
                                                  int *nbBoards) {
  (void)nbDevices;
  (void)deviceId;
  (void)nbBoards;
  fprintf(stderr, "acblasXtGetNumBoards is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasXtMaxBoards(int *nbGpuBoards) {
  (void)nbGpuBoards;
  fprintf(stderr, "acblasXtMaxBoards is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZaxpy_v2(acblasHandle_t handle, int n,
                                            const acDoubleComplex *alpha,
                                            const acDoubleComplex *x, int incx,
                                            acDoubleComplex *y, int incy) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZaxpy_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZaxpy_v2_64(acblasHandle_t handle, int64_t n,
                                               const acDoubleComplex *alpha,
                                               const acDoubleComplex *x,
                                               int64_t incx, acDoubleComplex *y,
                                               int64_t incy) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZaxpy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZcopy_v2(acblasHandle_t handle, int n,
                                            const acDoubleComplex *x, int incx,
                                            acDoubleComplex *y, int incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZcopy_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZcopy_v2_64(acblasHandle_t handle, int64_t n,
                                               const acDoubleComplex *x,
                                               int64_t incx, acDoubleComplex *y,
                                               int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZcopy_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZdgmm(acblasHandle_t handle,
                                         acblasSideMode_t mode, int m, int n,
                                         const acDoubleComplex *A, int lda,
                                         const acDoubleComplex *x, int incx,
                                         acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZdgmm is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZdgmm_64(acblasHandle_t handle, acblasSideMode_t mode, int64_t m,
               int64_t n, const acDoubleComplex *A, int64_t lda,
               const acDoubleComplex *x, int64_t incx, acDoubleComplex *C,
               int64_t ldc) {
  (void)handle;
  (void)mode;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZdgmm_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZdotc_v2(acblasHandle_t handle, int n,
                                            const acDoubleComplex *x, int incx,
                                            const acDoubleComplex *y, int incy,
                                            acDoubleComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasZdotc_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZdotc_v2_64(acblasHandle_t handle, int64_t n, const acDoubleComplex *x,
                  int64_t incx, const acDoubleComplex *y, int64_t incy,
                  acDoubleComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasZdotc_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZdotu_v2(acblasHandle_t handle, int n,
                                            const acDoubleComplex *x, int incx,
                                            const acDoubleComplex *y, int incy,
                                            acDoubleComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasZdotu_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZdotu_v2_64(acblasHandle_t handle, int64_t n, const acDoubleComplex *x,
                  int64_t incx, const acDoubleComplex *y, int64_t incy,
                  acDoubleComplex *result) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)result;
  fprintf(stderr, "acblasZdotu_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZdrot_v2(acblasHandle_t handle, int n,
                                            acDoubleComplex *x, int incx,
                                            acDoubleComplex *y, int incy,
                                            const double *c, const double *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasZdrot_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZdrot_v2_64(acblasHandle_t handle, int64_t n,
                                               acDoubleComplex *x, int64_t incx,
                                               acDoubleComplex *y, int64_t incy,
                                               const double *c,
                                               const double *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasZdrot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZdscal_v2(acblasHandle_t handle, int n,
                                             const double *alpha,
                                             acDoubleComplex *x, int incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZdscal_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZdscal_v2_64(acblasHandle_t handle,
                                                int64_t n, const double *alpha,
                                                acDoubleComplex *x,
                                                int64_t incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZdscal_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgbmv_v2(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
               int kl, int ku, const acDoubleComplex *alpha,
               const acDoubleComplex *A, int lda, const acDoubleComplex *x,
               int incx, const acDoubleComplex *beta, acDoubleComplex *y,
               int incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZgbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgbmv_v2_64(acblasHandle_t handle, acblasOperation_t trans, int64_t m,
                  int64_t n, int64_t kl, int64_t ku,
                  const acDoubleComplex *alpha, const acDoubleComplex *A,
                  int64_t lda, const acDoubleComplex *x, int64_t incx,
                  const acDoubleComplex *beta, acDoubleComplex *y,
                  int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)kl;
  (void)ku;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZgbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgeam(acblasHandle_t handle, acblasOperation_t transa,
            acblasOperation_t transb, int m, int n,
            const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
            const acDoubleComplex *beta, const acDoubleComplex *B, int ldb,
            acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZgeam is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgeam_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, const acDoubleComplex *alpha,
    const acDoubleComplex *A, int64_t lda, const acDoubleComplex *beta,
    const acDoubleComplex *B, int64_t ldb, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZgeam_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgelsBatched(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
                   int nrhs, acDoubleComplex *const Aarray[], int lda,
                   acDoubleComplex *const Carray[], int ldc, int *info,
                   int *devInfoArray, int batchSize) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)Aarray;
  (void)lda;
  (void)Carray;
  (void)ldc;
  (void)info;
  (void)devInfoArray;
  (void)batchSize;
  fprintf(stderr, "acblasZgelsBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgemm3m(acblasHandle_t handle, acblasOperation_t transa,
              acblasOperation_t transb, int m, int n, int k,
              const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
              const acDoubleComplex *B, int ldb, const acDoubleComplex *beta,
              acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZgemm3m is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgemm3m_64(acblasHandle_t handle, acblasOperation_t transa,
                 acblasOperation_t transb, int64_t m, int64_t n, int64_t k,
                 const acDoubleComplex *alpha, const acDoubleComplex *A,
                 int64_t lda, const acDoubleComplex *B, int64_t ldb,
                 const acDoubleComplex *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZgemm3m_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemmBatched(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int m, int n, int k, const acDoubleComplex *alpha,
    const acDoubleComplex *const Aarray[], int lda,
    const acDoubleComplex *const Barray[], int ldb, const acDoubleComplex *beta,
    acDoubleComplex *const Carray[], int ldc, int batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasZgemmBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemmBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const acDoubleComplex *alpha,
    const acDoubleComplex *const Aarray[], int64_t lda,
    const acDoubleComplex *const Barray[], int64_t ldb,
    const acDoubleComplex *beta, acDoubleComplex *const Carray[], int64_t ldc,
    int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)Barray;
  (void)ldb;
  (void)beta;
  (void)Carray;
  (void)ldc;
  (void)batchCount;
  fprintf(stderr, "acblasZgemmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemmStridedBatched(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int m, int n, int k, const acDoubleComplex *alpha, const acDoubleComplex *A,
    int lda, long long strideA, const acDoubleComplex *B, int ldb,
    long long strideB, const acDoubleComplex *beta, acDoubleComplex *C, int ldc,
    long long strideC, int batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasZgemmStridedBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemmStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const acDoubleComplex *alpha,
    const acDoubleComplex *A, int64_t lda, long long strideA,
    const acDoubleComplex *B, int64_t ldb, long long strideB,
    const acDoubleComplex *beta, acDoubleComplex *C, int64_t ldc,
    long long strideC, int64_t batchCount) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)B;
  (void)ldb;
  (void)strideB;
  (void)beta;
  (void)C;
  (void)ldc;
  (void)strideC;
  (void)batchCount;
  fprintf(stderr, "acblasZgemmStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgemm_v2(acblasHandle_t handle, acblasOperation_t transa,
               acblasOperation_t transb, int m, int n, int k,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *B, int ldb, const acDoubleComplex *beta,
               acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZgemm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemm_v2_64(
    acblasHandle_t handle, acblasOperation_t transa, acblasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const acDoubleComplex *alpha,
    const acDoubleComplex *A, int64_t lda, const acDoubleComplex *B,
    int64_t ldb, const acDoubleComplex *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZgemm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgemvBatched(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
                   const acDoubleComplex *alpha,
                   const acDoubleComplex *const Aarray[], int lda,
                   const acDoubleComplex *const xarray[], int incx,
                   const acDoubleComplex *beta, acDoubleComplex *const yarray[],
                   int incy, int batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasZgemvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemvBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const acDoubleComplex *alpha, const acDoubleComplex *const Aarray[],
    int64_t lda, const acDoubleComplex *const xarray[], int64_t incx,
    const acDoubleComplex *beta, acDoubleComplex *const yarray[], int64_t incy,
    int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)Aarray;
  (void)lda;
  (void)xarray;
  (void)incx;
  (void)beta;
  (void)yarray;
  (void)incy;
  (void)batchCount;
  fprintf(stderr, "acblasZgemvBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemvStridedBatched_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const acDoubleComplex *alpha, const acDoubleComplex *A, int64_t lda,
    long long strideA, const acDoubleComplex *x, int64_t incx,
    long long stridex, const acDoubleComplex *beta, acDoubleComplex *y,
    int64_t incy, long long stridey, int64_t batchCount) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)strideA;
  (void)x;
  (void)incx;
  (void)stridex;
  (void)beta;
  (void)y;
  (void)incy;
  (void)stridey;
  (void)batchCount;
  fprintf(stderr, "acblasZgemvStridedBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgemv_v2(acblasHandle_t handle, acblasOperation_t trans, int m, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *x, int incx, const acDoubleComplex *beta,
               acDoubleComplex *y, int incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZgemv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgemv_v2_64(
    acblasHandle_t handle, acblasOperation_t trans, int64_t m, int64_t n,
    const acDoubleComplex *alpha, const acDoubleComplex *A, int64_t lda,
    const acDoubleComplex *x, int64_t incx, const acDoubleComplex *beta,
    acDoubleComplex *y, int64_t incy) {
  (void)handle;
  (void)trans;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZgemv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgeqrfBatched(
    acblasHandle_t handle, int m, int n, acDoubleComplex *const Aarray[],
    int lda, acDoubleComplex *const TauArray[], int *info, int batchSize) {
  (void)handle;
  (void)m;
  (void)n;
  (void)Aarray;
  (void)lda;
  (void)TauArray;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasZgeqrfBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgerc_v2(acblasHandle_t handle, int m, int n,
                                            const acDoubleComplex *alpha,
                                            const acDoubleComplex *x, int incx,
                                            const acDoubleComplex *y, int incy,
                                            acDoubleComplex *A, int lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZgerc_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgerc_v2_64(acblasHandle_t handle, int64_t m, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *x,
                  int64_t incx, const acDoubleComplex *y, int64_t incy,
                  acDoubleComplex *A, int64_t lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZgerc_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgeru_v2(acblasHandle_t handle, int m, int n,
                                            const acDoubleComplex *alpha,
                                            const acDoubleComplex *x, int incx,
                                            const acDoubleComplex *y, int incy,
                                            acDoubleComplex *A, int lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZgeru_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgeru_v2_64(acblasHandle_t handle, int64_t m, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *x,
                  int64_t incx, const acDoubleComplex *y, int64_t incy,
                  acDoubleComplex *A, int64_t lda) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZgeru_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZgetrfBatched(acblasHandle_t handle, int n,
                                                 acDoubleComplex *const A[],
                                                 int lda, int *P, int *info,
                                                 int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)P;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasZgetrfBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgetriBatched(acblasHandle_t handle, int n,
                    const acDoubleComplex *const A[], int lda, const int *P,
                    acDoubleComplex *const C[], int ldc, int *info,
                    int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)P;
  (void)C;
  (void)ldc;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasZgetriBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZgetrsBatched(acblasHandle_t handle, acblasOperation_t trans, int n,
                    int nrhs, const acDoubleComplex *const Aarray[], int lda,
                    const int *devIpiv, acDoubleComplex *const Barray[],
                    int ldb, int *info, int batchSize) {
  (void)handle;
  (void)trans;
  (void)n;
  (void)nrhs;
  (void)Aarray;
  (void)lda;
  (void)devIpiv;
  (void)Barray;
  (void)ldb;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasZgetrsBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhbmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n, int k,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *x, int incx, const acDoubleComplex *beta,
               acDoubleComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZhbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZhbmv_v2_64(
    acblasHandle_t handle, acblasFillMode_t uplo, int64_t n, int64_t k,
    const acDoubleComplex *alpha, const acDoubleComplex *A, int64_t lda,
    const acDoubleComplex *x, int64_t incx, const acDoubleComplex *beta,
    acDoubleComplex *y, int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZhbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhemm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, int m, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *B, int ldb, const acDoubleComplex *beta,
               acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZhemm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZhemm_v2_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    int64_t m, int64_t n, const acDoubleComplex *alpha,
    const acDoubleComplex *A, int64_t lda, const acDoubleComplex *B,
    int64_t ldb, const acDoubleComplex *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZhemm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhemv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *x, int incx, const acDoubleComplex *beta,
               acDoubleComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZhemv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhemv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *A,
                  int64_t lda, const acDoubleComplex *x, int64_t incx,
                  const acDoubleComplex *beta, acDoubleComplex *y,
                  int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZhemv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZher2_v2(acblasHandle_t handle,
                                            acblasFillMode_t uplo, int n,
                                            const acDoubleComplex *alpha,
                                            const acDoubleComplex *x, int incx,
                                            const acDoubleComplex *y, int incy,
                                            acDoubleComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZher2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZher2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *x,
                  int64_t incx, const acDoubleComplex *y, int64_t incy,
                  acDoubleComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZher2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZher2k_v2(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k,
                const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
                const acDoubleComplex *B, int ldb, const double *beta,
                acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZher2k_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZher2k_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                   acblasOperation_t trans, int64_t n, int64_t k,
                   const acDoubleComplex *alpha, const acDoubleComplex *A,
                   int64_t lda, const acDoubleComplex *B, int64_t ldb,
                   const double *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZher2k_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZher_v2(acblasHandle_t handle,
                                           acblasFillMode_t uplo, int n,
                                           const double *alpha,
                                           const acDoubleComplex *x, int incx,
                                           acDoubleComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZher_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZher_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                 const double *alpha, const acDoubleComplex *x, int64_t incx,
                 acDoubleComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZher_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZherk_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, int n, int k, const double *alpha,
               const acDoubleComplex *A, int lda, const double *beta,
               acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZherk_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZherk_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, int64_t n, int64_t k,
                  const double *alpha, const acDoubleComplex *A, int64_t lda,
                  const double *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZherk_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZherkx(acblasHandle_t handle, acblasFillMode_t uplo,
             acblasOperation_t trans, int n, int k,
             const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
             const acDoubleComplex *B, int ldb, const double *beta,
             acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZherkx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZherkx_64(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int64_t n, int64_t k,
                const acDoubleComplex *alpha, const acDoubleComplex *A,
                int64_t lda, const acDoubleComplex *B, int64_t ldb,
                const double *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZherkx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhpmv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *AP,
               const acDoubleComplex *x, int incx, const acDoubleComplex *beta,
               acDoubleComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZhpmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhpmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *AP,
                  const acDoubleComplex *x, int64_t incx,
                  const acDoubleComplex *beta, acDoubleComplex *y,
                  int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)AP;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZhpmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhpr2_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *x, int incx,
               const acDoubleComplex *y, int incy, acDoubleComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)AP;
  fprintf(stderr, "acblasZhpr2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhpr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *x,
                  int64_t incx, const acDoubleComplex *y, int64_t incy,
                  acDoubleComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)AP;
  fprintf(stderr, "acblasZhpr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZhpr_v2(acblasHandle_t handle,
                                           acblasFillMode_t uplo, int n,
                                           const double *alpha,
                                           const acDoubleComplex *x, int incx,
                                           acDoubleComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)AP;
  fprintf(stderr, "acblasZhpr_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZhpr_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                 const double *alpha, const acDoubleComplex *x, int64_t incx,
                 acDoubleComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)AP;
  fprintf(stderr, "acblasZhpr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZmatinvBatched(
    acblasHandle_t handle, int n, const acDoubleComplex *const A[], int lda,
    acDoubleComplex *const Ainv[], int lda_inv, int *info, int batchSize) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)Ainv;
  (void)lda_inv;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acblasZmatinvBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZrot_v2(acblasHandle_t handle, int n,
                                           acDoubleComplex *x, int incx,
                                           acDoubleComplex *y, int incy,
                                           const double *c,
                                           const acDoubleComplex *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasZrot_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZrot_v2_64(acblasHandle_t handle, int64_t n,
                                              acDoubleComplex *x, int64_t incx,
                                              acDoubleComplex *y, int64_t incy,
                                              const double *c,
                                              const acDoubleComplex *s) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasZrot_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZrotg_v2(acblasHandle_t handle,
                                            acDoubleComplex *a,
                                            acDoubleComplex *b, double *c,
                                            acDoubleComplex *s) {
  (void)handle;
  (void)a;
  (void)b;
  (void)c;
  (void)s;
  fprintf(stderr, "acblasZrotg_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZscal_v2(acblasHandle_t handle, int n,
                                            const acDoubleComplex *alpha,
                                            acDoubleComplex *x, int incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZscal_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZscal_v2_64(acblasHandle_t handle, int64_t n,
                                               const acDoubleComplex *alpha,
                                               acDoubleComplex *x,
                                               int64_t incx) {
  (void)handle;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZscal_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZswap_v2(acblasHandle_t handle, int n,
                                            acDoubleComplex *x, int incx,
                                            acDoubleComplex *y, int incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZswap_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZswap_v2_64(acblasHandle_t handle, int64_t n,
                                               acDoubleComplex *x, int64_t incx,
                                               acDoubleComplex *y,
                                               int64_t incy) {
  (void)handle;
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZswap_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsymm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, int m, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *B, int ldb, const acDoubleComplex *beta,
               acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsymm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZsymm_v2_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    int64_t m, int64_t n, const acDoubleComplex *alpha,
    const acDoubleComplex *A, int64_t lda, const acDoubleComplex *B,
    int64_t ldb, const acDoubleComplex *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsymm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsymv_v2(acblasHandle_t handle, acblasFillMode_t uplo, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *x, int incx, const acDoubleComplex *beta,
               acDoubleComplex *y, int incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZsymv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsymv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *A,
                  int64_t lda, const acDoubleComplex *x, int64_t incx,
                  const acDoubleComplex *beta, acDoubleComplex *y,
                  int64_t incy) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  fprintf(stderr, "acblasZsymv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZsyr2_v2(acblasHandle_t handle,
                                            acblasFillMode_t uplo, int n,
                                            const acDoubleComplex *alpha,
                                            const acDoubleComplex *x, int incx,
                                            const acDoubleComplex *y, int incy,
                                            acDoubleComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZsyr2_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsyr2_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *x,
                  int64_t incx, const acDoubleComplex *y, int64_t incy,
                  acDoubleComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZsyr2_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsyr2k_v2(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int n, int k,
                const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
                const acDoubleComplex *B, int ldb, const acDoubleComplex *beta,
                acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsyr2k_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZsyr2k_v2_64(
    acblasHandle_t handle, acblasFillMode_t uplo, acblasOperation_t trans,
    int64_t n, int64_t k, const acDoubleComplex *alpha,
    const acDoubleComplex *A, int64_t lda, const acDoubleComplex *B,
    int64_t ldb, const acDoubleComplex *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsyr2k_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZsyr_v2(acblasHandle_t handle,
                                           acblasFillMode_t uplo, int n,
                                           const acDoubleComplex *alpha,
                                           const acDoubleComplex *x, int incx,
                                           acDoubleComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZsyr_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsyr_v2_64(acblasHandle_t handle, acblasFillMode_t uplo, int64_t n,
                 const acDoubleComplex *alpha, const acDoubleComplex *x,
                 int64_t incx, acDoubleComplex *A, int64_t lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZsyr_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsyrk_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, int n, int k,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *beta, acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsyrk_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsyrk_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, int64_t n, int64_t k,
                  const acDoubleComplex *alpha, const acDoubleComplex *A,
                  int64_t lda, const acDoubleComplex *beta, acDoubleComplex *C,
                  int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsyrk_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsyrkx(acblasHandle_t handle, acblasFillMode_t uplo,
             acblasOperation_t trans, int n, int k,
             const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
             const acDoubleComplex *B, int ldb, const acDoubleComplex *beta,
             acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsyrkx is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZsyrkx_64(acblasHandle_t handle, acblasFillMode_t uplo,
                acblasOperation_t trans, int64_t n, int64_t k,
                const acDoubleComplex *alpha, const acDoubleComplex *A,
                int64_t lda, const acDoubleComplex *B, int64_t ldb,
                const acDoubleComplex *beta, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZsyrkx_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtbmv_v2(acblasHandle_t handle,
                                            acblasFillMode_t uplo,
                                            acblasOperation_t trans,
                                            acblasDiagType_t diag, int n, int k,
                                            const acDoubleComplex *A, int lda,
                                            acDoubleComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtbmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtbmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const acDoubleComplex *A, int64_t lda,
                  acDoubleComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtbmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtbsv_v2(acblasHandle_t handle,
                                            acblasFillMode_t uplo,
                                            acblasOperation_t trans,
                                            acblasDiagType_t diag, int n, int k,
                                            const acDoubleComplex *A, int lda,
                                            acDoubleComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtbsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtbsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  int64_t k, const acDoubleComplex *A, int64_t lda,
                  acDoubleComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtbsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtpmv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const acDoubleComplex *AP, acDoubleComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtpmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtpmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acDoubleComplex *AP, acDoubleComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtpmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtpsv_v2(acblasHandle_t handle, acblasFillMode_t uplo,
               acblasOperation_t trans, acblasDiagType_t diag, int n,
               const acDoubleComplex *AP, acDoubleComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtpsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtpsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acDoubleComplex *AP, acDoubleComplex *x, int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)AP;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtpsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtpttr(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const acDoubleComplex *AP,
                                          acDoubleComplex *A, int lda) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)AP;
  (void)A;
  (void)lda;
  fprintf(stderr, "acblasZtpttr is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtrmm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, acblasOperation_t trans,
               acblasDiagType_t diag, int m, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               const acDoubleComplex *B, int ldb, acDoubleComplex *C, int ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZtrmm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtrmm_v2_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, acblasDiagType_t diag, int64_t m, int64_t n,
    const acDoubleComplex *alpha, const acDoubleComplex *A, int64_t lda,
    const acDoubleComplex *B, int64_t ldb, acDoubleComplex *C, int64_t ldc) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acblasZtrmm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtrmv_v2(acblasHandle_t handle,
                                            acblasFillMode_t uplo,
                                            acblasOperation_t trans,
                                            acblasDiagType_t diag, int n,
                                            const acDoubleComplex *A, int lda,
                                            acDoubleComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtrmv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtrmv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acDoubleComplex *A, int64_t lda, acDoubleComplex *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtrmv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtrsmBatched(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, acblasDiagType_t diag, int m, int n,
    const acDoubleComplex *alpha, const acDoubleComplex *const A[], int lda,
    acDoubleComplex *const B[], int ldb, int batchCount) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)batchCount;
  fprintf(stderr, "acblasZtrsmBatched is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtrsmBatched_64(
    acblasHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, acblasDiagType_t diag, int64_t m, int64_t n,
    const acDoubleComplex *alpha, const acDoubleComplex *const A[], int64_t lda,
    acDoubleComplex *const B[], int64_t ldb, int64_t batchCount) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)batchCount;
  fprintf(stderr, "acblasZtrsmBatched_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtrsm_v2(acblasHandle_t handle, acblasSideMode_t side,
               acblasFillMode_t uplo, acblasOperation_t trans,
               acblasDiagType_t diag, int m, int n,
               const acDoubleComplex *alpha, const acDoubleComplex *A, int lda,
               acDoubleComplex *B, int ldb) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasZtrsm_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtrsm_v2_64(acblasHandle_t handle, acblasSideMode_t side,
                  acblasFillMode_t uplo, acblasOperation_t trans,
                  acblasDiagType_t diag, int64_t m, int64_t n,
                  const acDoubleComplex *alpha, const acDoubleComplex *A,
                  int64_t lda, acDoubleComplex *B, int64_t ldb) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  fprintf(stderr, "acblasZtrsm_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtrsv_v2(acblasHandle_t handle,
                                            acblasFillMode_t uplo,
                                            acblasOperation_t trans,
                                            acblasDiagType_t diag, int n,
                                            const acDoubleComplex *A, int lda,
                                            acDoubleComplex *x, int incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtrsv_v2 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasZtrsv_v2_64(acblasHandle_t handle, acblasFillMode_t uplo,
                  acblasOperation_t trans, acblasDiagType_t diag, int64_t n,
                  const acDoubleComplex *A, int64_t lda, acDoubleComplex *x,
                  int64_t incx) {
  (void)handle;
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)A;
  (void)lda;
  (void)x;
  (void)incx;
  fprintf(stderr, "acblasZtrsv_v2_64 is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasZtrttp(acblasHandle_t handle,
                                          acblasFillMode_t uplo, int n,
                                          const acDoubleComplex *A, int lda,
                                          acDoubleComplex *AP) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)AP;
  fprintf(stderr, "acblasZtrttp is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

/* ── cublasLt: 11 unsupported APIs ── */

static inline unsigned int
acblasLtDisableCpuInstructionsSetMask(unsigned int mask) {
  (void)mask;
  fprintf(stderr, "acblasLtDisableCpuInstructionsSetMask is not supported.\n");
  exit(1);
  return (unsigned int)0; /* unreachable */
}

static inline acblasStatus_t acblasLtGetProperty(hggcLibraryPropertyType type,
                                                 int *value) {
  (void)type;
  (void)value;
  fprintf(stderr, "acblasLtGetProperty is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline const char *acblasLtGetStatusName(acblasStatus_t status) {
  (void)status;
  fprintf(stderr, "acblasLtGetStatusName is not supported.\n");
  exit(1);
  return (const char *)0; /* unreachable */
}

static inline const char *acblasLtGetStatusString(acblasStatus_t status) {
  (void)status;
  fprintf(stderr, "acblasLtGetStatusString is not supported.\n");
  exit(1);
  return (const char *)0; /* unreachable */
}

static inline acblasStatus_t
acblasLtHeuristicsCacheGetCapacity(size_t *capacity) {
  (void)capacity;
  fprintf(stderr, "acblasLtHeuristicsCacheGetCapacity is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t
acblasLtHeuristicsCacheSetCapacity(size_t capacity) {
  (void)capacity;
  fprintf(stderr, "acblasLtHeuristicsCacheSetCapacity is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasLtLoggerForceDisable(void) {
  fprintf(stderr, "acblasLtLoggerForceDisable is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasLtLoggerOpenFile(const char *logFile) {
  (void)logFile;
  fprintf(stderr, "acblasLtLoggerOpenFile is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasLtLoggerSetFile(FILE *file) {
  (void)file;
  fprintf(stderr, "acblasLtLoggerSetFile is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasLtLoggerSetLevel(int level) {
  (void)level;
  fprintf(stderr, "acblasLtLoggerSetLevel is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

static inline acblasStatus_t acblasLtLoggerSetMask(int mask) {
  (void)mask;
  fprintf(stderr, "acblasLtLoggerSetMask is not supported.\n");
  exit(1);
  return (acblasStatus_t)0; /* unreachable */
}

/* ── cuda: 10 unsupported APIs ── */

static inline HGresult hgGLCtxCreate(HGcontext *pCtx, unsigned int Flags,
                                     HGdevice device) {
  (void)pCtx;
  (void)Flags;
  (void)device;
  fprintf(stderr, "hgGLCtxCreate is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult hgGLCtxCreate_v2(HGcontext *pCtx, unsigned int Flags,
                                        HGdevice device) {
  (void)pCtx;
  (void)Flags;
  (void)device;
  fprintf(stderr, "hgGLCtxCreate_v2 is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult hgGLInit(void) {
  fprintf(stderr, "hgGLInit is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult
hgGraphicsMapResources_ptsz(unsigned int count, HGgraphicsResource *resources,
                            HGstream hStream) {
  (void)count;
  (void)resources;
  (void)hStream;
  fprintf(stderr, "hgGraphicsMapResources_ptsz is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult
hgGraphicsResourceGetMappedMipmappedArray(HGmipmappedArray *pMipmappedArray,
                                          HGgraphicsResource resource) {
  (void)pMipmappedArray;
  (void)resource;
  fprintf(stderr,
          "hgGraphicsResourceGetMappedMipmappedArray is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult
hgGraphicsResourceGetMappedPointer_v2(HGdeviceptr *pDevPtr, size_t *pSize,
                                      HGgraphicsResource resource) {
  (void)pDevPtr;
  (void)pSize;
  (void)resource;
  fprintf(stderr, "hgGraphicsResourceGetMappedPointer_v2 is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult
hgGraphicsResourceSetMapFlags_v2(HGgraphicsResource resource,
                                 unsigned int flags) {
  (void)resource;
  (void)flags;
  fprintf(stderr, "hgGraphicsResourceSetMapFlags_v2 is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult hgGraphicsSubResourceGetMappedArray(
    HGarray *pArray, HGgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  (void)pArray;
  (void)resource;
  (void)arrayIndex;
  (void)mipLevel;
  fprintf(stderr, "hgGraphicsSubResourceGetMappedArray is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult
hgGraphicsUnmapResources_ptsz(unsigned int count, HGgraphicsResource *resources,
                              HGstream hStream) {
  (void)count;
  (void)resources;
  (void)hStream;
  fprintf(stderr, "hgGraphicsUnmapResources_ptsz is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

static inline HGresult
hgGraphicsUnregisterResource(HGgraphicsResource resource) {
  (void)resource;
  fprintf(stderr, "hgGraphicsUnregisterResource is not supported.\n");
  exit(1);
  return (HGresult)0; /* unreachable */
}

/* ── cudart: 8 unsupported APIs ── */

static inline hggcError_t hggcGLSetGLDevice(int device) {
  (void)device;
  fprintf(stderr, "hggcGLSetGLDevice is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

static inline hggcError_t
hggcGraphicsMapResources(int count, hggcGraphicsResource_t *resources,
                         hggcStream_t stream) {
  (void)count;
  (void)resources;
  (void)stream;
  fprintf(stderr, "hggcGraphicsMapResources is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

static inline hggcError_t hggcGraphicsResourceGetMappedMipmappedArray(
    hggcMipmappedArray_t *mipmappedArray, hggcGraphicsResource_t resource) {
  (void)mipmappedArray;
  (void)resource;
  fprintf(stderr,
          "hggcGraphicsResourceGetMappedMipmappedArray is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

static inline hggcError_t
hggcGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
                                     hggcGraphicsResource_t resource) {
  (void)devPtr;
  (void)size;
  (void)resource;
  fprintf(stderr, "hggcGraphicsResourceGetMappedPointer is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

static inline hggcError_t
hggcGraphicsResourceSetMapFlags(hggcGraphicsResource_t resource,
                                unsigned int flags) {
  (void)resource;
  (void)flags;
  fprintf(stderr, "hggcGraphicsResourceSetMapFlags is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

static inline hggcError_t hggcGraphicsSubResourceGetMappedArray(
    hggcArray_t *array, hggcGraphicsResource_t resource,
    unsigned int arrayIndex, unsigned int mipLevel) {
  (void)array;
  (void)resource;
  (void)arrayIndex;
  (void)mipLevel;
  fprintf(stderr, "hggcGraphicsSubResourceGetMappedArray is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

static inline hggcError_t
hggcGraphicsUnmapResources(int count, hggcGraphicsResource_t *resources,
                           hggcStream_t stream) {
  (void)count;
  (void)resources;
  (void)stream;
  fprintf(stderr, "hggcGraphicsUnmapResources is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

static inline hggcError_t
hggcGraphicsUnregisterResource(hggcGraphicsResource_t resource) {
  (void)resource;
  fprintf(stderr, "hggcGraphicsUnregisterResource is not supported.\n");
  exit(1);
  return (hggcError_t)0; /* unreachable */
}

/* ── cudnn: 42 unsupported APIs ── */

static inline acdnnStatus_t acdnnBuildRNNDynamic(acdnnHandle_t handle,
                                                 acdnnRNNDescriptor_t rnnDesc,
                                                 int miniBatch) {
  (void)handle;
  (void)rnnDesc;
  (void)miniBatch;
  fprintf(stderr, "acdnnBuildRNNDynamic is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnConvolutionBackwardBias(acdnnHandle_t handle, const void *alpha,
                             const acdnnTensorDescriptor_t dyDesc,
                             const void *dy, const void *beta,
                             const acdnnTensorDescriptor_t dbDesc, void *db) {
  (void)handle;
  (void)alpha;
  (void)dyDesc;
  (void)dy;
  (void)beta;
  (void)dbDesc;
  (void)db;
  fprintf(stderr, "acdnnConvolutionBackwardBias is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnCopyAlgorithmDescriptor(const acdnnAlgorithmDescriptor_t src,
                             acdnnAlgorithmDescriptor_t dest) {
  (void)src;
  (void)dest;
  fprintf(stderr, "acdnnCopyAlgorithmDescriptor is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnCreateAlgorithmDescriptor(acdnnAlgorithmDescriptor_t *algoDesc) {
  (void)algoDesc;
  fprintf(stderr, "acdnnCreateAlgorithmDescriptor is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnCreateAlgorithmPerformance(acdnnAlgorithmPerformance_t *algoPerf,
                                int numberToCreate) {
  (void)algoPerf;
  (void)numberToCreate;
  fprintf(stderr, "acdnnCreateAlgorithmPerformance is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnCreatePersistentRNNPlan(acdnnRNNDescriptor_t rnnDesc, const int minibatch,
                             const acdnnDataType_t dataType,
                             acdnnPersistentRNNPlan_t *plan) {
  (void)rnnDesc;
  (void)minibatch;
  (void)dataType;
  (void)plan;
  fprintf(stderr, "acdnnCreatePersistentRNNPlan is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnDestroyAlgorithmDescriptor(acdnnAlgorithmDescriptor_t algoDesc) {
  (void)algoDesc;
  fprintf(stderr, "acdnnDestroyAlgorithmDescriptor is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnDestroyAlgorithmPerformance(acdnnAlgorithmPerformance_t *algoPerf,
                                 int numberToDestroy) {
  (void)algoPerf;
  (void)numberToDestroy;
  fprintf(stderr, "acdnnDestroyAlgorithmPerformance is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnDestroyPersistentRNNPlan(acdnnPersistentRNNPlan_t plan) {
  (void)plan;
  fprintf(stderr, "acdnnDestroyPersistentRNNPlan is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnFindRNNBackwardDataAlgorithmEx(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const acdnnTensorDescriptor_t *yDesc, const void *y,
    const acdnnTensorDescriptor_t *dyDesc, const void *dy,
    const acdnnTensorDescriptor_t dhyDesc, const void *dhy,
    const acdnnTensorDescriptor_t dcyDesc, const void *dcy,
    const acdnnFilterDescriptor_t wDesc, const void *w,
    const acdnnTensorDescriptor_t hxDesc, const void *hx,
    const acdnnTensorDescriptor_t cxDesc, const void *cx,
    const acdnnTensorDescriptor_t *dxDesc, void *dx,
    const acdnnTensorDescriptor_t dhxDesc, void *dhx,
    const acdnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    acdnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)seqLength;
  (void)yDesc;
  (void)y;
  (void)dyDesc;
  (void)dy;
  (void)dhyDesc;
  (void)dhy;
  (void)dcyDesc;
  (void)dcy;
  (void)wDesc;
  (void)w;
  (void)hxDesc;
  (void)hx;
  (void)cxDesc;
  (void)cx;
  (void)dxDesc;
  (void)dx;
  (void)dhxDesc;
  (void)dhx;
  (void)dcxDesc;
  (void)dcx;
  (void)findIntensity;
  (void)requestedAlgoCount;
  (void)returnedAlgoCount;
  (void)perfResults;
  (void)workspace;
  (void)workSpaceSizeInBytes;
  (void)reserveSpace;
  (void)reserveSpaceSizeInBytes;
  fprintf(stderr, "acdnnFindRNNBackwardDataAlgorithmEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnFindRNNBackwardWeightsAlgorithmEx(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const acdnnTensorDescriptor_t *xDesc, const void *x,
    const acdnnTensorDescriptor_t hxDesc, const void *hx,
    const acdnnTensorDescriptor_t *yDesc, const void *y,
    const float findIntensity, const int requestedAlgoCount,
    int *returnedAlgoCount, acdnnAlgorithmPerformance_t *perfResults,
    const void *workspace, size_t workSpaceSizeInBytes,
    const acdnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)seqLength;
  (void)xDesc;
  (void)x;
  (void)hxDesc;
  (void)hx;
  (void)yDesc;
  (void)y;
  (void)findIntensity;
  (void)requestedAlgoCount;
  (void)returnedAlgoCount;
  (void)perfResults;
  (void)workspace;
  (void)workSpaceSizeInBytes;
  (void)dwDesc;
  (void)dw;
  (void)reserveSpace;
  (void)reserveSpaceSizeInBytes;
  fprintf(stderr, "acdnnFindRNNBackwardWeightsAlgorithmEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnFindRNNForwardInferenceAlgorithmEx(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const acdnnTensorDescriptor_t *xDesc, const void *x,
    const acdnnTensorDescriptor_t hxDesc, const void *hx,
    const acdnnTensorDescriptor_t cxDesc, const void *cx,
    const acdnnFilterDescriptor_t wDesc, const void *w,
    const acdnnTensorDescriptor_t *yDesc, void *y,
    const acdnnTensorDescriptor_t hyDesc, void *hy,
    const acdnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    acdnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)seqLength;
  (void)xDesc;
  (void)x;
  (void)hxDesc;
  (void)hx;
  (void)cxDesc;
  (void)cx;
  (void)wDesc;
  (void)w;
  (void)yDesc;
  (void)y;
  (void)hyDesc;
  (void)hy;
  (void)cyDesc;
  (void)cy;
  (void)findIntensity;
  (void)requestedAlgoCount;
  (void)returnedAlgoCount;
  (void)perfResults;
  (void)workspace;
  (void)workSpaceSizeInBytes;
  fprintf(stderr,
          "acdnnFindRNNForwardInferenceAlgorithmEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnFindRNNForwardTrainingAlgorithmEx(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const acdnnTensorDescriptor_t *xDesc, const void *x,
    const acdnnTensorDescriptor_t hxDesc, const void *hx,
    const acdnnTensorDescriptor_t cxDesc, const void *cx,
    const acdnnFilterDescriptor_t wDesc, const void *w,
    const acdnnTensorDescriptor_t *yDesc, void *y,
    const acdnnTensorDescriptor_t hyDesc, void *hy,
    const acdnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    acdnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)seqLength;
  (void)xDesc;
  (void)x;
  (void)hxDesc;
  (void)hx;
  (void)cxDesc;
  (void)cx;
  (void)wDesc;
  (void)w;
  (void)yDesc;
  (void)y;
  (void)hyDesc;
  (void)hy;
  (void)cyDesc;
  (void)cy;
  (void)findIntensity;
  (void)requestedAlgoCount;
  (void)returnedAlgoCount;
  (void)perfResults;
  (void)workspace;
  (void)workSpaceSizeInBytes;
  (void)reserveSpace;
  (void)reserveSpaceSizeInBytes;
  fprintf(stderr, "acdnnFindRNNForwardTrainingAlgorithmEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetActivationDescriptorSwishBeta(
    acdnnActivationDescriptor_t activationDesc, double *swish_beta) {
  (void)activationDesc;
  (void)swish_beta;
  fprintf(stderr, "acdnnGetActivationDescriptorSwishBeta is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnGetAlgorithmPerformance(const acdnnAlgorithmPerformance_t algoPerf,
                             acdnnAlgorithmDescriptor_t *algoDesc,
                             acdnnStatus_t *status, float *time,
                             size_t *memory) {
  (void)algoPerf;
  (void)algoDesc;
  (void)status;
  (void)time;
  (void)memory;
  fprintf(stderr, "acdnnGetAlgorithmPerformance is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnGetAlgorithmSpaceSize(acdnnHandle_t handle,
                           acdnnAlgorithmDescriptor_t algoDesc,
                           size_t *algoSpaceSizeInBytes) {
  (void)handle;
  (void)algoDesc;
  (void)algoSpaceSizeInBytes;
  fprintf(stderr, "acdnnGetAlgorithmSpaceSize is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnGetConvolutionReorderType(acdnnConvolutionDescriptor_t convDesc,
                               acdnnReorderType_t *reorderType) {
  (void)convDesc;
  (void)reorderType;
  fprintf(stderr, "acdnnGetConvolutionReorderType is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline size_t acdnnGetMaxDeviceVersion(void) {
  fprintf(stderr, "acdnnGetMaxDeviceVersion is not supported.\n");
  exit(1);
  return (size_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetPooling2dDescriptor(
    const acdnnPoolingDescriptor_t poolingDesc, acdnnPoolingMode_t *mode,
    acdnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride) {
  (void)poolingDesc;
  (void)mode;
  (void)maxpoolingNanOpt;
  (void)windowHeight;
  (void)windowWidth;
  (void)verticalPadding;
  (void)horizontalPadding;
  (void)verticalStride;
  (void)horizontalStride;
  fprintf(stderr, "acdnnGetPooling2dDescriptor is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetPoolingNdDescriptor(
    const acdnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    acdnnPoolingMode_t *mode, acdnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[], int paddingA[], int strideA[]) {
  (void)poolingDesc;
  (void)nbDimsRequested;
  (void)mode;
  (void)maxpoolingNanOpt;
  (void)nbDims;
  (void)windowDimA;
  (void)paddingA;
  (void)strideA;
  fprintf(stderr, "acdnnGetPoolingNdDescriptor is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetRNNBackwardDataAlgorithmMaxCount(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc, int *count) {
  (void)handle;
  (void)rnnDesc;
  (void)count;
  fprintf(stderr,
          "acdnnGetRNNBackwardDataAlgorithmMaxCount is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetRNNBackwardWeightsAlgorithmMaxCount(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc, int *count) {
  (void)handle;
  (void)rnnDesc;
  (void)count;
  fprintf(stderr,
          "acdnnGetRNNBackwardWeightsAlgorithmMaxCount is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetRNNForwardInferenceAlgorithmMaxCount(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc, int *count) {
  (void)handle;
  (void)rnnDesc;
  (void)count;
  fprintf(stderr,
          "acdnnGetRNNForwardInferenceAlgorithmMaxCount is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetRNNForwardTrainingAlgorithmMaxCount(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc, int *count) {
  (void)handle;
  (void)rnnDesc;
  (void)count;
  fprintf(stderr,
          "acdnnGetRNNForwardTrainingAlgorithmMaxCount is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnGetRNNPaddingMode(acdnnRNNDescriptor_t rnnDesc,
                                                   unsigned int *paddingMode) {
  (void)rnnDesc;
  (void)paddingMode;
  fprintf(stderr, "acdnnGetRNNPaddingMode is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnIm2Col(acdnnHandle_t handle, const acdnnTensorDescriptor_t xDesc,
            const void *x, const acdnnFilterDescriptor_t wDesc,
            const acdnnConvolutionDescriptor_t convDesc, void *colBuffer) {
  (void)handle;
  (void)xDesc;
  (void)x;
  (void)wDesc;
  (void)convDesc;
  (void)colBuffer;
  fprintf(stderr, "acdnnIm2Col is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnMultiHeadAttnBackwardData(
    acdnnHandle_t handle, const acdnnAttnDescriptor_t attnDesc,
    const int loWinIdx[], const int hiWinIdx[], const int devSeqLengthsDQDO[],
    const int devSeqLengthsDKDV[], const acdnnSeqDataDescriptor_t doDesc,
    const void *dout, const acdnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const acdnnSeqDataDescriptor_t dkDesc, void *dkeys,
    const void *keys, const acdnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  (void)handle;
  (void)attnDesc;
  (void)loWinIdx;
  (void)hiWinIdx;
  (void)devSeqLengthsDQDO;
  (void)devSeqLengthsDKDV;
  (void)doDesc;
  (void)dout;
  (void)dqDesc;
  (void)dqueries;
  (void)queries;
  (void)dkDesc;
  (void)dkeys;
  (void)keys;
  (void)dvDesc;
  (void)dvalues;
  (void)values;
  (void)weightSizeInBytes;
  (void)weights;
  (void)workSpaceSizeInBytes;
  (void)workSpace;
  (void)reserveSpaceSizeInBytes;
  (void)reserveSpace;
  fprintf(stderr, "acdnnMultiHeadAttnBackwardData is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnMultiHeadAttnBackwardWeights(
    acdnnHandle_t handle, const acdnnAttnDescriptor_t attnDesc,
    acdnnWgradMode_t addGrad, const acdnnSeqDataDescriptor_t qDesc,
    const void *queries, const acdnnSeqDataDescriptor_t kDesc, const void *keys,
    const acdnnSeqDataDescriptor_t vDesc, const void *values,
    const acdnnSeqDataDescriptor_t doDesc, const void *dout,
    size_t weightSizeInBytes, const void *weights, void *dweights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  (void)handle;
  (void)attnDesc;
  (void)addGrad;
  (void)qDesc;
  (void)queries;
  (void)kDesc;
  (void)keys;
  (void)vDesc;
  (void)values;
  (void)doDesc;
  (void)dout;
  (void)weightSizeInBytes;
  (void)weights;
  (void)dweights;
  (void)workSpaceSizeInBytes;
  (void)workSpace;
  (void)reserveSpaceSizeInBytes;
  (void)reserveSpace;
  fprintf(stderr, "acdnnMultiHeadAttnBackwardWeights is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnRNNBackwardDataEx(acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
                       const acdnnRNNDataDescriptor_t yDesc, const void *y,
                       const acdnnRNNDataDescriptor_t dyDesc, const void *dy,
                       const acdnnRNNDataDescriptor_t dcDesc,
                       const void *dcAttn,
                       const acdnnTensorDescriptor_t dhyDesc, const void *dhy,
                       const acdnnTensorDescriptor_t dcyDesc, const void *dcy,
                       const acdnnFilterDescriptor_t wDesc, const void *w,
                       const acdnnTensorDescriptor_t hxDesc, const void *hx,
                       const acdnnTensorDescriptor_t cxDesc, const void *cx,
                       const acdnnRNNDataDescriptor_t dxDesc, void *dx,
                       const acdnnTensorDescriptor_t dhxDesc, void *dhx,
                       const acdnnTensorDescriptor_t dcxDesc, void *dcx,
                       const acdnnRNNDataDescriptor_t dkDesc, void *dkeys,
                       void *workSpace, size_t workSpaceSizeInBytes,
                       void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)yDesc;
  (void)y;
  (void)dyDesc;
  (void)dy;
  (void)dcDesc;
  (void)dcAttn;
  (void)dhyDesc;
  (void)dhy;
  (void)dcyDesc;
  (void)dcy;
  (void)wDesc;
  (void)w;
  (void)hxDesc;
  (void)hx;
  (void)cxDesc;
  (void)cx;
  (void)dxDesc;
  (void)dx;
  (void)dhxDesc;
  (void)dhx;
  (void)dcxDesc;
  (void)dcx;
  (void)dkDesc;
  (void)dkeys;
  (void)workSpace;
  (void)workSpaceSizeInBytes;
  (void)reserveSpace;
  (void)reserveSpaceSizeInBytes;
  fprintf(stderr, "acdnnRNNBackwardDataEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnRNNBackwardWeightsEx(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
    const acdnnRNNDataDescriptor_t xDesc, const void *x,
    const acdnnTensorDescriptor_t hxDesc, const void *hx,
    const acdnnRNNDataDescriptor_t yDesc, const void *y, void *workSpace,
    size_t workSpaceSizeInBytes, const acdnnFilterDescriptor_t dwDesc, void *dw,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)xDesc;
  (void)x;
  (void)hxDesc;
  (void)hx;
  (void)yDesc;
  (void)y;
  (void)workSpace;
  (void)workSpaceSizeInBytes;
  (void)dwDesc;
  (void)dw;
  (void)reserveSpace;
  (void)reserveSpaceSizeInBytes;
  fprintf(stderr, "acdnnRNNBackwardWeightsEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnRNNForwardInferenceEx(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
    const acdnnRNNDataDescriptor_t xDesc, const void *x,
    const acdnnTensorDescriptor_t hxDesc, const void *hx,
    const acdnnTensorDescriptor_t cxDesc, const void *cx,
    const acdnnFilterDescriptor_t wDesc, const void *w,
    const acdnnRNNDataDescriptor_t yDesc, void *y,
    const acdnnTensorDescriptor_t hyDesc, void *hy,
    const acdnnTensorDescriptor_t cyDesc, void *cy,
    const acdnnRNNDataDescriptor_t kDesc, const void *keys,
    const acdnnRNNDataDescriptor_t cDesc, void *cAttn,
    const acdnnRNNDataDescriptor_t iDesc, void *iAttn,
    const acdnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace,
    size_t workSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)xDesc;
  (void)x;
  (void)hxDesc;
  (void)hx;
  (void)cxDesc;
  (void)cx;
  (void)wDesc;
  (void)w;
  (void)yDesc;
  (void)y;
  (void)hyDesc;
  (void)hy;
  (void)cyDesc;
  (void)cy;
  (void)kDesc;
  (void)keys;
  (void)cDesc;
  (void)cAttn;
  (void)iDesc;
  (void)iAttn;
  (void)qDesc;
  (void)queries;
  (void)workSpace;
  (void)workSpaceSizeInBytes;
  fprintf(stderr, "acdnnRNNForwardInferenceEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnRNNForwardTrainingEx(
    acdnnHandle_t handle, const acdnnRNNDescriptor_t rnnDesc,
    const acdnnRNNDataDescriptor_t xDesc, const void *x,
    const acdnnTensorDescriptor_t hxDesc, const void *hx,
    const acdnnTensorDescriptor_t cxDesc, const void *cx,
    const acdnnFilterDescriptor_t wDesc, const void *w,
    const acdnnRNNDataDescriptor_t yDesc, void *y,
    const acdnnTensorDescriptor_t hyDesc, void *hy,
    const acdnnTensorDescriptor_t cyDesc, void *cy,
    const acdnnRNNDataDescriptor_t kDesc, const void *keys,
    const acdnnRNNDataDescriptor_t cDesc, void *cAttn,
    const acdnnRNNDataDescriptor_t iDesc, void *iAttn,
    const acdnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  (void)handle;
  (void)rnnDesc;
  (void)xDesc;
  (void)x;
  (void)hxDesc;
  (void)hx;
  (void)cxDesc;
  (void)cx;
  (void)wDesc;
  (void)w;
  (void)yDesc;
  (void)y;
  (void)hyDesc;
  (void)hy;
  (void)cyDesc;
  (void)cy;
  (void)kDesc;
  (void)keys;
  (void)cDesc;
  (void)cAttn;
  (void)iDesc;
  (void)iAttn;
  (void)qDesc;
  (void)queries;
  (void)workSpace;
  (void)workSpaceSizeInBytes;
  (void)reserveSpace;
  (void)reserveSpaceSizeInBytes;
  fprintf(stderr, "acdnnRNNForwardTrainingEx is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnReorderFilterAndBias(
    acdnnHandle_t handle, const acdnnFilterDescriptor_t filterDesc,
    acdnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias, const void *biasData,
    void *reorderedBiasData) {
  (void)handle;
  (void)filterDesc;
  (void)reorderType;
  (void)filterData;
  (void)reorderedFilterData;
  (void)reorderBias;
  (void)biasData;
  (void)reorderedBiasData;
  fprintf(stderr, "acdnnReorderFilterAndBias is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnRestoreAlgorithm(acdnnHandle_t handle, void *algoSpace,
                      size_t algoSpaceSizeInBytes,
                      acdnnAlgorithmDescriptor_t algoDesc) {
  (void)handle;
  (void)algoSpace;
  (void)algoSpaceSizeInBytes;
  (void)algoDesc;
  fprintf(stderr, "acdnnRestoreAlgorithm is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnSaveAlgorithm(acdnnHandle_t handle, acdnnAlgorithmDescriptor_t algoDesc,
                   void *algoSpace, size_t algoSpaceSizeInBytes) {
  (void)handle;
  (void)algoDesc;
  (void)algoSpace;
  (void)algoSpaceSizeInBytes;
  fprintf(stderr, "acdnnSaveAlgorithm is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnSetActivationDescriptorSwishBeta(
    acdnnActivationDescriptor_t activationDesc, double swish_beta) {
  (void)activationDesc;
  (void)swish_beta;
  fprintf(stderr, "acdnnSetActivationDescriptorSwishBeta is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnSetAlgorithmPerformance(acdnnAlgorithmPerformance_t algoPerf,
                             acdnnAlgorithmDescriptor_t algoDesc,
                             acdnnStatus_t status, float time, size_t memory) {
  (void)algoPerf;
  (void)algoDesc;
  (void)status;
  (void)time;
  (void)memory;
  fprintf(stderr, "acdnnSetAlgorithmPerformance is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnSetConvolutionReorderType(acdnnConvolutionDescriptor_t convDesc,
                               acdnnReorderType_t reorderType) {
  (void)convDesc;
  (void)reorderType;
  fprintf(stderr, "acdnnSetConvolutionReorderType is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnSetPersistentRNNPlan(acdnnRNNDescriptor_t rnnDesc,
                          acdnnPersistentRNNPlan_t plan) {
  (void)rnnDesc;
  (void)plan;
  fprintf(stderr, "acdnnSetPersistentRNNPlan is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t
acdnnSetRNNAlgorithmDescriptor(acdnnHandle_t handle,
                               acdnnRNNDescriptor_t rnnDesc,
                               acdnnAlgorithmDescriptor_t algoDesc) {
  (void)handle;
  (void)rnnDesc;
  (void)algoDesc;
  fprintf(stderr, "acdnnSetRNNAlgorithmDescriptor is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnSetRNNPaddingMode(acdnnRNNDescriptor_t rnnDesc,
                                                   unsigned int paddingMode) {
  (void)rnnDesc;
  (void)paddingMode;
  fprintf(stderr, "acdnnSetRNNPaddingMode is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

static inline acdnnStatus_t acdnnSetTensor(acdnnHandle_t handle,
                                           const acdnnTensorDescriptor_t yDesc,
                                           void *y, const void *valuePtr) {
  (void)handle;
  (void)yDesc;
  (void)y;
  (void)valuePtr;
  fprintf(stderr, "acdnnSetTensor is not supported.\n");
  exit(1);
  return (acdnnStatus_t)0; /* unreachable */
}

/* ── cufft: 2 unsupported APIs ── */

static inline acfftResult acfftXtSetGPUs(acfftHandle handle, int nGPUs,
                                         int *whichGPUs) {
  (void)handle;
  (void)nGPUs;
  (void)whichGPUs;
  fprintf(stderr, "acfftXtSetGPUs is not supported.\n");
  exit(1);
  return (acfftResult)0; /* unreachable */
}

static inline acfftResult acfftXtSetWorkArea(acfftHandle plan,
                                             void **workArea) {
  (void)plan;
  (void)workArea;
  fprintf(stderr, "acfftXtSetWorkArea is not supported.\n");
  exit(1);
  return (acfftResult)0; /* unreachable */
}

/* ── cufftw: 12 unsupported APIs ── */

static inline void fftw_cleanup(void) {
  fprintf(stderr, "fftw_cleanup is not supported.\n");
  exit(1);
}

static inline void fftw_export_wisdom_to_file(FILE *output_file) {
  (void)output_file;
  fprintf(stderr, "fftw_export_wisdom_to_file is not supported.\n");
  exit(1);
}

static inline void fftw_free(void *pointer) {
  (void)pointer;
  fprintf(stderr, "fftw_free is not supported.\n");
  exit(1);
}

static inline int fftw_import_wisdom_from_file(FILE *input_file) {
  (void)input_file;
  fprintf(stderr, "fftw_import_wisdom_from_file is not supported.\n");
  exit(1);
  return (int)0; /* unreachable */
}

static inline void *fftw_malloc(size_t n) {
  (void)n;
  fprintf(stderr, "fftw_malloc is not supported.\n");
  exit(1);
  return (void *)0; /* unreachable */
}

static inline void fftw_set_timelimit(double seconds) {
  (void)seconds;
  fprintf(stderr, "fftw_set_timelimit is not supported.\n");
  exit(1);
}

static inline void fftwf_cleanup(void) {
  fprintf(stderr, "fftwf_cleanup is not supported.\n");
  exit(1);
}

static inline void fftwf_export_wisdom_to_file(FILE *output_file) {
  (void)output_file;
  fprintf(stderr, "fftwf_export_wisdom_to_file is not supported.\n");
  exit(1);
}

static inline void fftwf_free(void *pointer) {
  (void)pointer;
  fprintf(stderr, "fftwf_free is not supported.\n");
  exit(1);
}

static inline int fftwf_import_wisdom_from_file(FILE *input_file) {
  (void)input_file;
  fprintf(stderr, "fftwf_import_wisdom_from_file is not supported.\n");
  exit(1);
  return (int)0; /* unreachable */
}

static inline void *fftwf_malloc(size_t n) {
  (void)n;
  fprintf(stderr, "fftwf_malloc is not supported.\n");
  exit(1);
  return (void *)0; /* unreachable */
}

static inline void fftwf_set_timelimit(double seconds) {
  (void)seconds;
  fprintf(stderr, "fftwf_set_timelimit is not supported.\n");
  exit(1);
}

/* ── cufile: 1 unsupported APIs ── */

static inline long hgFileUseCount(void) {
  fprintf(stderr, "hgFileUseCount is not supported.\n");
  exit(1);
  return (long)0; /* unreachable */
}

/* ── cupti: 1 unsupported APIs ── */

static inline HGptiResult
hgptiActivityEnableHggcEventDeviceTimestamps(uint8_t enable) {
  (void)enable;
  fprintf(stderr,
          "hgptiActivityEnableHggcEventDeviceTimestamps is not supported.\n");
  exit(1);
  return (HGptiResult)0; /* unreachable */
}

/* ── curand: 2 unsupported APIs ── */

static inline acrandStatus_t acrandGenerateBinomial(acrandGenerator_t generator,
                                                    unsigned int *outputPtr,
                                                    size_t num, unsigned int n,
                                                    double p) {
  (void)generator;
  (void)outputPtr;
  (void)num;
  (void)n;
  (void)p;
  fprintf(stderr, "acrandGenerateBinomial is not supported.\n");
  exit(1);
  return (acrandStatus_t)0; /* unreachable */
}

static inline acrandStatus_t
acrandGenerateBinomialMethod(acrandGenerator_t generator,
                             unsigned int *outputPtr, size_t num,
                             unsigned int n, double p, acrandMethod_t method) {
  (void)generator;
  (void)outputPtr;
  (void)num;
  (void)n;
  (void)p;
  (void)method;
  fprintf(stderr, "acrandGenerateBinomialMethod is not supported.\n");
  exit(1);
  return (acrandStatus_t)0; /* unreachable */
}

/* ── cusolver: 207 unsupported APIs ── */

static inline acsolverStatus_t
acsolverDnCCgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda,
                 acComplex *dB, acsolver_int_t lddb, acComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCCgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCCgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCCgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCCgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCCgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCCgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCCgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCEgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda,
                 acComplex *dB, acsolver_int_t lddb, acComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCEgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCEgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCEgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCEgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCEgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCEgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCEgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCKgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda,
                 acComplex *dB, acsolver_int_t lddb, acComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCKgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCKgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCKgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCKgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCKgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCKgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCKgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCYgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda,
                 acComplex *dB, acsolver_int_t lddb, acComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCYgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCYgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acComplex *dA, acsolver_int_t ldda, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCYgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCYgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnCYgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCYgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, acComplex *dB,
    acsolver_int_t lddb, acComplex *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnCYgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgebrd(acsolverDnHandle_t handle, int m, int n, acComplex *A, int lda,
                 float *D, float *E, acComplex *TAUQ, acComplex *TAUP,
                 acComplex *Work, int Lwork, int *devInfo) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)D;
  (void)E;
  (void)TAUQ;
  (void)TAUP;
  (void)Work;
  (void)Lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCgebrd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgebrd_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            int *Lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)Lwork;
  fprintf(stderr, "acsolverDnCgebrd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCgeqrf(acsolverDnHandle_t handle,
                                                int m, int n, acComplex *A,
                                                int lda, acComplex *TAU,
                                                acComplex *Workspace, int Lwork,
                                                int *devInfo) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)TAU;
  (void)Workspace;
  (void)Lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCgeqrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgeqrf_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            acComplex *A, int lda, int *lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnCgeqrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgesvd(acsolverDnHandle_t handle, signed char jobu, signed char jobvt,
                 int m, int n, acComplex *A, int lda, float *S, acComplex *U,
                 int ldu, acComplex *VT, int ldvt, acComplex *work, int lwork,
                 float *rwork, int *info) {
  (void)handle;
  (void)jobu;
  (void)jobvt;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)VT;
  (void)ldvt;
  (void)work;
  (void)lwork;
  (void)rwork;
  (void)info;
  fprintf(stderr, "acsolverDnCgesvd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgesvd_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            int *lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)lwork;
  fprintf(stderr, "acsolverDnCgesvd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCgesvdaStridedBatched(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, int rank, int m, int n,
    const acComplex *d_A, int lda, long long strideA, float *d_S,
    long long strideS, acComplex *d_U, int ldu, long long strideU,
    acComplex *d_V, int ldv, long long strideV, acComplex *d_work, int lwork,
    int *d_info, double *h_R_nrmF, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)rank;
  (void)m;
  (void)n;
  (void)d_A;
  (void)lda;
  (void)strideA;
  (void)d_S;
  (void)strideS;
  (void)d_U;
  (void)ldu;
  (void)strideU;
  (void)d_V;
  (void)ldv;
  (void)strideV;
  (void)d_work;
  (void)lwork;
  (void)d_info;
  (void)h_R_nrmF;
  (void)batchSize;
  fprintf(stderr, "acsolverDnCgesvdaStridedBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCgesvdaStridedBatched_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, int rank, int m, int n,
    const acComplex *d_A, int lda, long long strideA, const float *d_S,
    long long strideS, const acComplex *d_U, int ldu, long long strideU,
    const acComplex *d_V, int ldv, long long strideV, int *lwork,
    int batchSize) {
  (void)handle;
  (void)jobz;
  (void)rank;
  (void)m;
  (void)n;
  (void)d_A;
  (void)lda;
  (void)strideA;
  (void)d_S;
  (void)strideS;
  (void)d_U;
  (void)ldu;
  (void)strideU;
  (void)d_V;
  (void)ldv;
  (void)strideV;
  (void)lwork;
  (void)batchSize;
  fprintf(stderr,
          "acsolverDnCgesvdaStridedBatched_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgesvdj(acsolverDnHandle_t handle, acsolverEigMode_t jobz, int econ,
                  int m, int n, acComplex *A, int lda, float *S, acComplex *U,
                  int ldu, acComplex *V, int ldv, acComplex *work, int lwork,
                  int *info, gesvdjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)econ;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  fprintf(stderr, "acsolverDnCgesvdj is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgesvdjBatched(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                         int m, int n, acComplex *A, int lda, float *S,
                         acComplex *U, int ldu, acComplex *V, int ldv,
                         acComplex *work, int lwork, int *info,
                         gesvdjInfo_t params, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnCgesvdjBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCgesvdjBatched_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, int m, int n,
    const acComplex *A, int lda, const float *S, const acComplex *U, int ldu,
    const acComplex *V, int ldv, int *lwork, gesvdjInfo_t params,
    int batchSize) {
  (void)handle;
  (void)jobz;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)lwork;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnCgesvdjBatched_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCgesvdj_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, int econ, int m, int n,
    const acComplex *A, int lda, const float *S, const acComplex *U, int ldu,
    const acComplex *V, int ldv, int *lwork, gesvdjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)econ;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)lwork;
  (void)params;
  fprintf(stderr, "acsolverDnCgesvdj_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCgetrf(acsolverDnHandle_t handle,
                                                int m, int n, acComplex *A,
                                                int lda, acComplex *Workspace,
                                                int *devIpiv, int *devInfo) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)Workspace;
  (void)devIpiv;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCgetrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgetrf_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            acComplex *A, int lda, int *Lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)Lwork;
  fprintf(stderr, "acsolverDnCgetrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCgetrs(acsolverDnHandle_t handle, acblasOperation_t trans, int n,
                 int nrhs, const acComplex *A, int lda, const int *devIpiv,
                 acComplex *B, int ldb, int *devInfo) {
  (void)handle;
  (void)trans;
  (void)n;
  (void)nrhs;
  (void)A;
  (void)lda;
  (void)devIpiv;
  (void)B;
  (void)ldb;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCgetrs is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCheevd(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                 acblasFillMode_t uplo, int n, acComplex *A, int lda, float *W,
                 acComplex *work, int lwork, int *info) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnCheevd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCheevd_bufferSize(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                            acblasFillMode_t uplo, int n, const acComplex *A,
                            int lda, const float *W, int *lwork) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)lwork;
  fprintf(stderr, "acsolverDnCheevd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCheevj(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                 acblasFillMode_t uplo, int n, acComplex *A, int lda, float *W,
                 acComplex *work, int lwork, int *info, syevjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  fprintf(stderr, "acsolverDnCheevj is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCheevjBatched(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                        acblasFillMode_t uplo, int n, acComplex *A, int lda,
                        float *W, acComplex *work, int lwork, int *info,
                        syevjInfo_t params, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnCheevjBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCheevjBatched_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, acblasFillMode_t uplo,
    int n, const acComplex *A, int lda, const float *W, int *lwork,
    syevjInfo_t params, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)lwork;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnCheevjBatched_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCheevj_bufferSize(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                            acblasFillMode_t uplo, int n, const acComplex *A,
                            int lda, const float *W, int *lwork,
                            syevjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)lwork;
  (void)params;
  fprintf(stderr, "acsolverDnCheevj_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnChetrd(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 acComplex *A, int lda, float *d, float *e, acComplex *tau,
                 acComplex *work, int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)d;
  (void)e;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnChetrd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnChetrd_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, const acComplex *A, int lda, const float *d,
                            const float *e, const acComplex *tau, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)d;
  (void)e;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnChetrd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnClaswp(acsolverDnHandle_t handle,
                                                int n, acComplex *A, int lda,
                                                int k1, int k2,
                                                const int *devIpiv, int incx) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)k1;
  (void)k2;
  (void)devIpiv;
  (void)incx;
  fprintf(stderr, "acsolverDnClaswp is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnClauum(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                acComplex *A, int lda,
                                                acComplex *work, int lwork,
                                                int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnClauum is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnClauum_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acComplex *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnClauum_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCpotrf(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                acComplex *A, int lda,
                                                acComplex *Workspace, int Lwork,
                                                int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)Workspace;
  (void)Lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCpotrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCpotrfBatched(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                        acComplex *Aarray[], int lda, int *infoArray,
                        int batchSize) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)Aarray;
  (void)lda;
  (void)infoArray;
  (void)batchSize;
  fprintf(stderr, "acsolverDnCpotrfBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCpotrf_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acComplex *A, int lda, int *Lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)Lwork;
  fprintf(stderr, "acsolverDnCpotrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCpotri(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                acComplex *A, int lda,
                                                acComplex *work, int lwork,
                                                int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCpotri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCpotri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acComplex *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnCpotri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCpotrs(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                int nrhs, const acComplex *A,
                                                int lda, acComplex *B, int ldb,
                                                int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCpotrs is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCpotrsBatched(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                        int nrhs, acComplex *A[], int lda, acComplex *B[],
                        int ldb, int *d_info, int batchSize) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)d_info;
  (void)batchSize;
  fprintf(stderr, "acsolverDnCpotrsBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCsytrf(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                acComplex *A, int lda,
                                                int *ipiv, acComplex *work,
                                                int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnCsytrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCsytrf_bufferSize(acsolverDnHandle_t handle, int n, acComplex *A,
                            int lda, int *lwork) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnCsytrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCsytri(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 acComplex *A, int lda, const int *ipiv, acComplex *work,
                 int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnCsytri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCsytri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acComplex *A, int lda, const int *ipiv,
                            int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)lwork;
  fprintf(stderr, "acsolverDnCsytri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCungbr(acsolverDnHandle_t handle, acblasSideMode_t side, int m, int n,
                 int k, acComplex *A, int lda, const acComplex *tau,
                 acComplex *work, int lwork, int *info) {
  (void)handle;
  (void)side;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnCungbr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCungbr_bufferSize(acsolverDnHandle_t handle, acblasSideMode_t side,
                            int m, int n, int k, const acComplex *A, int lda,
                            const acComplex *tau, int *lwork) {
  (void)handle;
  (void)side;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnCungbr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCungqr(acsolverDnHandle_t handle, int m, int n, int k, acComplex *A,
                 int lda, const acComplex *tau, acComplex *work, int lwork,
                 int *info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnCungqr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCungqr_bufferSize(acsolverDnHandle_t handle, int m, int n, int k,
                            const acComplex *A, int lda, const acComplex *tau,
                            int *lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnCungqr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCungtr(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 acComplex *A, int lda, const acComplex *tau, acComplex *work,
                 int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnCungtr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCungtr_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, const acComplex *A, int lda,
                            const acComplex *tau, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnCungtr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCunmqr(
    acsolverDnHandle_t handle, acblasSideMode_t side, acblasOperation_t trans,
    int m, int n, int k, const acComplex *A, int lda, const acComplex *tau,
    acComplex *C, int ldc, acComplex *work, int lwork, int *devInfo) {
  (void)handle;
  (void)side;
  (void)trans;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnCunmqr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCunmqr_bufferSize(acsolverDnHandle_t handle, acblasSideMode_t side,
                            acblasOperation_t trans, int m, int n, int k,
                            const acComplex *A, int lda, const acComplex *tau,
                            const acComplex *C, int ldc, int *lwork) {
  (void)handle;
  (void)side;
  (void)trans;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)lwork;
  fprintf(stderr, "acsolverDnCunmqr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnCunmtr(acsolverDnHandle_t handle, acblasSideMode_t side,
                 acblasFillMode_t uplo, acblasOperation_t trans, int m, int n,
                 acComplex *A, int lda, acComplex *tau, acComplex *C, int ldc,
                 acComplex *work, int lwork, int *info) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnCunmtr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnCunmtr_bufferSize(
    acsolverDnHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, int m, int n, const acComplex *A, int lda,
    const acComplex *tau, const acComplex *C, int ldc, int *lwork) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)lwork;
  fprintf(stderr, "acsolverDnCunmtr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDBgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, double *dA, acsolver_int_t ldda,
                 double *dB, acsolver_int_t lddb, double *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDBgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDBgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, double *dA, acsolver_int_t ldda, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDBgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDBgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDBgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDBgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDBgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDDgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, double *dA, acsolver_int_t ldda,
                 double *dB, acsolver_int_t lddb, double *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDDgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDDgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, double *dA, acsolver_int_t ldda, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDDgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDDgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDDgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDDgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDDgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDHgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, double *dA, acsolver_int_t ldda,
                 double *dB, acsolver_int_t lddb, double *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDHgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDHgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, double *dA, acsolver_int_t ldda, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDHgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDHgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDHgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDHgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDHgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDSgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, double *dA, acsolver_int_t ldda,
                 double *dB, acsolver_int_t lddb, double *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDSgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDSgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, double *dA, acsolver_int_t ldda, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDSgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDSgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDSgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDSgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDSgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDXgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, double *dA, acsolver_int_t ldda,
                 double *dB, acsolver_int_t lddb, double *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDXgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDXgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, double *dA, acsolver_int_t ldda, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDXgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDXgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t lwork_bytes, acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnDXgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDXgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    double *dA, acsolver_int_t ldda, acsolver_int_t *dipiv, double *dB,
    acsolver_int_t lddb, double *dX, acsolver_int_t lddx, void *dWorkspace,
    size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnDXgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDgemmHost(acblasOperation_t transa, acblasOperation_t transb, int m,
                    int n, int k, const double *alpha, const double *A, int lda,
                    const double *B, int ldb, const double *beta, double *C,
                    int ldc) {
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acsolverDnDgemmHost is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDlaswp(acsolverDnHandle_t handle,
                                                int n, double *A, int lda,
                                                int k1, int k2,
                                                const int *devIpiv, int incx) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)k1;
  (void)k2;
  (void)devIpiv;
  (void)incx;
  fprintf(stderr, "acsolverDnDlaswp is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDlauum(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 double *A, int lda, double *work, int lwork, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnDlauum is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDlauum_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, double *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnDlauum_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDpotri(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 double *A, int lda, double *work, int lwork, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnDpotri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDpotri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, double *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnDpotri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDsteqrHost(const signed char *compz,
                                                    int n, double *d, double *e,
                                                    double *z, int ldz,
                                                    double *work, int *info) {
  (void)compz;
  (void)n;
  (void)d;
  (void)e;
  (void)z;
  (void)ldz;
  (void)work;
  (void)info;
  fprintf(stderr, "acsolverDnDsteqrHost is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDsterfHost(int n, double *d, double *e,
                                                    int *info) {
  (void)n;
  (void)d;
  (void)e;
  (void)info;
  fprintf(stderr, "acsolverDnDsterfHost is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDsytrf(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                double *A, int lda, int *ipiv,
                                                double *work, int lwork,
                                                int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnDsytrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDsytrf_bufferSize(acsolverDnHandle_t handle, int n, double *A,
                            int lda, int *lwork) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnDsytrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnDsytri(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                double *A, int lda,
                                                const int *ipiv, double *work,
                                                int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnDsytri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnDsytri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, double *A, int lda, const int *ipiv,
                            int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)lwork;
  fprintf(stderr, "acsolverDnDsytri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnLoggerForceDisable(void) {
  fprintf(stderr, "acsolverDnLoggerForceDisable is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnLoggerOpenFile(const char *logFile) {
  (void)logFile;
  fprintf(stderr, "acsolverDnLoggerOpenFile is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnLoggerSetFile(FILE *file) {
  (void)file;
  fprintf(stderr, "acsolverDnLoggerSetFile is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnLoggerSetLevel(int level) {
  (void)level;
  fprintf(stderr, "acsolverDnLoggerSetLevel is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnLoggerSetMask(int mask) {
  (void)mask;
  fprintf(stderr, "acsolverDnLoggerSetMask is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSBgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, float *dA, acsolver_int_t ldda, float *dB,
                 acsolver_int_t lddb, float *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSBgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSBgels_bufferSize(acsolverDnHandle_t handle, acsolver_int_t m,
                            acsolver_int_t n, acsolver_int_t nrhs, float *dA,
                            acsolver_int_t ldda, float *dB, acsolver_int_t lddb,
                            float *dX, acsolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSBgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSBgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSBgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSBgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSBgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSHgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, float *dA, acsolver_int_t ldda, float *dB,
                 acsolver_int_t lddb, float *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSHgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSHgels_bufferSize(acsolverDnHandle_t handle, acsolver_int_t m,
                            acsolver_int_t n, acsolver_int_t nrhs, float *dA,
                            acsolver_int_t ldda, float *dB, acsolver_int_t lddb,
                            float *dX, acsolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSHgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSHgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSHgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSHgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSHgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSSgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, float *dA, acsolver_int_t ldda, float *dB,
                 acsolver_int_t lddb, float *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSSgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSSgels_bufferSize(acsolverDnHandle_t handle, acsolver_int_t m,
                            acsolver_int_t n, acsolver_int_t nrhs, float *dA,
                            acsolver_int_t ldda, float *dB, acsolver_int_t lddb,
                            float *dX, acsolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSSgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSSgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSSgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSSgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSSgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSXgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, float *dA, acsolver_int_t ldda, float *dB,
                 acsolver_int_t lddb, float *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSXgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSXgels_bufferSize(acsolverDnHandle_t handle, acsolver_int_t m,
                            acsolver_int_t n, acsolver_int_t nrhs, float *dA,
                            acsolver_int_t ldda, float *dB, acsolver_int_t lddb,
                            float *dX, acsolver_int_t lddx, void *dWorkspace,
                            size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSXgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSXgesv(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
    acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnSXgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSXgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs, float *dA,
    acsolver_int_t ldda, acsolver_int_t *dipiv, float *dB, acsolver_int_t lddb,
    float *dX, acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnSXgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSgemmHost(acblasOperation_t transa, acblasOperation_t transb, int m,
                    int n, int k, const float *alpha, const float *A, int lda,
                    const float *B, int ldb, const float *beta, float *C,
                    int ldc) {
  (void)transa;
  (void)transb;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)beta;
  (void)C;
  (void)ldc;
  fprintf(stderr, "acsolverDnSgemmHost is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSlaswp(acsolverDnHandle_t handle,
                                                int n, float *A, int lda,
                                                int k1, int k2,
                                                const int *devIpiv, int incx) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)k1;
  (void)k2;
  (void)devIpiv;
  (void)incx;
  fprintf(stderr, "acsolverDnSlaswp is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSlauum(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                float *A, int lda, float *work,
                                                int lwork, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnSlauum is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSlauum_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, float *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnSlauum_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSpotri(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                float *A, int lda, float *work,
                                                int lwork, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnSpotri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSpotri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, float *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnSpotri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSsteqrHost(const signed char *compz,
                                                    int n, float *d, float *e,
                                                    float *z, int ldz,
                                                    float *work, int *info) {
  (void)compz;
  (void)n;
  (void)d;
  (void)e;
  (void)z;
  (void)ldz;
  (void)work;
  (void)info;
  fprintf(stderr, "acsolverDnSsteqrHost is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSsterfHost(int n, float *d, float *e,
                                                    int *info) {
  (void)n;
  (void)d;
  (void)e;
  (void)info;
  fprintf(stderr, "acsolverDnSsterfHost is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSsytrf(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                float *A, int lda, int *ipiv,
                                                float *work, int lwork,
                                                int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnSsytrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSsytrf_bufferSize(acsolverDnHandle_t handle, int n, float *A, int lda,
                            int *lwork) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnSsytrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnSsytri(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                float *A, int lda,
                                                const int *ipiv, float *work,
                                                int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnSsytri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnSsytri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, float *A, int lda, const int *ipiv,
                            int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)lwork;
  fprintf(stderr, "acsolverDnSsytri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnXgeev(acsolverDnHandle_t handle, acsolverDnParams_t params,
                acsolverEigMode_t jobvl, acsolverEigMode_t jobvr, int64_t n,
                hggcDataType dataTypeA, void *A, int64_t lda,
                hggcDataType dataTypeW, void *W, hggcDataType dataTypeVL,
                void *VL, int64_t ldvl, hggcDataType dataTypeVR, void *VR,
                int64_t ldvr, hggcDataType computeType, void *bufferOnDevice,
                size_t workspaceInBytesOnDevice, void *bufferOnHost,
                size_t workspaceInBytesOnHost, int *info) {
  (void)handle;
  (void)params;
  (void)jobvl;
  (void)jobvr;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeW;
  (void)W;
  (void)dataTypeVL;
  (void)VL;
  (void)ldvl;
  (void)dataTypeVR;
  (void)VR;
  (void)ldvr;
  (void)computeType;
  (void)bufferOnDevice;
  (void)workspaceInBytesOnDevice;
  (void)bufferOnHost;
  (void)workspaceInBytesOnHost;
  (void)info;
  fprintf(stderr, "acsolverDnXgeev is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXgeev_bufferSize(
    acsolverDnHandle_t handle, acsolverDnParams_t params,
    acsolverEigMode_t jobvl, acsolverEigMode_t jobvr, int64_t n,
    hggcDataType dataTypeA, const void *A, int64_t lda, hggcDataType dataTypeW,
    const void *W, hggcDataType dataTypeVL, const void *VL, int64_t ldvl,
    hggcDataType dataTypeVR, const void *VR, int64_t ldvr,
    hggcDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
  (void)handle;
  (void)params;
  (void)jobvl;
  (void)jobvr;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeW;
  (void)W;
  (void)dataTypeVL;
  (void)VL;
  (void)ldvl;
  (void)dataTypeVR;
  (void)VR;
  (void)ldvr;
  (void)computeType;
  (void)workspaceInBytesOnDevice;
  (void)workspaceInBytesOnHost;
  fprintf(stderr, "acsolverDnXgeev_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXgesvdp(
    acsolverDnHandle_t handle, acsolverDnParams_t params,
    acsolverEigMode_t jobz, int econ, int64_t m, int64_t n,
    hggcDataType dataTypeA, void *A, int64_t lda, hggcDataType dataTypeS,
    void *S, hggcDataType dataTypeU, void *U, int64_t ldu,
    hggcDataType dataTypeV, void *V, int64_t ldv, hggcDataType computeType,
    void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *d_info, double *h_err_sigma) {
  (void)handle;
  (void)params;
  (void)jobz;
  (void)econ;
  (void)m;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeS;
  (void)S;
  (void)dataTypeU;
  (void)U;
  (void)ldu;
  (void)dataTypeV;
  (void)V;
  (void)ldv;
  (void)computeType;
  (void)bufferOnDevice;
  (void)workspaceInBytesOnDevice;
  (void)bufferOnHost;
  (void)workspaceInBytesOnHost;
  (void)d_info;
  (void)h_err_sigma;
  fprintf(stderr, "acsolverDnXgesvdp is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXgesvdp_bufferSize(
    acsolverDnHandle_t handle, acsolverDnParams_t params,
    acsolverEigMode_t jobz, int econ, int64_t m, int64_t n,
    hggcDataType dataTypeA, const void *A, int64_t lda, hggcDataType dataTypeS,
    const void *S, hggcDataType dataTypeU, const void *U, int64_t ldu,
    hggcDataType dataTypeV, const void *V, int64_t ldv,
    hggcDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
  (void)handle;
  (void)params;
  (void)jobz;
  (void)econ;
  (void)m;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeS;
  (void)S;
  (void)dataTypeU;
  (void)U;
  (void)ldu;
  (void)dataTypeV;
  (void)V;
  (void)ldv;
  (void)computeType;
  (void)workspaceInBytesOnDevice;
  (void)workspaceInBytesOnHost;
  fprintf(stderr, "acsolverDnXgesvdp_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnXgesvdr(acsolverDnHandle_t handle, acsolverDnParams_t params,
                  signed char jobu, signed char jobv, int64_t m, int64_t n,
                  int64_t k, int64_t p, int64_t niters, hggcDataType dataTypeA,
                  void *A, int64_t lda, hggcDataType dataTypeSrand, void *Srand,
                  hggcDataType dataTypeUrand, void *Urand, int64_t ldUrand,
                  hggcDataType dataTypeVrand, void *Vrand, int64_t ldVrand,
                  hggcDataType computeType, void *bufferOnDevice,
                  size_t workspaceInBytesOnDevice, void *bufferOnHost,
                  size_t workspaceInBytesOnHost, int *d_info) {
  (void)handle;
  (void)params;
  (void)jobu;
  (void)jobv;
  (void)m;
  (void)n;
  (void)k;
  (void)p;
  (void)niters;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeSrand;
  (void)Srand;
  (void)dataTypeUrand;
  (void)Urand;
  (void)ldUrand;
  (void)dataTypeVrand;
  (void)Vrand;
  (void)ldVrand;
  (void)computeType;
  (void)bufferOnDevice;
  (void)workspaceInBytesOnDevice;
  (void)bufferOnHost;
  (void)workspaceInBytesOnHost;
  (void)d_info;
  fprintf(stderr, "acsolverDnXgesvdr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXgesvdr_bufferSize(
    acsolverDnHandle_t handle, acsolverDnParams_t params, signed char jobu,
    signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p,
    int64_t niters, hggcDataType dataTypeA, const void *A, int64_t lda,
    hggcDataType dataTypeSrand, const void *Srand, hggcDataType dataTypeUrand,
    const void *Urand, int64_t ldUrand, hggcDataType dataTypeVrand,
    const void *Vrand, int64_t ldVrand, hggcDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  (void)handle;
  (void)params;
  (void)jobu;
  (void)jobv;
  (void)m;
  (void)n;
  (void)k;
  (void)p;
  (void)niters;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeSrand;
  (void)Srand;
  (void)dataTypeUrand;
  (void)Urand;
  (void)ldUrand;
  (void)dataTypeVrand;
  (void)Vrand;
  (void)ldVrand;
  (void)computeType;
  (void)workspaceInBytesOnDevice;
  (void)workspaceInBytesOnHost;
  fprintf(stderr, "acsolverDnXgesvdr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXsyevBatched(
    acsolverDnHandle_t handle, acsolverDnParams_t params,
    acsolverEigMode_t jobz, acblasFillMode_t uplo, int64_t n,
    hggcDataType dataTypeA, void *A, int64_t lda, hggcDataType dataTypeW,
    void *W, hggcDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost,
    size_t workspaceInBytesOnHost, int *info, int64_t batchSize) {
  (void)handle;
  (void)params;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeW;
  (void)W;
  (void)computeType;
  (void)bufferOnDevice;
  (void)workspaceInBytesOnDevice;
  (void)bufferOnHost;
  (void)workspaceInBytesOnHost;
  (void)info;
  (void)batchSize;
  fprintf(stderr, "acsolverDnXsyevBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXsyevBatched_bufferSize(
    acsolverDnHandle_t handle, acsolverDnParams_t params,
    acsolverEigMode_t jobz, acblasFillMode_t uplo, int64_t n,
    hggcDataType dataTypeA, const void *A, int64_t lda, hggcDataType dataTypeW,
    const void *W, hggcDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost, int64_t batchSize) {
  (void)handle;
  (void)params;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)dataTypeW;
  (void)W;
  (void)computeType;
  (void)workspaceInBytesOnDevice;
  (void)workspaceInBytesOnHost;
  (void)batchSize;
  fprintf(stderr, "acsolverDnXsyevBatched_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnXsytrs(acsolverDnHandle_t handle, acblasFillMode_t uplo, int64_t n,
                 int64_t nrhs, hggcDataType dataTypeA, const void *A,
                 int64_t lda, const int64_t *ipiv, hggcDataType dataTypeB,
                 void *B, int64_t ldb, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)dataTypeB;
  (void)B;
  (void)ldb;
  (void)bufferOnDevice;
  (void)workspaceInBytesOnDevice;
  (void)bufferOnHost;
  (void)workspaceInBytesOnHost;
  (void)info;
  fprintf(stderr, "acsolverDnXsytrs is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXsytrs_bufferSize(
    acsolverDnHandle_t handle, acblasFillMode_t uplo, int64_t n, int64_t nrhs,
    hggcDataType dataTypeA, const void *A, int64_t lda, const int64_t *ipiv,
    hggcDataType dataTypeB, void *B, int64_t ldb,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)dataTypeB;
  (void)B;
  (void)ldb;
  (void)workspaceInBytesOnDevice;
  (void)workspaceInBytesOnHost;
  fprintf(stderr, "acsolverDnXsytrs_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnXtrtri(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                 acblasDiagType_t diag, int64_t n, hggcDataType dataTypeA,
                 void *A, int64_t lda, void *bufferOnDevice,
                 size_t workspaceInBytesOnDevice, void *bufferOnHost,
                 size_t workspaceInBytesOnHost, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)diag;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)bufferOnDevice;
  (void)workspaceInBytesOnDevice;
  (void)bufferOnHost;
  (void)workspaceInBytesOnHost;
  (void)devInfo;
  fprintf(stderr, "acsolverDnXtrtri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnXtrtri_bufferSize(
    acsolverDnHandle_t handle, acblasFillMode_t uplo, acblasDiagType_t diag,
    int64_t n, hggcDataType dataTypeA, void *A, int64_t lda,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  (void)handle;
  (void)uplo;
  (void)diag;
  (void)n;
  (void)dataTypeA;
  (void)A;
  (void)lda;
  (void)workspaceInBytesOnDevice;
  (void)workspaceInBytesOnHost;
  fprintf(stderr, "acsolverDnXtrtri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZCgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZCgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZCgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZCgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZCgesv(acsolverDnHandle_t handle, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acsolver_int_t *dipiv, acDoubleComplex *dB,
                 acsolver_int_t lddb, acDoubleComplex *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZCgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZCgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acDoubleComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZCgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZEgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZEgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZEgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZEgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZEgesv(acsolverDnHandle_t handle, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acsolver_int_t *dipiv, acDoubleComplex *dB,
                 acsolver_int_t lddb, acDoubleComplex *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZEgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZEgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acDoubleComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZEgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZKgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZKgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZKgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZKgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZKgesv(acsolverDnHandle_t handle, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acsolver_int_t *dipiv, acDoubleComplex *dB,
                 acsolver_int_t lddb, acDoubleComplex *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZKgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZKgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acDoubleComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZKgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZYgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZYgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZYgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZYgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZYgesv(acsolverDnHandle_t handle, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acsolver_int_t *dipiv, acDoubleComplex *dB,
                 acsolver_int_t lddb, acDoubleComplex *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZYgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZYgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acDoubleComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZYgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZZgels(acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
                 acsolver_int_t lddx, void *dWorkspace, size_t lwork_bytes,
                 acsolver_int_t *iter, acsolver_int_t *d_info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZZgels is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZZgels_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t m, acsolver_int_t n,
    acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZZgels_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZZgesv(acsolverDnHandle_t handle, acsolver_int_t n,
                 acsolver_int_t nrhs, acDoubleComplex *dA, acsolver_int_t ldda,
                 acsolver_int_t *dipiv, acDoubleComplex *dB,
                 acsolver_int_t lddb, acDoubleComplex *dX, acsolver_int_t lddx,
                 void *dWorkspace, size_t lwork_bytes, acsolver_int_t *iter,
                 acsolver_int_t *d_info) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  (void)iter;
  (void)d_info;
  fprintf(stderr, "acsolverDnZZgesv is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZZgesv_bufferSize(
    acsolverDnHandle_t handle, acsolver_int_t n, acsolver_int_t nrhs,
    acDoubleComplex *dA, acsolver_int_t ldda, acsolver_int_t *dipiv,
    acDoubleComplex *dB, acsolver_int_t lddb, acDoubleComplex *dX,
    acsolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
  (void)handle;
  (void)n;
  (void)nrhs;
  (void)dA;
  (void)ldda;
  (void)dipiv;
  (void)dB;
  (void)lddb;
  (void)dX;
  (void)lddx;
  (void)dWorkspace;
  (void)lwork_bytes;
  fprintf(stderr, "acsolverDnZZgesv_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgebrd(acsolverDnHandle_t handle, int m, int n, acDoubleComplex *A,
                 int lda, double *D, double *E, acDoubleComplex *TAUQ,
                 acDoubleComplex *TAUP, acDoubleComplex *Work, int Lwork,
                 int *devInfo) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)D;
  (void)E;
  (void)TAUQ;
  (void)TAUP;
  (void)Work;
  (void)Lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZgebrd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgebrd_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            int *Lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)Lwork;
  fprintf(stderr, "acsolverDnZgebrd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgeqrf(acsolverDnHandle_t handle, int m, int n, acDoubleComplex *A,
                 int lda, acDoubleComplex *TAU, acDoubleComplex *Workspace,
                 int Lwork, int *devInfo) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)TAU;
  (void)Workspace;
  (void)Lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZgeqrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgeqrf_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            acDoubleComplex *A, int lda, int *lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnZgeqrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgesvd(acsolverDnHandle_t handle, signed char jobu, signed char jobvt,
                 int m, int n, acDoubleComplex *A, int lda, double *S,
                 acDoubleComplex *U, int ldu, acDoubleComplex *VT, int ldvt,
                 acDoubleComplex *work, int lwork, double *rwork, int *info) {
  (void)handle;
  (void)jobu;
  (void)jobvt;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)VT;
  (void)ldvt;
  (void)work;
  (void)lwork;
  (void)rwork;
  (void)info;
  fprintf(stderr, "acsolverDnZgesvd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgesvd_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            int *lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)lwork;
  fprintf(stderr, "acsolverDnZgesvd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZgesvdaStridedBatched(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, int rank, int m, int n,
    const acDoubleComplex *d_A, int lda, long long strideA, double *d_S,
    long long strideS, acDoubleComplex *d_U, int ldu, long long strideU,
    acDoubleComplex *d_V, int ldv, long long strideV, acDoubleComplex *d_work,
    int lwork, int *d_info, double *h_R_nrmF, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)rank;
  (void)m;
  (void)n;
  (void)d_A;
  (void)lda;
  (void)strideA;
  (void)d_S;
  (void)strideS;
  (void)d_U;
  (void)ldu;
  (void)strideU;
  (void)d_V;
  (void)ldv;
  (void)strideV;
  (void)d_work;
  (void)lwork;
  (void)d_info;
  (void)h_R_nrmF;
  (void)batchSize;
  fprintf(stderr, "acsolverDnZgesvdaStridedBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZgesvdaStridedBatched_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, int rank, int m, int n,
    const acDoubleComplex *d_A, int lda, long long strideA, const double *d_S,
    long long strideS, const acDoubleComplex *d_U, int ldu, long long strideU,
    const acDoubleComplex *d_V, int ldv, long long strideV, int *lwork,
    int batchSize) {
  (void)handle;
  (void)jobz;
  (void)rank;
  (void)m;
  (void)n;
  (void)d_A;
  (void)lda;
  (void)strideA;
  (void)d_S;
  (void)strideS;
  (void)d_U;
  (void)ldu;
  (void)strideU;
  (void)d_V;
  (void)ldv;
  (void)strideV;
  (void)lwork;
  (void)batchSize;
  fprintf(stderr,
          "acsolverDnZgesvdaStridedBatched_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgesvdj(acsolverDnHandle_t handle, acsolverEigMode_t jobz, int econ,
                  int m, int n, acDoubleComplex *A, int lda, double *S,
                  acDoubleComplex *U, int ldu, acDoubleComplex *V, int ldv,
                  acDoubleComplex *work, int lwork, int *info,
                  gesvdjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)econ;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  fprintf(stderr, "acsolverDnZgesvdj is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgesvdjBatched(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                         int m, int n, acDoubleComplex *A, int lda, double *S,
                         acDoubleComplex *U, int ldu, acDoubleComplex *V,
                         int ldv, acDoubleComplex *work, int lwork, int *info,
                         gesvdjInfo_t params, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnZgesvdjBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZgesvdjBatched_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, int m, int n,
    const acDoubleComplex *A, int lda, const double *S,
    const acDoubleComplex *U, int ldu, const acDoubleComplex *V, int ldv,
    int *lwork, gesvdjInfo_t params, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)lwork;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnZgesvdjBatched_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgesvdj_bufferSize(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                             int econ, int m, int n, const acDoubleComplex *A,
                             int lda, const double *S, const acDoubleComplex *U,
                             int ldu, const acDoubleComplex *V, int ldv,
                             int *lwork, gesvdjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)econ;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)S;
  (void)U;
  (void)ldu;
  (void)V;
  (void)ldv;
  (void)lwork;
  (void)params;
  fprintf(stderr, "acsolverDnZgesvdj_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZgetrf(acsolverDnHandle_t handle,
                                                int m, int n,
                                                acDoubleComplex *A, int lda,
                                                acDoubleComplex *Workspace,
                                                int *devIpiv, int *devInfo) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)Workspace;
  (void)devIpiv;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZgetrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgetrf_bufferSize(acsolverDnHandle_t handle, int m, int n,
                            acDoubleComplex *A, int lda, int *Lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)Lwork;
  fprintf(stderr, "acsolverDnZgetrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZgetrs(acsolverDnHandle_t handle, acblasOperation_t trans, int n,
                 int nrhs, const acDoubleComplex *A, int lda,
                 const int *devIpiv, acDoubleComplex *B, int ldb,
                 int *devInfo) {
  (void)handle;
  (void)trans;
  (void)n;
  (void)nrhs;
  (void)A;
  (void)lda;
  (void)devIpiv;
  (void)B;
  (void)ldb;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZgetrs is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZheevd(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                 acblasFillMode_t uplo, int n, acDoubleComplex *A, int lda,
                 double *W, acDoubleComplex *work, int lwork, int *info) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZheevd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZheevd_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, acblasFillMode_t uplo,
    int n, const acDoubleComplex *A, int lda, const double *W, int *lwork) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)lwork;
  fprintf(stderr, "acsolverDnZheevd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZheevj(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                 acblasFillMode_t uplo, int n, acDoubleComplex *A, int lda,
                 double *W, acDoubleComplex *work, int lwork, int *info,
                 syevjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  fprintf(stderr, "acsolverDnZheevj is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZheevjBatched(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                        acblasFillMode_t uplo, int n, acDoubleComplex *A,
                        int lda, double *W, acDoubleComplex *work, int lwork,
                        int *info, syevjInfo_t params, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)work;
  (void)lwork;
  (void)info;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnZheevjBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZheevjBatched_bufferSize(
    acsolverDnHandle_t handle, acsolverEigMode_t jobz, acblasFillMode_t uplo,
    int n, const acDoubleComplex *A, int lda, const double *W, int *lwork,
    syevjInfo_t params, int batchSize) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)lwork;
  (void)params;
  (void)batchSize;
  fprintf(stderr, "acsolverDnZheevjBatched_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZheevj_bufferSize(acsolverDnHandle_t handle, acsolverEigMode_t jobz,
                            acblasFillMode_t uplo, int n,
                            const acDoubleComplex *A, int lda, const double *W,
                            int *lwork, syevjInfo_t params) {
  (void)handle;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)W;
  (void)lwork;
  (void)params;
  fprintf(stderr, "acsolverDnZheevj_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZhetrd(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 acDoubleComplex *A, int lda, double *d, double *e,
                 acDoubleComplex *tau, acDoubleComplex *work, int lwork,
                 int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)d;
  (void)e;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZhetrd is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZhetrd_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, const acDoubleComplex *A, int lda,
                            const double *d, const double *e,
                            const acDoubleComplex *tau, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)d;
  (void)e;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnZhetrd_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZlaswp(acsolverDnHandle_t handle,
                                                int n, acDoubleComplex *A,
                                                int lda, int k1, int k2,
                                                const int *devIpiv, int incx) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)k1;
  (void)k2;
  (void)devIpiv;
  (void)incx;
  fprintf(stderr, "acsolverDnZlaswp is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZlauum(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                acDoubleComplex *A, int lda,
                                                acDoubleComplex *work,
                                                int lwork, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZlauum is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZlauum_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acDoubleComplex *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnZlauum_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZpotrf(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                acDoubleComplex *A, int lda,
                                                acDoubleComplex *Workspace,
                                                int Lwork, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)Workspace;
  (void)Lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZpotrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZpotrfBatched(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                        acDoubleComplex *Aarray[], int lda, int *infoArray,
                        int batchSize) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)Aarray;
  (void)lda;
  (void)infoArray;
  (void)batchSize;
  fprintf(stderr, "acsolverDnZpotrfBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZpotrf_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acDoubleComplex *A, int lda, int *Lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)Lwork;
  fprintf(stderr, "acsolverDnZpotrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZpotri(acsolverDnHandle_t handle,
                                                acblasFillMode_t uplo, int n,
                                                acDoubleComplex *A, int lda,
                                                acDoubleComplex *work,
                                                int lwork, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZpotri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZpotri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acDoubleComplex *A, int lda, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnZpotri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZpotrs(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 int nrhs, const acDoubleComplex *A, int lda,
                 acDoubleComplex *B, int ldb, int *devInfo) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZpotrs is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZpotrsBatched(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                        int nrhs, acDoubleComplex *A[], int lda,
                        acDoubleComplex *B[], int ldb, int *d_info,
                        int batchSize) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)A;
  (void)lda;
  (void)B;
  (void)ldb;
  (void)d_info;
  (void)batchSize;
  fprintf(stderr, "acsolverDnZpotrsBatched is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZsytrf(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 acDoubleComplex *A, int lda, int *ipiv, acDoubleComplex *work,
                 int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZsytrf is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZsytrf_bufferSize(acsolverDnHandle_t handle, int n,
                            acDoubleComplex *A, int lda, int *lwork) {
  (void)handle;
  (void)n;
  (void)A;
  (void)lda;
  (void)lwork;
  fprintf(stderr, "acsolverDnZsytrf_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZsytri(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 acDoubleComplex *A, int lda, const int *ipiv,
                 acDoubleComplex *work, int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZsytri is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZsytri_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, acDoubleComplex *A, int lda, const int *ipiv,
                            int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)ipiv;
  (void)lwork;
  fprintf(stderr, "acsolverDnZsytri_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZungbr(acsolverDnHandle_t handle, acblasSideMode_t side, int m, int n,
                 int k, acDoubleComplex *A, int lda, const acDoubleComplex *tau,
                 acDoubleComplex *work, int lwork, int *info) {
  (void)handle;
  (void)side;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZungbr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZungbr_bufferSize(acsolverDnHandle_t handle, acblasSideMode_t side,
                            int m, int n, int k, const acDoubleComplex *A,
                            int lda, const acDoubleComplex *tau, int *lwork) {
  (void)handle;
  (void)side;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnZungbr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZungqr(acsolverDnHandle_t handle, int m, int n, int k,
                 acDoubleComplex *A, int lda, const acDoubleComplex *tau,
                 acDoubleComplex *work, int lwork, int *info) {
  (void)handle;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZungqr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZungqr_bufferSize(acsolverDnHandle_t handle, int m, int n, int k,
                            const acDoubleComplex *A, int lda,
                            const acDoubleComplex *tau, int *lwork) {
  (void)handle;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnZungqr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZungtr(acsolverDnHandle_t handle, acblasFillMode_t uplo, int n,
                 acDoubleComplex *A, int lda, const acDoubleComplex *tau,
                 acDoubleComplex *work, int lwork, int *info) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZungtr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZungtr_bufferSize(acsolverDnHandle_t handle, acblasFillMode_t uplo,
                            int n, const acDoubleComplex *A, int lda,
                            const acDoubleComplex *tau, int *lwork) {
  (void)handle;
  (void)uplo;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)lwork;
  fprintf(stderr, "acsolverDnZungtr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZunmqr(acsolverDnHandle_t handle, acblasSideMode_t side,
                 acblasOperation_t trans, int m, int n, int k,
                 const acDoubleComplex *A, int lda, const acDoubleComplex *tau,
                 acDoubleComplex *C, int ldc, acDoubleComplex *work, int lwork,
                 int *devInfo) {
  (void)handle;
  (void)side;
  (void)trans;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)work;
  (void)lwork;
  (void)devInfo;
  fprintf(stderr, "acsolverDnZunmqr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZunmqr_bufferSize(
    acsolverDnHandle_t handle, acblasSideMode_t side, acblasOperation_t trans,
    int m, int n, int k, const acDoubleComplex *A, int lda,
    const acDoubleComplex *tau, const acDoubleComplex *C, int ldc, int *lwork) {
  (void)handle;
  (void)side;
  (void)trans;
  (void)m;
  (void)n;
  (void)k;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)lwork;
  fprintf(stderr, "acsolverDnZunmqr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t
acsolverDnZunmtr(acsolverDnHandle_t handle, acblasSideMode_t side,
                 acblasFillMode_t uplo, acblasOperation_t trans, int m, int n,
                 acDoubleComplex *A, int lda, acDoubleComplex *tau,
                 acDoubleComplex *C, int ldc, acDoubleComplex *work, int lwork,
                 int *info) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)work;
  (void)lwork;
  (void)info;
  fprintf(stderr, "acsolverDnZunmtr is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

static inline acsolverStatus_t acsolverDnZunmtr_bufferSize(
    acsolverDnHandle_t handle, acblasSideMode_t side, acblasFillMode_t uplo,
    acblasOperation_t trans, int m, int n, const acDoubleComplex *A, int lda,
    const acDoubleComplex *tau, const acDoubleComplex *C, int ldc, int *lwork) {
  (void)handle;
  (void)side;
  (void)uplo;
  (void)trans;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)tau;
  (void)C;
  (void)ldc;
  (void)lwork;
  fprintf(stderr, "acsolverDnZunmtr_bufferSize is not supported.\n");
  exit(1);
  return (acsolverStatus_t)0; /* unreachable */
}

/* ── cusparse: 293 unsupported APIs ── */

static inline acsparseStatus_t acsparseAxpby(acsparseHandle_t handle,
                                             const void *alpha,
                                             acsparseConstSpVecDescr_t vecX,
                                             const void *beta,
                                             acsparseDnVecDescr_t vecY) {
  (void)handle;
  (void)alpha;
  (void)vecX;
  (void)beta;
  (void)vecY;
  fprintf(stderr, "acsparseAxpby is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseBsrSetStridedBatch(
    acsparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride,
    int64_t columnsBatchStride, int64_t ValuesBatchStride) {
  (void)spMatDescr;
  (void)batchCount;
  (void)offsetsBatchStride;
  (void)columnsBatchStride;
  (void)ValuesBatchStride;
  fprintf(stderr, "acsparseBsrSetStridedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCbsr2csr(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                 int nb, const acsparseMatDescr_t descrA,
                 const acComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA, int blockDim,
                 const acsparseMatDescr_t descrC, acComplex *csrSortedValC,
                 int *csrSortedRowPtrC, int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseCbsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCbsric02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                 int nnzb, const acsparseMatDescr_t descrA,
                 acComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                 acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCbsric02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsric02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, acsparseSolvePolicy_t policy, void *pInputBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pInputBuffer;
  fprintf(stderr, "acsparseCbsric02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCbsric02_bufferSize(acsparseHandle_t handle, acsparseDirection_t dirA,
                            int mb, int nnzb, const acsparseMatDescr_t descrA,
                            acComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd, int blockDim,
                            bsric02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCbsric02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsric02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCbsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCbsrilu02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                  int nnzb, const acsparseMatDescr_t descrA,
                  acComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCbsrilu02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrilu02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCbsrilu02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrilu02_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCbsrilu02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrilu02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCbsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCbsrilu02_numericBoost(acsparseHandle_t handle, bsrilu02Info_t info,
                               int enable_boost, double *tol,
                               acComplex *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseCbsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsm2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, const acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCbsrsm2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsm2_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCbsrsm2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsm2_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transB, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, acComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transB;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCbsrsm2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsm2_solve(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acComplex *alpha, const acsparseMatDescr_t descrA,
    const acComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const acComplex *B, int ldb, acComplex *X, int ldx,
    acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)B;
  (void)ldb;
  (void)X;
  (void)ldx;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCbsrsm2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsv2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCbsrsv2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsv2_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCbsrsv2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsv2_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
    bsrsv2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCbsrsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCbsrsv2_solve(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb, const acComplex *alpha,
    const acsparseMatDescr_t descrA, const acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const acComplex *f, acComplex *x,
    acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)f;
  (void)x;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCbsrsv2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCbsrxmv(acsparseHandle_t handle, acsparseDirection_t dirA,
                acsparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const acComplex *alpha,
                const acsparseMatDescr_t descrA, const acComplex *bsrSortedValA,
                const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA,
                const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
                int blockDim, const acComplex *x, const acComplex *beta,
                acComplex *y) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)sizeOfMask;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedMaskPtrA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedEndPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)x;
  (void)beta;
  (void)y;
  fprintf(stderr, "acsparseCbsrxmv is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsr2bsr(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                 int n, const acsparseMatDescr_t descrA,
                 const acComplex *csrSortedValA, const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA, int blockDim,
                 const acsparseMatDescr_t descrC, acComplex *bsrSortedValC,
                 int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  fprintf(stderr, "acsparseCcsr2bsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsr2csr_compress(
    acsparseHandle_t handle, int m, int n, const acsparseMatDescr_t descrA,
    const acComplex *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    acComplex *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    acComplex tol) {
  (void)handle;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedColIndA;
  (void)csrSortedRowPtrA;
  (void)nnzA;
  (void)nnzPerRow;
  (void)csrSortedValC;
  (void)csrSortedColIndC;
  (void)csrSortedRowPtrC;
  (void)tol;
  fprintf(stderr, "acsparseCcsr2csr_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsr2csru(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, acComplex *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseCcsr2csru is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsr2gebsr(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                   int n, const acsparseMatDescr_t descrA,
                   const acComplex *csrSortedValA, const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA, const acsparseMatDescr_t descrC,
                   acComplex *bsrSortedValC, int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC, int rowBlockDim, int colBlockDim,
                   void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBuffer;
  fprintf(stderr, "acsparseCcsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const acComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCcsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const acComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCcsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsrcolor(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    const acComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const float *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const acsparseColorInfo_t info) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)fractionToColor;
  (void)ncolors;
  (void)coloring;
  (void)reordering;
  (void)info;
  fprintf(stderr, "acsparseCcsrcolor is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsrgeam2(acsparseHandle_t handle, int m, int n, const acComplex *alpha,
                  const acsparseMatDescr_t descrA, int nnzA,
                  const acComplex *csrSortedValA, const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA, const acComplex *beta,
                  const acsparseMatDescr_t descrB, int nnzB,
                  const acComplex *csrSortedValB, const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB, const acsparseMatDescr_t descrC,
                  acComplex *csrSortedValC, int *csrSortedRowPtrC,
                  int *csrSortedColIndC, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseCcsrgeam2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsrgeam2_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const acComplex *alpha,
    const acsparseMatDescr_t descrA, int nnzA, const acComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const acComplex *beta, const acsparseMatDescr_t descrB, int nnzB,
    const acComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const acsparseMatDescr_t descrC,
    const acComplex *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCcsrgeam2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsric02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    acComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCcsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsrilu02(acsparseHandle_t handle, int m, int nnz,
                  const acsparseMatDescr_t descrA,
                  acComplex *csrSortedValA_valM, const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA, csrilu02Info_t info,
                  acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedValA_valM;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCcsrilu02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsrilu02_analysis(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    const acComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info,
    acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseCcsrilu02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsrilu02_bufferSize(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    acComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCcsrilu02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCcsrilu02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    acComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCcsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsrilu02_numericBoost(acsparseHandle_t handle, csrilu02Info_t info,
                               int enable_boost, double *tol,
                               acComplex *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseCcsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsru2csr(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, acComplex *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseCcsru2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCcsru2csr_bufferSizeExt(acsparseHandle_t handle, int m, int n, int nnz,
                                acComplex *csrVal, const int *csrRowPtr,
                                int *csrColInd, csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCcsru2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgebsr2csr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb,
    const acsparseMatDescr_t descrA, const acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const acsparseMatDescr_t descrC, acComplex *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseCgebsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgebsr2gebsc_bufferSizeExt(acsparseHandle_t handle, int mb, int nb,
                                   int nnzb, const acComplex *bsrSortedVal,
                                   const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd, int rowBlockDim,
                                   int colBlockDim, size_t *pBufferSize) {
  (void)handle;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCgebsr2gebsc_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgebsr2gebsr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const acsparseMatDescr_t descrC, acComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBuffer;
  fprintf(stderr, "acsparseCgebsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgebsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseCgebsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgebsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const acComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCgebsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgemvi(acsparseHandle_t handle, acsparseOperation_t transA, int m,
               int n, const acComplex *alpha, const acComplex *A, int lda,
               int nnz, const acComplex *xVal, const int *xInd,
               const acComplex *beta, acComplex *y, acsparseIndexBase_t idxBase,
               void *pBuffer) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)nnz;
  (void)xVal;
  (void)xInd;
  (void)beta;
  (void)y;
  (void)idxBase;
  (void)pBuffer;
  fprintf(stderr, "acsparseCgemvi is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgemvi_bufferSize(acsparseHandle_t handle, acsparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)nnz;
  (void)pBufferSize;
  fprintf(stderr, "acsparseCgemvi_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgpsvInterleavedBatch(acsparseHandle_t handle, int algo, int m,
                              acComplex *ds, acComplex *dl, acComplex *d,
                              acComplex *du, acComplex *dw, acComplex *x,
                              int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)ds;
  (void)dl;
  (void)d;
  (void)du;
  (void)dw;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseCgpsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgpsvInterleavedBatch_bufferSizeExt(
    acsparseHandle_t handle, int algo, int m, const acComplex *ds,
    const acComplex *dl, const acComplex *d, const acComplex *du,
    const acComplex *dw, const acComplex *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)ds;
  (void)dl;
  (void)d;
  (void)du;
  (void)dw;
  (void)x;
  (void)batchCount;
  (void)pBufferSizeInBytes;
  fprintf(stderr,
          "acsparseCgpsvInterleavedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgtsv2(acsparseHandle_t handle, int m,
                                              int n, const acComplex *dl,
                                              const acComplex *d,
                                              const acComplex *du, acComplex *B,
                                              int ldb, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseCgtsv2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgtsv2StridedBatch(acsparseHandle_t handle, int m, const acComplex *dl,
                           const acComplex *d, const acComplex *du,
                           acComplex *x, int batchCount, int batchStride,
                           void *pBuffer) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)pBuffer;
  fprintf(stderr, "acsparseCgtsv2StridedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgtsv2StridedBatch_bufferSizeExt(
    acsparseHandle_t handle, int m, const acComplex *dl, const acComplex *d,
    const acComplex *du, const acComplex *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)bufferSizeInBytes;
  fprintf(stderr,
          "acsparseCgtsv2StridedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgtsv2_bufferSizeExt(acsparseHandle_t handle, int m, int n,
                             const acComplex *dl, const acComplex *d,
                             const acComplex *du, const acComplex *B, int ldb,
                             size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseCgtsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgtsv2_nopivot(acsparseHandle_t handle, int m, int n,
                       const acComplex *dl, const acComplex *d,
                       const acComplex *du, acComplex *B, int ldb,
                       void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseCgtsv2_nopivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgtsv2_nopivot_bufferSizeExt(acsparseHandle_t handle, int m, int n,
                                     const acComplex *dl, const acComplex *d,
                                     const acComplex *du, const acComplex *B,
                                     int ldb, size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseCgtsv2_nopivot_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCgtsvInterleavedBatch(acsparseHandle_t handle, int algo, int m,
                              acComplex *dl, acComplex *d, acComplex *du,
                              acComplex *x, int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseCgtsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCgtsvInterleavedBatch_bufferSizeExt(
    acsparseHandle_t handle, int algo, int m, const acComplex *dl,
    const acComplex *d, const acComplex *du, const acComplex *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBufferSizeInBytes;
  fprintf(stderr,
          "acsparseCgtsvInterleavedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCnnz(acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
             const acsparseMatDescr_t descrA, const acComplex *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)A;
  (void)lda;
  (void)nnzPerRowCol;
  (void)nnzTotalDevHostPtr;
  fprintf(stderr, "acsparseCnnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCnnz_compress(
    acsparseHandle_t handle, int m, const acsparseMatDescr_t descr,
    const acComplex *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow,
    int *nnzC, acComplex tol) {
  (void)handle;
  (void)m;
  (void)descr;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)nnzPerRow;
  (void)nnzC;
  (void)tol;
  fprintf(stderr, "acsparseCnnz_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCooSetPointers(acsparseSpMatDescr_t spMatDescr, void *cooRows,
                       void *cooColumns, void *cooValues) {
  (void)spMatDescr;
  (void)cooRows;
  (void)cooColumns;
  (void)cooValues;
  fprintf(stderr, "acsparseCooSetPointers is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCreateBsr(acsparseSpMatDescr_t *spMatDescr, int64_t brows,
                  int64_t bcols, int64_t bnnz, int64_t rowBlockSize,
                  int64_t colBlockSize, void *bsrRowOffsets, void *bsrColInd,
                  void *bsrValues, acsparseIndexType_t bsrRowOffsetsType,
                  acsparseIndexType_t bsrColIndType,
                  acsparseIndexBase_t idxBase, hggcDataType valueType,
                  acsparseOrder_t order) {
  (void)spMatDescr;
  (void)brows;
  (void)bcols;
  (void)bnnz;
  (void)rowBlockSize;
  (void)colBlockSize;
  (void)bsrRowOffsets;
  (void)bsrColInd;
  (void)bsrValues;
  (void)bsrRowOffsetsType;
  (void)bsrColIndType;
  (void)idxBase;
  (void)valueType;
  (void)order;
  fprintf(stderr, "acsparseCreateBsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCreateConstBsr(
    acsparseConstSpMatDescr_t *spMatDescr, int64_t brows, int64_t bcols,
    int64_t bnnz, int64_t rowBlockDim, int64_t colBlockDim,
    const void *bsrRowOffsets, const void *bsrColInd, const void *bsrValues,
    acsparseIndexType_t bsrRowOffsetsType, acsparseIndexType_t bsrColIndType,
    acsparseIndexBase_t idxBase, hggcDataType valueType,
    acsparseOrder_t order) {
  (void)spMatDescr;
  (void)brows;
  (void)bcols;
  (void)bnnz;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)bsrRowOffsets;
  (void)bsrColInd;
  (void)bsrValues;
  (void)bsrRowOffsetsType;
  (void)bsrColIndType;
  (void)idxBase;
  (void)valueType;
  (void)order;
  fprintf(stderr, "acsparseCreateConstBsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseCreateConstSlicedEll(
    acsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
    int64_t nnz, int64_t sellValuesSize, int64_t sliceSize,
    const void *sellSliceOffsets, const void *sellColInd,
    const void *sellValues, acsparseIndexType_t sellSliceOffsetsType,
    acsparseIndexType_t sellColIndType, acsparseIndexBase_t idxBase,
    hggcDataType valueType) {
  (void)spMatDescr;
  (void)rows;
  (void)cols;
  (void)nnz;
  (void)sellValuesSize;
  (void)sliceSize;
  (void)sellSliceOffsets;
  (void)sellColInd;
  (void)sellValues;
  (void)sellSliceOffsetsType;
  (void)sellColIndType;
  (void)idxBase;
  (void)valueType;
  fprintf(stderr, "acsparseCreateConstSlicedEll is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCreateCsru2csrInfo(csru2csrInfo_t *info) {
  (void)info;
  fprintf(stderr, "acsparseCreateCsru2csrInfo is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCreateSlicedEll(acsparseSpMatDescr_t *spMatDescr, int64_t rows,
                        int64_t cols, int64_t nnz, int64_t sellValuesSize,
                        int64_t sliceSize, void *sellSliceOffsets,
                        void *sellColInd, void *sellValues,
                        acsparseIndexType_t sellSliceOffsetsType,
                        acsparseIndexType_t sellColIndType,
                        acsparseIndexBase_t idxBase, hggcDataType valueType) {
  (void)spMatDescr;
  (void)rows;
  (void)cols;
  (void)nnz;
  (void)sellValuesSize;
  (void)sliceSize;
  (void)sellSliceOffsets;
  (void)sellColInd;
  (void)sellValues;
  (void)sellSliceOffsetsType;
  (void)sellColIndType;
  (void)idxBase;
  (void)valueType;
  fprintf(stderr, "acsparseCreateSlicedEll is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseCscSetPointers(acsparseSpMatDescr_t spMatDescr, void *cscColOffsets,
                       void *cscRowInd, void *cscValues) {
  (void)spMatDescr;
  (void)cscColOffsets;
  (void)cscRowInd;
  (void)cscValues;
  fprintf(stderr, "acsparseCscSetPointers is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsr2csr(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                 int nb, const acsparseMatDescr_t descrA,
                 const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA, int blockDim,
                 const acsparseMatDescr_t descrC, double *csrSortedValC,
                 int *csrSortedRowPtrC, int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseDbsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsric02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                 int nnzb, const acsparseMatDescr_t descrA,
                 double *bsrSortedVal, const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                 acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseDbsric02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsric02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, acsparseSolvePolicy_t policy, void *pInputBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pInputBuffer;
  fprintf(stderr, "acsparseDbsric02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsric02_bufferSize(acsparseHandle_t handle, acsparseDirection_t dirA,
                            int mb, int nnzb, const acsparseMatDescr_t descrA,
                            double *bsrSortedVal, const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd, int blockDim,
                            bsric02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDbsric02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsric02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDbsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsrilu02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                  int nnzb, const acsparseMatDescr_t descrA,
                  double *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseDbsrilu02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsrilu02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseDbsrilu02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsrilu02_bufferSize(acsparseHandle_t handle, acsparseDirection_t dirA,
                             int mb, int nnzb, const acsparseMatDescr_t descrA,
                             double *bsrSortedVal, const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd, int blockDim,
                             bsrilu02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDbsrilu02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsrilu02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDbsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsrilu02_numericBoost(acsparseHandle_t handle, bsrilu02Info_t info,
                               int enable_boost, double *tol,
                               double *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseDbsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsrsm2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, const double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseDbsrsm2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsrsm2_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDbsrsm2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsrsm2_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transB, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transB;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDbsrsm2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsrsm2_solve(acsparseHandle_t handle, acsparseDirection_t dirA,
                      acsparseOperation_t transA, acsparseOperation_t transXY,
                      int mb, int n, int nnzb, const double *alpha,
                      const acsparseMatDescr_t descrA,
                      const double *bsrSortedVal, const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd, int blockSize,
                      bsrsm2Info_t info, const double *B, int ldb, double *X,
                      int ldx, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)B;
  (void)ldb;
  (void)X;
  (void)ldx;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseDbsrsm2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsrsv2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseDbsrsv2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsrsv2_bufferSize(acsparseHandle_t handle, acsparseDirection_t dirA,
                           acsparseOperation_t transA, int mb, int nnzb,
                           const acsparseMatDescr_t descrA,
                           double *bsrSortedValA, const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA, int blockDim,
                           bsrsv2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDbsrsv2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDbsrsv2_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
    bsrsv2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDbsrsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsrsv2_solve(acsparseHandle_t handle, acsparseDirection_t dirA,
                      acsparseOperation_t transA, int mb, int nnzb,
                      const double *alpha, const acsparseMatDescr_t descrA,
                      const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA, int blockDim,
                      bsrsv2Info_t info, const double *f, double *x,
                      acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)f;
  (void)x;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseDbsrsv2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDbsrxmv(acsparseHandle_t handle, acsparseDirection_t dirA,
                acsparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const double *alpha, const acsparseMatDescr_t descrA,
                const double *bsrSortedValA, const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA, int blockDim, const double *x,
                const double *beta, double *y) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)sizeOfMask;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedMaskPtrA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedEndPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)x;
  (void)beta;
  (void)y;
  fprintf(stderr, "acsparseDbsrxmv is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsr2bsr(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                 int n, const acsparseMatDescr_t descrA,
                 const double *csrSortedValA, const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA, int blockDim,
                 const acsparseMatDescr_t descrC, double *bsrSortedValC,
                 int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  fprintf(stderr, "acsparseDcsr2bsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDcsr2csr_compress(
    acsparseHandle_t handle, int m, int n, const acsparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    double *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    double tol) {
  (void)handle;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedColIndA;
  (void)csrSortedRowPtrA;
  (void)nnzA;
  (void)nnzPerRow;
  (void)csrSortedValC;
  (void)csrSortedColIndC;
  (void)csrSortedRowPtrC;
  (void)tol;
  fprintf(stderr, "acsparseDcsr2csr_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsr2csru(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, double *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseDcsr2csru is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsr2gebsr(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                   int n, const acsparseMatDescr_t descrA,
                   const double *csrSortedValA, const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA, const acsparseMatDescr_t descrC,
                   double *bsrSortedValC, int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC, int rowBlockDim, int colBlockDim,
                   void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBuffer;
  fprintf(stderr, "acsparseDcsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDcsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDcsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDcsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDcsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsrcolor(acsparseHandle_t handle, int m, int nnz,
                  const acsparseMatDescr_t descrA, const double *csrSortedValA,
                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                  const double *fractionToColor, int *ncolors, int *coloring,
                  int *reordering, const acsparseColorInfo_t info) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)fractionToColor;
  (void)ncolors;
  (void)coloring;
  (void)reordering;
  (void)info;
  fprintf(stderr, "acsparseDcsrcolor is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsrgeam2(acsparseHandle_t handle, int m, int n, const double *alpha,
                  const acsparseMatDescr_t descrA, int nnzA,
                  const double *csrSortedValA, const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA, const double *beta,
                  const acsparseMatDescr_t descrB, int nnzB,
                  const double *csrSortedValB, const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB, const acsparseMatDescr_t descrC,
                  double *csrSortedValC, int *csrSortedRowPtrC,
                  int *csrSortedColIndC, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseDcsrgeam2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDcsrgeam2_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const double *alpha,
    const acsparseMatDescr_t descrA, int nnzA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *beta, const acsparseMatDescr_t descrB, int nnzB,
    const double *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const acsparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDcsrgeam2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDcsric02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    double *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDcsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDcsrilu02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    double *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDcsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsrilu02_numericBoost(acsparseHandle_t handle, csrilu02Info_t info,
                               int enable_boost, double *tol,
                               double *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseDcsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsru2csr(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, double *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseDcsru2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDcsru2csr_bufferSizeExt(acsparseHandle_t handle, int m, int n, int nnz,
                                double *csrVal, const int *csrRowPtr,
                                int *csrColInd, csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDcsru2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDenseToSparse_analysis(
    acsparseHandle_t handle, acsparseConstDnMatDescr_t matA,
    acsparseSpMatDescr_t matB, acsparseDenseToSparseAlg_t alg,
    void *externalBuffer) {
  (void)handle;
  (void)matA;
  (void)matB;
  (void)alg;
  (void)externalBuffer;
  fprintf(stderr, "acsparseDenseToSparse_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDenseToSparse_convert(
    acsparseHandle_t handle, acsparseConstDnMatDescr_t matA,
    acsparseSpMatDescr_t matB, acsparseDenseToSparseAlg_t alg,
    void *externalBuffer) {
  (void)handle;
  (void)matA;
  (void)matB;
  (void)alg;
  (void)externalBuffer;
  fprintf(stderr, "acsparseDenseToSparse_convert is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDestroyCsru2csrInfo(csru2csrInfo_t info) {
  (void)info;
  fprintf(stderr, "acsparseDestroyCsru2csrInfo is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgebsr2csr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb,
    const acsparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const acsparseMatDescr_t descrC, double *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseDgebsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDgebsr2gebsc_bufferSizeExt(acsparseHandle_t handle, int mb, int nb,
                                   int nnzb, const double *bsrSortedVal,
                                   const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd, int rowBlockDim,
                                   int colBlockDim, size_t *pBufferSize) {
  (void)handle;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDgebsr2gebsc_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgebsr2gebsr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const acsparseMatDescr_t descrC, double *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBuffer;
  fprintf(stderr, "acsparseDgebsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgebsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDgebsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgebsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDgebsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDgemvi(acsparseHandle_t handle, acsparseOperation_t transA, int m,
               int n, const double *alpha, const double *A, int lda, int nnz,
               const double *xVal, const int *xInd, const double *beta,
               double *y, acsparseIndexBase_t idxBase, void *pBuffer) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)nnz;
  (void)xVal;
  (void)xInd;
  (void)beta;
  (void)y;
  (void)idxBase;
  (void)pBuffer;
  fprintf(stderr, "acsparseDgemvi is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDgemvi_bufferSize(acsparseHandle_t handle, acsparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)nnz;
  (void)pBufferSize;
  fprintf(stderr, "acsparseDgemvi_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgpsvInterleavedBatch(
    acsparseHandle_t handle, int algo, int m, double *ds, double *dl, double *d,
    double *du, double *dw, double *x, int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)ds;
  (void)dl;
  (void)d;
  (void)du;
  (void)dw;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseDgpsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgpsvInterleavedBatch_bufferSizeExt(
    acsparseHandle_t handle, int algo, int m, const double *ds,
    const double *dl, const double *d, const double *du, const double *dw,
    const double *x, int batchCount, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)ds;
  (void)dl;
  (void)d;
  (void)du;
  (void)dw;
  (void)x;
  (void)batchCount;
  (void)pBufferSizeInBytes;
  fprintf(stderr,
          "acsparseDgpsvInterleavedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgtsv2(acsparseHandle_t handle, int m,
                                              int n, const double *dl,
                                              const double *d, const double *du,
                                              double *B, int ldb,
                                              void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseDgtsv2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDgtsv2StridedBatch(acsparseHandle_t handle, int m, const double *dl,
                           const double *d, const double *du, double *x,
                           int batchCount, int batchStride, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)pBuffer;
  fprintf(stderr, "acsparseDgtsv2StridedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgtsv2StridedBatch_bufferSizeExt(
    acsparseHandle_t handle, int m, const double *dl, const double *d,
    const double *du, const double *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)bufferSizeInBytes;
  fprintf(stderr,
          "acsparseDgtsv2StridedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgtsv2_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const double *dl, const double *d,
    const double *du, const double *B, int ldb, size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseDgtsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDgtsv2_nopivot(acsparseHandle_t handle, int m, int n, const double *dl,
                       const double *d, const double *du, double *B, int ldb,
                       void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseDgtsv2_nopivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgtsv2_nopivot_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const double *dl, const double *d,
    const double *du, const double *B, int ldb, size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseDgtsv2_nopivot_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDgtsvInterleavedBatch(acsparseHandle_t handle, int algo, int m,
                              double *dl, double *d, double *du, double *x,
                              int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseDgtsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDgtsvInterleavedBatch_bufferSizeExt(
    acsparseHandle_t handle, int algo, int m, const double *dl, const double *d,
    const double *du, const double *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBufferSizeInBytes;
  fprintf(stderr,
          "acsparseDgtsvInterleavedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDnnz(acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
             const acsparseMatDescr_t descrA, const double *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)A;
  (void)lda;
  (void)nnzPerRowCol;
  (void)nnzTotalDevHostPtr;
  fprintf(stderr, "acsparseDnnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDnnz_compress(acsparseHandle_t handle, int m,
                      const acsparseMatDescr_t descr,
                      const double *csrSortedValA, const int *csrSortedRowPtrA,
                      int *nnzPerRow, int *nnzC, double tol) {
  (void)handle;
  (void)m;
  (void)descr;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)nnzPerRow;
  (void)nnzC;
  (void)tol;
  fprintf(stderr, "acsparseDnnz_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDpruneCsr2csr(acsparseHandle_t handle, int m, int n, int nnzA,
                      const acsparseMatDescr_t descrA,
                      const double *csrSortedValA, const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA, const double *threshold,
                      const acsparseMatDescr_t descrC, double *csrSortedValC,
                      const int *csrSortedRowPtrC, int *csrSortedColIndC,
                      void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneCsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneCsr2csrByPercentage(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, double *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneCsr2csrByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneCsr2csrByPercentage_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, const double *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(
      stderr,
      "acsparseDpruneCsr2csrByPercentage_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneCsr2csrNnz(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *threshold, const acsparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneCsr2csrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneCsr2csrNnzByPercentage(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneCsr2csrNnzByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneCsr2csr_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const double *threshold, const acsparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDpruneCsr2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDpruneDense2csr(acsparseHandle_t handle, int m, int n, const double *A,
                        int lda, const double *threshold,
                        const acsparseMatDescr_t descrC, double *csrSortedValC,
                        const int *csrSortedRowPtrC, int *csrSortedColIndC,
                        void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneDense2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneDense2csrByPercentage(
    acsparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const acsparseMatDescr_t descrC, double *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneDense2csrByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseDpruneDense2csrByPercentage_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const acsparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(
      stderr,
      "acsparseDpruneDense2csrByPercentage_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneDense2csrNnz(
    acsparseHandle_t handle, int m, int n, const double *A, int lda,
    const double *threshold, const acsparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneDense2csrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneDense2csrNnzByPercentage(
    acsparseHandle_t handle, int m, int n, const double *A, int lda,
    float percentage, const acsparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseDpruneDense2csrNnzByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseDpruneDense2csr_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const double *A, int lda,
    const double *threshold, const acsparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseDpruneDense2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseGetProperty(hggcLibraryPropertyType type,
                                                   int *value) {
  (void)type;
  (void)value;
  fprintf(stderr, "acsparseGetProperty is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseHpruneCsr2csr(acsparseHandle_t handle, int m, int n, int nnzA,
                      const acsparseMatDescr_t descrA,
                      const __half *csrSortedValA, const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA, const __half *threshold,
                      const acsparseMatDescr_t descrC, __half *csrSortedValC,
                      const int *csrSortedRowPtrC, int *csrSortedColIndC,
                      void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneCsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneCsr2csrByPercentage(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, __half *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneCsr2csrByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneCsr2csrByPercentage_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, const __half *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(
      stderr,
      "acsparseHpruneCsr2csrByPercentage_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneCsr2csrNnz(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const __half *threshold, const acsparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneCsr2csrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneCsr2csrNnzByPercentage(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneCsr2csrNnzByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneCsr2csr_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const __half *threshold, const acsparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseHpruneCsr2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseHpruneDense2csr(acsparseHandle_t handle, int m, int n, const __half *A,
                        int lda, const __half *threshold,
                        const acsparseMatDescr_t descrC, __half *csrSortedValC,
                        const int *csrSortedRowPtrC, int *csrSortedColIndC,
                        void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneDense2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneDense2csrByPercentage(
    acsparseHandle_t handle, int m, int n, const __half *A, int lda,
    float percentage, const acsparseMatDescr_t descrC, __half *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneDense2csrByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseHpruneDense2csrByPercentage_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const __half *A, int lda,
    float percentage, const acsparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(
      stderr,
      "acsparseHpruneDense2csrByPercentage_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseHpruneDense2csrNnz(acsparseHandle_t handle, int m, int n,
                           const __half *A, int lda, const __half *threshold,
                           const acsparseMatDescr_t descrC, int *csrRowPtrC,
                           int *nnzTotalDevHostPtr, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneDense2csrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneDense2csrNnzByPercentage(
    acsparseHandle_t handle, int m, int n, const __half *A, int lda,
    float percentage, const acsparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseHpruneDense2csrNnzByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseHpruneDense2csr_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const __half *A, int lda,
    const __half *threshold, const acsparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseHpruneDense2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseLoggerForceDisable(void) {
  fprintf(stderr, "acsparseLoggerForceDisable is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseLoggerOpenFile(const char *logFile) {
  (void)logFile;
  fprintf(stderr, "acsparseLoggerOpenFile is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseLoggerSetCallback(acsparseLoggerCallback_t callback) {
  (void)callback;
  fprintf(stderr, "acsparseLoggerSetCallback is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseLoggerSetFile(FILE *file) {
  (void)file;
  fprintf(stderr, "acsparseLoggerSetFile is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseLoggerSetLevel(int level) {
  (void)level;
  fprintf(stderr, "acsparseLoggerSetLevel is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseLoggerSetMask(int mask) {
  (void)mask;
  fprintf(stderr, "acsparseLoggerSetMask is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSDDMM_preprocess(
    acsparseHandle_t handle, acsparseOperation_t opA, acsparseOperation_t opB,
    const void *alpha, acsparseConstDnMatDescr_t matA,
    acsparseConstDnMatDescr_t matB, const void *beta, acsparseSpMatDescr_t matC,
    hggcDataType computeType, acsparseSDDMMAlg_t alg, void *externalBuffer) {
  (void)handle;
  (void)opA;
  (void)opB;
  (void)alpha;
  (void)matA;
  (void)matB;
  (void)beta;
  (void)matC;
  (void)computeType;
  (void)alg;
  (void)externalBuffer;
  fprintf(stderr, "acsparseSDDMM_preprocess is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsr2csr(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                 int nb, const acsparseMatDescr_t descrA,
                 const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA, int blockDim,
                 const acsparseMatDescr_t descrC, float *csrSortedValC,
                 int *csrSortedRowPtrC, int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseSbsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsric02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                 int nnzb, const acsparseMatDescr_t descrA, float *bsrSortedVal,
                 const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                 int blockDim, bsric02Info_t info, acsparseSolvePolicy_t policy,
                 void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseSbsric02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsric02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, acsparseSolvePolicy_t policy, void *pInputBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pInputBuffer;
  fprintf(stderr, "acsparseSbsric02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsric02_bufferSize(acsparseHandle_t handle, acsparseDirection_t dirA,
                            int mb, int nnzb, const acsparseMatDescr_t descrA,
                            float *bsrSortedVal, const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd, int blockDim,
                            bsric02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseSbsric02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsric02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseSbsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrilu02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                  int nnzb, const acsparseMatDescr_t descrA,
                  float *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseSbsrilu02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsrilu02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseSbsrilu02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrilu02_bufferSize(acsparseHandle_t handle, acsparseDirection_t dirA,
                             int mb, int nnzb, const acsparseMatDescr_t descrA,
                             float *bsrSortedVal, const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd, int blockDim,
                             bsrilu02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseSbsrilu02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsrilu02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseSbsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrilu02_numericBoost(acsparseHandle_t handle, bsrilu02Info_t info,
                               int enable_boost, double *tol,
                               float *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseSbsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsrsm2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, const float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseSbsrsm2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsrsm2_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseSbsrsm2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsrsm2_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transB, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transB;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseSbsrsm2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrsm2_solve(acsparseHandle_t handle, acsparseDirection_t dirA,
                      acsparseOperation_t transA, acsparseOperation_t transXY,
                      int mb, int n, int nnzb, const float *alpha,
                      const acsparseMatDescr_t descrA,
                      const float *bsrSortedVal, const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd, int blockSize,
                      bsrsm2Info_t info, const float *B, int ldb, float *X,
                      int ldx, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)B;
  (void)ldb;
  (void)X;
  (void)ldx;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseSbsrsm2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSbsrsv2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseSbsrsv2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrsv2_bufferSize(acsparseHandle_t handle, acsparseDirection_t dirA,
                           acsparseOperation_t transA, int mb, int nnzb,
                           const acsparseMatDescr_t descrA,
                           float *bsrSortedValA, const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA, int blockDim,
                           bsrsv2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseSbsrsv2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrsv2_bufferSizeExt(acsparseHandle_t handle, acsparseDirection_t dirA,
                              acsparseOperation_t transA, int mb, int nnzb,
                              const acsparseMatDescr_t descrA,
                              float *bsrSortedValA, const int *bsrSortedRowPtrA,
                              const int *bsrSortedColIndA, int blockSize,
                              bsrsv2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseSbsrsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrsv2_solve(acsparseHandle_t handle, acsparseDirection_t dirA,
                      acsparseOperation_t transA, int mb, int nnzb,
                      const float *alpha, const acsparseMatDescr_t descrA,
                      const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA, int blockDim,
                      bsrsv2Info_t info, const float *f, float *x,
                      acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)f;
  (void)x;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseSbsrsv2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSbsrxmv(acsparseHandle_t handle, acsparseDirection_t dirA,
                acsparseOperation_t transA, int sizeOfMask, int mb, int nb,
                int nnzb, const float *alpha, const acsparseMatDescr_t descrA,
                const float *bsrSortedValA, const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA, int blockDim, const float *x,
                const float *beta, float *y) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)sizeOfMask;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedMaskPtrA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedEndPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)x;
  (void)beta;
  (void)y;
  fprintf(stderr, "acsparseSbsrxmv is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseScsr2bsr(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                 int n, const acsparseMatDescr_t descrA,
                 const float *csrSortedValA, const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA, int blockDim,
                 const acsparseMatDescr_t descrC, float *bsrSortedValC,
                 int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  fprintf(stderr, "acsparseScsr2bsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseScsr2csr_compress(
    acsparseHandle_t handle, int m, int n, const acsparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    float *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC,
    float tol) {
  (void)handle;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedColIndA;
  (void)csrSortedRowPtrA;
  (void)nnzA;
  (void)nnzPerRow;
  (void)csrSortedValC;
  (void)csrSortedColIndC;
  (void)csrSortedRowPtrC;
  (void)tol;
  fprintf(stderr, "acsparseScsr2csr_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseScsr2csru(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, float *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseScsr2csru is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseScsr2gebsr(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                   int n, const acsparseMatDescr_t descrA,
                   const float *csrSortedValA, const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA, const acsparseMatDescr_t descrC,
                   float *bsrSortedValC, int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC, int rowBlockDim, int colBlockDim,
                   void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBuffer;
  fprintf(stderr, "acsparseScsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseScsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseScsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseScsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseScsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseScsrcolor(acsparseHandle_t handle, int m, int nnz,
                  const acsparseMatDescr_t descrA, const float *csrSortedValA,
                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                  const float *fractionToColor, int *ncolors, int *coloring,
                  int *reordering, const acsparseColorInfo_t info) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)fractionToColor;
  (void)ncolors;
  (void)coloring;
  (void)reordering;
  (void)info;
  fprintf(stderr, "acsparseScsrcolor is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseScsrgeam2(
    acsparseHandle_t handle, int m, int n, const float *alpha,
    const acsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const acsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const acsparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseScsrgeam2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseScsrgeam2_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const float *alpha,
    const acsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const acsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB,
    const acsparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseScsrgeam2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseScsric02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    float *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd,
    csric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseScsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseScsrilu02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    float *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd,
    csrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseScsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseScsrilu02_numericBoost(acsparseHandle_t handle, csrilu02Info_t info,
                               int enable_boost, double *tol,
                               float *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseScsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseScsru2csr(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, float *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseScsru2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseScsru2csr_bufferSizeExt(acsparseHandle_t handle, int m, int n, int nnz,
                                float *csrVal, const int *csrRowPtr,
                                int *csrColInd, csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseScsru2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgebsr2csr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb,
    const acsparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const acsparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseSgebsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSgebsr2gebsc_bufferSizeExt(acsparseHandle_t handle, int mb, int nb,
                                   int nnzb, const float *bsrSortedVal,
                                   const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd, int rowBlockDim,
                                   int colBlockDim, size_t *pBufferSize) {
  (void)handle;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseSgebsr2gebsc_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgebsr2gebsr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const acsparseMatDescr_t descrC, float *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC,
    int colBlockDimC, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBuffer;
  fprintf(stderr, "acsparseSgebsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgebsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseSgebsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgebsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSize;
  fprintf(stderr, "acsparseSgebsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSgemvi(acsparseHandle_t handle, acsparseOperation_t transA, int m,
               int n, const float *alpha, const float *A, int lda, int nnz,
               const float *xVal, const int *xInd, const float *beta, float *y,
               acsparseIndexBase_t idxBase, void *pBuffer) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)nnz;
  (void)xVal;
  (void)xInd;
  (void)beta;
  (void)y;
  (void)idxBase;
  (void)pBuffer;
  fprintf(stderr, "acsparseSgemvi is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSgemvi_bufferSize(acsparseHandle_t handle, acsparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)nnz;
  (void)pBufferSize;
  fprintf(stderr, "acsparseSgemvi_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgpsvInterleavedBatch(
    acsparseHandle_t handle, int algo, int m, float *ds, float *dl, float *d,
    float *du, float *dw, float *x, int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)ds;
  (void)dl;
  (void)d;
  (void)du;
  (void)dw;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseSgpsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgtsv2(acsparseHandle_t handle, int m,
                                              int n, const float *dl,
                                              const float *d, const float *du,
                                              float *B, int ldb,
                                              void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseSgtsv2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSgtsv2StridedBatch(acsparseHandle_t handle, int m, const float *dl,
                           const float *d, const float *du, float *x,
                           int batchCount, int batchStride, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)pBuffer;
  fprintf(stderr, "acsparseSgtsv2StridedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgtsv2StridedBatch_bufferSizeExt(
    acsparseHandle_t handle, int m, const float *dl, const float *d,
    const float *du, const float *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)bufferSizeInBytes;
  fprintf(stderr,
          "acsparseSgtsv2StridedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgtsv2_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const float *dl, const float *d,
    const float *du, const float *B, int ldb, size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseSgtsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSgtsv2_nopivot(acsparseHandle_t handle, int m, int n, const float *dl,
                       const float *d, const float *du, float *B, int ldb,
                       void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseSgtsv2_nopivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgtsv2_nopivot_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const float *dl, const float *d,
    const float *du, const float *B, int ldb, size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseSgtsv2_nopivot_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSgtsvInterleavedBatch(acsparseHandle_t handle, int algo, int m,
                              float *dl, float *d, float *du, float *x,
                              int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseSgtsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSgtsvInterleavedBatch_bufferSizeExt(
    acsparseHandle_t handle, int algo, int m, const float *dl, const float *d,
    const float *du, const float *x, int batchCount,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBufferSizeInBytes;
  fprintf(stderr,
          "acsparseSgtsvInterleavedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSnnz(acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
             const acsparseMatDescr_t descrA, const float *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)A;
  (void)lda;
  (void)nnzPerRowCol;
  (void)nnzTotalDevHostPtr;
  fprintf(stderr, "acsparseSnnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSnnz_compress(acsparseHandle_t handle, int m,
                      const acsparseMatDescr_t descr,
                      const float *csrSortedValA, const int *csrSortedRowPtrA,
                      int *nnzPerRow, int *nnzC, float tol) {
  (void)handle;
  (void)m;
  (void)descr;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)nnzPerRow;
  (void)nnzC;
  (void)tol;
  fprintf(stderr, "acsparseSnnz_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpGEMM_estimateMemory(
    acsparseHandle_t handle, acsparseOperation_t opA, acsparseOperation_t opB,
    const void *alpha, acsparseConstSpMatDescr_t matA,
    acsparseConstSpMatDescr_t matB, const void *beta, acsparseSpMatDescr_t matC,
    hggcDataType computeType, acsparseSpGEMMAlg_t alg,
    acsparseSpGEMMDescr_t spgemmDescr, float chunk_fraction,
    size_t *bufferSize3, void *externalBuffer3, size_t *bufferSize2) {
  (void)handle;
  (void)opA;
  (void)opB;
  (void)alpha;
  (void)matA;
  (void)matB;
  (void)beta;
  (void)matC;
  (void)computeType;
  (void)alg;
  (void)spgemmDescr;
  (void)chunk_fraction;
  (void)bufferSize3;
  (void)externalBuffer3;
  (void)bufferSize2;
  fprintf(stderr, "acsparseSpGEMM_estimateMemory is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpGEMM_getNumProducts(acsparseSpGEMMDescr_t spgemmDescr,
                              int64_t *num_prods) {
  (void)spgemmDescr;
  (void)num_prods;
  fprintf(stderr, "acsparseSpGEMM_getNumProducts is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpGEMMreuse_compute(
    acsparseHandle_t handle, acsparseOperation_t opA, acsparseOperation_t opB,
    const void *alpha, acsparseConstSpMatDescr_t matA,
    acsparseConstSpMatDescr_t matB, const void *beta, acsparseSpMatDescr_t matC,
    hggcDataType computeType, acsparseSpGEMMAlg_t alg,
    acsparseSpGEMMDescr_t spgemmDescr) {
  (void)handle;
  (void)opA;
  (void)opB;
  (void)alpha;
  (void)matA;
  (void)matB;
  (void)beta;
  (void)matC;
  (void)computeType;
  (void)alg;
  (void)spgemmDescr;
  fprintf(stderr, "acsparseSpGEMMreuse_compute is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpGEMMreuse_copy(
    acsparseHandle_t handle, acsparseOperation_t opA, acsparseOperation_t opB,
    acsparseConstSpMatDescr_t matA, acsparseConstSpMatDescr_t matB,
    acsparseSpMatDescr_t matC, acsparseSpGEMMAlg_t alg,
    acsparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize5,
    void *externalBuffer5) {
  (void)handle;
  (void)opA;
  (void)opB;
  (void)matA;
  (void)matB;
  (void)matC;
  (void)alg;
  (void)spgemmDescr;
  (void)bufferSize5;
  (void)externalBuffer5;
  fprintf(stderr, "acsparseSpGEMMreuse_copy is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpGEMMreuse_nnz(
    acsparseHandle_t handle, acsparseOperation_t opA, acsparseOperation_t opB,
    acsparseConstSpMatDescr_t matA, acsparseConstSpMatDescr_t matB,
    acsparseSpMatDescr_t matC, acsparseSpGEMMAlg_t alg,
    acsparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2,
    void *externalBuffer2, size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize4, void *externalBuffer4) {
  (void)handle;
  (void)opA;
  (void)opB;
  (void)matA;
  (void)matB;
  (void)matC;
  (void)alg;
  (void)spgemmDescr;
  (void)bufferSize2;
  (void)externalBuffer2;
  (void)bufferSize3;
  (void)externalBuffer3;
  (void)bufferSize4;
  (void)externalBuffer4;
  fprintf(stderr, "acsparseSpGEMMreuse_nnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpGEMMreuse_workEstimation(
    acsparseHandle_t handle, acsparseOperation_t opA, acsparseOperation_t opB,
    acsparseConstSpMatDescr_t matA, acsparseConstSpMatDescr_t matB,
    acsparseSpMatDescr_t matC, acsparseSpGEMMAlg_t alg,
    acsparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize1,
    void *externalBuffer1) {
  (void)handle;
  (void)opA;
  (void)opB;
  (void)matA;
  (void)matB;
  (void)matC;
  (void)alg;
  (void)spgemmDescr;
  (void)bufferSize1;
  (void)externalBuffer1;
  fprintf(stderr, "acsparseSpGEMMreuse_workEstimation is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpSM_updateMatrix(acsparseHandle_t handle,
                          acsparseSpSMDescr_t spsmDescr, void *newValues,
                          acsparseSpSMUpdate_t updatePart) {
  (void)handle;
  (void)spsmDescr;
  (void)newValues;
  (void)updatePart;
  fprintf(stderr, "acsparseSpSM_updateMatrix is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpSV_analysis(acsparseHandle_t handle, acsparseOperation_t opA,
                      const void *alpha, acsparseConstSpMatDescr_t matA,
                      acsparseConstDnVecDescr_t vecX, acsparseDnVecDescr_t vecY,
                      hggcDataType computeType, acsparseSpSVAlg_t alg,
                      acsparseSpSVDescr_t spsvDescr, void *externalBuffer) {
  (void)handle;
  (void)opA;
  (void)alpha;
  (void)matA;
  (void)vecX;
  (void)vecY;
  (void)computeType;
  (void)alg;
  (void)spsvDescr;
  (void)externalBuffer;
  fprintf(stderr, "acsparseSpSV_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpSV_bufferSize(
    acsparseHandle_t handle, acsparseOperation_t opA, const void *alpha,
    acsparseConstSpMatDescr_t matA, acsparseConstDnVecDescr_t vecX,
    acsparseDnVecDescr_t vecY, hggcDataType computeType, acsparseSpSVAlg_t alg,
    acsparseSpSVDescr_t spsvDescr, size_t *bufferSize) {
  (void)handle;
  (void)opA;
  (void)alpha;
  (void)matA;
  (void)vecX;
  (void)vecY;
  (void)computeType;
  (void)alg;
  (void)spsvDescr;
  (void)bufferSize;
  fprintf(stderr, "acsparseSpSV_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpSV_destroyDescr(acsparseSpSVDescr_t descr) {
  (void)descr;
  fprintf(stderr, "acsparseSpSV_destroyDescr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpSV_solve(acsparseHandle_t handle, acsparseOperation_t opA,
                   const void *alpha, acsparseConstSpMatDescr_t matA,
                   acsparseConstDnVecDescr_t vecX, acsparseDnVecDescr_t vecY,
                   hggcDataType computeType, acsparseSpSVAlg_t alg,
                   acsparseSpSVDescr_t spsvDescr) {
  (void)handle;
  (void)opA;
  (void)alpha;
  (void)matA;
  (void)vecX;
  (void)vecY;
  (void)computeType;
  (void)alg;
  (void)spsvDescr;
  fprintf(stderr, "acsparseSpSV_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpSV_updateMatrix(acsparseHandle_t handle,
                          acsparseSpSVDescr_t spsvDescr, void *newValues,
                          acsparseSpSVUpdate_t updatePart) {
  (void)handle;
  (void)spsvDescr;
  (void)newValues;
  (void)updatePart;
  fprintf(stderr, "acsparseSpSV_updateMatrix is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpVV(acsparseHandle_t handle, acsparseOperation_t opX,
             acsparseConstSpVecDescr_t vecX, acsparseConstDnVecDescr_t vecY,
             void *result, hggcDataType computeType, void *externalBuffer) {
  (void)handle;
  (void)opX;
  (void)vecX;
  (void)vecY;
  (void)result;
  (void)computeType;
  (void)externalBuffer;
  fprintf(stderr, "acsparseSpVV is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpruneCsr2csr(acsparseHandle_t handle, int m, int n, int nnzA,
                      const acsparseMatDescr_t descrA,
                      const float *csrSortedValA, const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA, const float *threshold,
                      const acsparseMatDescr_t descrC, float *csrSortedValC,
                      const int *csrSortedRowPtrC, int *csrSortedColIndC,
                      void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneCsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneCsr2csrByPercentage(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneCsr2csrByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneCsr2csrByPercentage_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(
      stderr,
      "acsparseSpruneCsr2csrByPercentage_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneCsr2csrNnz(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const float *threshold, const acsparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneCsr2csrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneCsr2csrNnzByPercentage(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage,
    const acsparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)percentage;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneCsr2csrNnzByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneCsr2csr_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, int nnzA,
    const acsparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const float *threshold, const acsparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnzA;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseSpruneCsr2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpruneDense2csr(acsparseHandle_t handle, int m, int n, const float *A,
                        int lda, const float *threshold,
                        const acsparseMatDescr_t descrC, float *csrSortedValC,
                        const int *csrSortedRowPtrC, int *csrSortedColIndC,
                        void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneDense2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneDense2csrByPercentage(
    acsparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const acsparseMatDescr_t descrC, float *csrSortedValC,
    const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info,
    void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneDense2csrByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpruneDense2csrByPercentage_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const acsparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(
      stderr,
      "acsparseSpruneDense2csrByPercentage_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseSpruneDense2csrNnz(acsparseHandle_t handle, int m, int n,
                           const float *A, int lda, const float *threshold,
                           const acsparseMatDescr_t descrC, int *csrRowPtrC,
                           int *nnzTotalDevHostPtr, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneDense2csrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneDense2csrNnzByPercentage(
    acsparseHandle_t handle, int m, int n, const float *A, int lda,
    float percentage, const acsparseMatDescr_t descrC, int *csrRowPtrC,
    int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)percentage;
  (void)descrC;
  (void)csrRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseSpruneDense2csrNnzByPercentage is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseSpruneDense2csr_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const float *A, int lda,
    const float *threshold, const acsparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)A;
  (void)lda;
  (void)threshold;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseSpruneDense2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXbsric02_zeroPivot(acsparseHandle_t handle, bsric02Info_t info,
                           int *position) {
  (void)handle;
  (void)info;
  (void)position;
  fprintf(stderr, "acsparseXbsric02_zeroPivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXbsrilu02_zeroPivot(acsparseHandle_t handle, bsrilu02Info_t info,
                            int *position) {
  (void)handle;
  (void)info;
  (void)position;
  fprintf(stderr, "acsparseXbsrilu02_zeroPivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXbsrsm2_zeroPivot(acsparseHandle_t handle, bsrsm2Info_t info,
                          int *position) {
  (void)handle;
  (void)info;
  (void)position;
  fprintf(stderr, "acsparseXbsrsm2_zeroPivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXbsrsv2_zeroPivot(acsparseHandle_t handle, bsrsv2Info_t info,
                          int *position) {
  (void)handle;
  (void)info;
  (void)position;
  fprintf(stderr, "acsparseXbsrsv2_zeroPivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXcoosortByColumn(acsparseHandle_t handle, int m, int n, int nnz,
                         int *cooRowsA, int *cooColsA, int *P, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)cooRowsA;
  (void)cooColsA;
  (void)P;
  (void)pBuffer;
  fprintf(stderr, "acsparseXcoosortByColumn is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXcscsort(acsparseHandle_t handle, int m, int n, int nnz,
                 const acsparseMatDescr_t descrA, const int *cscColPtrA,
                 int *cscRowIndA, int *P, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)cscColPtrA;
  (void)cscRowIndA;
  (void)P;
  (void)pBuffer;
  fprintf(stderr, "acsparseXcscsort is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXcscsort_bufferSizeExt(acsparseHandle_t handle, int m, int n, int nnz,
                               const int *cscColPtrA, const int *cscRowIndA,
                               size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)cscColPtrA;
  (void)cscRowIndA;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseXcscsort_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXcsr2bsrNnz(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                    int n, const acsparseMatDescr_t descrA,
                    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                    int blockDim, const acsparseMatDescr_t descrC,
                    int *bsrSortedRowPtrC, int *nnzTotalDevHostPtr) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)bsrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  fprintf(stderr, "acsparseXcsr2bsrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXcsr2gebsrNnz(acsparseHandle_t handle, acsparseDirection_t dirA, int m,
                      int n, const acsparseMatDescr_t descrA,
                      const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                      const acsparseMatDescr_t descrC, int *bsrSortedRowPtrC,
                      int rowBlockDim, int colBlockDim, int *nnzTotalDevHostPtr,
                      void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)descrC;
  (void)bsrSortedRowPtrC;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseXcsr2gebsrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseXcsrgeam2Nnz(
    acsparseHandle_t handle, int m, int n, const acsparseMatDescr_t descrA,
    int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const acsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const acsparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace) {
  (void)handle;
  (void)m;
  (void)n;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)nnzTotalDevHostPtr;
  (void)workspace;
  fprintf(stderr, "acsparseXcsrgeam2Nnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXcsric02_zeroPivot(acsparseHandle_t handle, csric02Info_t info,
                           int *position) {
  (void)handle;
  (void)info;
  (void)position;
  fprintf(stderr, "acsparseXcsric02_zeroPivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXcsrilu02_zeroPivot(acsparseHandle_t handle, csrilu02Info_t info,
                            int *position) {
  (void)handle;
  (void)info;
  (void)position;
  fprintf(stderr, "acsparseXcsrilu02_zeroPivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseXgebsr2csr(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                   int nb, const acsparseMatDescr_t descrA,
                   const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                   int rowBlockDim, int colBlockDim,
                   const acsparseMatDescr_t descrC, int *csrSortedRowPtrC,
                   int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)descrC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseXgebsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseXgebsr2gebsrNnz(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
    const acsparseMatDescr_t descrC, int *bsrSortedRowPtrC, int rowBlockDimC,
    int colBlockDimC, int *nnzTotalDevHostPtr, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)descrC;
  (void)bsrSortedRowPtrC;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)nnzTotalDevHostPtr;
  (void)pBuffer;
  fprintf(stderr, "acsparseXgebsr2gebsrNnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsr2csr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const acsparseMatDescr_t descrC, acDoubleComplex *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseZbsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZbsric02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                 int nnzb, const acsparseMatDescr_t descrA,
                 acDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
                 acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseZbsric02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsric02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, acsparseSolvePolicy_t policy, void *pInputBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pInputBuffer;
  fprintf(stderr, "acsparseZbsric02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsric02_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZbsric02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsric02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZbsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZbsrilu02(acsparseHandle_t handle, acsparseDirection_t dirA, int mb,
                  int nnzb, const acsparseMatDescr_t descrA,
                  acDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
                  acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseZbsrilu02 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrilu02_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseZbsrilu02_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrilu02_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZbsrilu02_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrilu02_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZbsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZbsrilu02_numericBoost(acsparseHandle_t handle, bsrilu02Info_t info,
                               int enable_boost, double *tol,
                               acDoubleComplex *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseZbsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsm2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA,
    const acDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseZbsrsm2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsm2_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZbsrsm2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsm2_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transB, int mb, int n,
    int nnzb, const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrsm2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transB;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZbsrsm2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsm2_solve(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, acsparseOperation_t transXY, int mb, int n,
    int nnzb, const acDoubleComplex *alpha, const acsparseMatDescr_t descrA,
    const acDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info,
    const acDoubleComplex *B, int ldb, acDoubleComplex *X, int ldx,
    acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)transXY;
  (void)mb;
  (void)n;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)blockSize;
  (void)info;
  (void)B;
  (void)ldb;
  (void)X;
  (void)ldx;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseZbsrsm2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsv2_analysis(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseZbsrsv2_analysis is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsv2_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZbsrsv2_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsv2_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb,
    const acsparseMatDescr_t descrA, acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
    bsrsv2Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockSize;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZbsrsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrsv2_solve(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int mb, int nnzb, const acDoubleComplex *alpha,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    bsrsv2Info_t info, const acDoubleComplex *f, acDoubleComplex *x,
    acsparseSolvePolicy_t policy, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)mb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)info;
  (void)f;
  (void)x;
  (void)policy;
  (void)pBuffer;
  fprintf(stderr, "acsparseZbsrsv2_solve is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZbsrxmv(
    acsparseHandle_t handle, acsparseDirection_t dirA,
    acsparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb,
    const acDoubleComplex *alpha, const acsparseMatDescr_t descrA,
    const acDoubleComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
    const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
    const int *bsrSortedColIndA, int blockDim, const acDoubleComplex *x,
    const acDoubleComplex *beta, acDoubleComplex *y) {
  (void)handle;
  (void)dirA;
  (void)transA;
  (void)sizeOfMask;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)alpha;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedMaskPtrA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedEndPtrA;
  (void)bsrSortedColIndA;
  (void)blockDim;
  (void)x;
  (void)beta;
  (void)y;
  fprintf(stderr, "acsparseZbsrxmv is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsr2bsr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const acDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const acsparseMatDescr_t descrC, acDoubleComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)blockDim;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  fprintf(stderr, "acsparseZcsr2bsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsr2csr_compress(
    acsparseHandle_t handle, int m, int n, const acsparseMatDescr_t descrA,
    const acDoubleComplex *csrSortedValA, const int *csrSortedColIndA,
    const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    acDoubleComplex *csrSortedValC, int *csrSortedColIndC,
    int *csrSortedRowPtrC, acDoubleComplex tol) {
  (void)handle;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedColIndA;
  (void)csrSortedRowPtrA;
  (void)nnzA;
  (void)nnzPerRow;
  (void)csrSortedValC;
  (void)csrSortedColIndC;
  (void)csrSortedRowPtrC;
  (void)tol;
  fprintf(stderr, "acsparseZcsr2csr_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZcsr2csru(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, acDoubleComplex *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseZcsr2csru is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsr2gebsr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const acDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const acsparseMatDescr_t descrC, acDoubleComplex *bsrSortedValC,
    int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim,
    int colBlockDim, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBuffer;
  fprintf(stderr, "acsparseZcsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const acDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZcsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
    const acsparseMatDescr_t descrA, const acDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZcsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsrcolor(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    const acDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const acsparseColorInfo_t info) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)fractionToColor;
  (void)ncolors;
  (void)coloring;
  (void)reordering;
  (void)info;
  fprintf(stderr, "acsparseZcsrcolor is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZcsrgeam2(acsparseHandle_t handle, int m, int n,
                  const acDoubleComplex *alpha, const acsparseMatDescr_t descrA,
                  int nnzA, const acDoubleComplex *csrSortedValA,
                  const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                  const acDoubleComplex *beta, const acsparseMatDescr_t descrB,
                  int nnzB, const acDoubleComplex *csrSortedValB,
                  const int *csrSortedRowPtrB, const int *csrSortedColIndB,
                  const acsparseMatDescr_t descrC,
                  acDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
                  int *csrSortedColIndC, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBuffer;
  fprintf(stderr, "acsparseZcsrgeam2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsrgeam2_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const acDoubleComplex *alpha,
    const acsparseMatDescr_t descrA, int nnzA,
    const acDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const acDoubleComplex *beta,
    const acsparseMatDescr_t descrB, int nnzB,
    const acDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const acsparseMatDescr_t descrC,
    const acDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)alpha;
  (void)descrA;
  (void)nnzA;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)csrSortedColIndA;
  (void)beta;
  (void)descrB;
  (void)nnzB;
  (void)csrSortedValB;
  (void)csrSortedRowPtrB;
  (void)csrSortedColIndB;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZcsrgeam2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsric02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    acDoubleComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZcsric02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZcsrilu02_bufferSizeExt(
    acsparseHandle_t handle, int m, int nnz, const acsparseMatDescr_t descrA,
    acDoubleComplex *csrSortedVal, const int *csrSortedRowPtr,
    const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
  (void)handle;
  (void)m;
  (void)nnz;
  (void)descrA;
  (void)csrSortedVal;
  (void)csrSortedRowPtr;
  (void)csrSortedColInd;
  (void)info;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZcsrilu02_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZcsrilu02_numericBoost(acsparseHandle_t handle, csrilu02Info_t info,
                               int enable_boost, double *tol,
                               acDoubleComplex *boost_val) {
  (void)handle;
  (void)info;
  (void)enable_boost;
  (void)tol;
  (void)boost_val;
  fprintf(stderr, "acsparseZcsrilu02_numericBoost is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZcsru2csr(acsparseHandle_t handle, int m, int n, int nnz,
                  const acsparseMatDescr_t descrA, acDoubleComplex *csrVal,
                  const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info,
                  void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)descrA;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBuffer;
  fprintf(stderr, "acsparseZcsru2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZcsru2csr_bufferSizeExt(acsparseHandle_t handle, int m, int n, int nnz,
                                acDoubleComplex *csrVal, const int *csrRowPtr,
                                int *csrColInd, csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)nnz;
  (void)csrVal;
  (void)csrRowPtr;
  (void)csrColInd;
  (void)info;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZcsru2csr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgebsr2csr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const acsparseMatDescr_t descrC,
    acDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
    int *csrSortedColIndC) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)descrC;
  (void)csrSortedValC;
  (void)csrSortedRowPtrC;
  (void)csrSortedColIndC;
  fprintf(stderr, "acsparseZgebsr2csr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgebsr2gebsc_bufferSizeExt(
    acsparseHandle_t handle, int mb, int nb, int nnzb,
    const acDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
    size_t *pBufferSize) {
  (void)handle;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)bsrSortedVal;
  (void)bsrSortedRowPtr;
  (void)bsrSortedColInd;
  (void)rowBlockDim;
  (void)colBlockDim;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZgebsr2gebsc_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgebsr2gebsr(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const acsparseMatDescr_t descrC,
    acDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC,
    int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)descrC;
  (void)bsrSortedValC;
  (void)bsrSortedRowPtrC;
  (void)bsrSortedColIndC;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBuffer;
  fprintf(stderr, "acsparseZgebsr2gebsr is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgebsr2gebsr_bufferSize(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC,
    int *pBufferSizeInBytes) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSizeInBytes;
  fprintf(stderr, "acsparseZgebsr2gebsr_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgebsr2gebsr_bufferSizeExt(
    acsparseHandle_t handle, acsparseDirection_t dirA, int mb, int nb, int nnzb,
    const acsparseMatDescr_t descrA, const acDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
  (void)handle;
  (void)dirA;
  (void)mb;
  (void)nb;
  (void)nnzb;
  (void)descrA;
  (void)bsrSortedValA;
  (void)bsrSortedRowPtrA;
  (void)bsrSortedColIndA;
  (void)rowBlockDimA;
  (void)colBlockDimA;
  (void)rowBlockDimC;
  (void)colBlockDimC;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZgebsr2gebsr_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZgemvi(acsparseHandle_t handle, acsparseOperation_t transA, int m,
               int n, const acDoubleComplex *alpha, const acDoubleComplex *A,
               int lda, int nnz, const acDoubleComplex *xVal, const int *xInd,
               const acDoubleComplex *beta, acDoubleComplex *y,
               acsparseIndexBase_t idxBase, void *pBuffer) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)alpha;
  (void)A;
  (void)lda;
  (void)nnz;
  (void)xVal;
  (void)xInd;
  (void)beta;
  (void)y;
  (void)idxBase;
  (void)pBuffer;
  fprintf(stderr, "acsparseZgemvi is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZgemvi_bufferSize(acsparseHandle_t handle, acsparseOperation_t transA,
                          int m, int n, int nnz, int *pBufferSize) {
  (void)handle;
  (void)transA;
  (void)m;
  (void)n;
  (void)nnz;
  (void)pBufferSize;
  fprintf(stderr, "acsparseZgemvi_bufferSize is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgpsvInterleavedBatch(
    acsparseHandle_t handle, int algo, int m, acDoubleComplex *ds,
    acDoubleComplex *dl, acDoubleComplex *d, acDoubleComplex *du,
    acDoubleComplex *dw, acDoubleComplex *x, int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)ds;
  (void)dl;
  (void)d;
  (void)du;
  (void)dw;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseZgpsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgpsvInterleavedBatch_bufferSizeExt(
    acsparseHandle_t handle, int algo, int m, const acDoubleComplex *ds,
    const acDoubleComplex *dl, const acDoubleComplex *d,
    const acDoubleComplex *du, const acDoubleComplex *dw,
    const acDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)ds;
  (void)dl;
  (void)d;
  (void)du;
  (void)dw;
  (void)x;
  (void)batchCount;
  (void)pBufferSizeInBytes;
  fprintf(stderr,
          "acsparseZgpsvInterleavedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZgtsv2(acsparseHandle_t handle, int m, int n, const acDoubleComplex *dl,
               const acDoubleComplex *d, const acDoubleComplex *du,
               acDoubleComplex *B, int ldb, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseZgtsv2 is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZgtsv2StridedBatch(acsparseHandle_t handle, int m,
                           const acDoubleComplex *dl, const acDoubleComplex *d,
                           const acDoubleComplex *du, acDoubleComplex *x,
                           int batchCount, int batchStride, void *pBuffer) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)pBuffer;
  fprintf(stderr, "acsparseZgtsv2StridedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgtsv2StridedBatch_bufferSizeExt(
    acsparseHandle_t handle, int m, const acDoubleComplex *dl,
    const acDoubleComplex *d, const acDoubleComplex *du,
    const acDoubleComplex *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)batchStride;
  (void)bufferSizeInBytes;
  fprintf(stderr,
          "acsparseZgtsv2StridedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgtsv2_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const acDoubleComplex *dl,
    const acDoubleComplex *d, const acDoubleComplex *du,
    const acDoubleComplex *B, int ldb, size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseZgtsv2_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZgtsv2_nopivot(acsparseHandle_t handle, int m, int n,
                       const acDoubleComplex *dl, const acDoubleComplex *d,
                       const acDoubleComplex *du, acDoubleComplex *B, int ldb,
                       void *pBuffer) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)pBuffer;
  fprintf(stderr, "acsparseZgtsv2_nopivot is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgtsv2_nopivot_bufferSizeExt(
    acsparseHandle_t handle, int m, int n, const acDoubleComplex *dl,
    const acDoubleComplex *d, const acDoubleComplex *du,
    const acDoubleComplex *B, int ldb, size_t *bufferSizeInBytes) {
  (void)handle;
  (void)m;
  (void)n;
  (void)dl;
  (void)d;
  (void)du;
  (void)B;
  (void)ldb;
  (void)bufferSizeInBytes;
  fprintf(stderr, "acsparseZgtsv2_nopivot_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZgtsvInterleavedBatch(acsparseHandle_t handle, int algo, int m,
                              acDoubleComplex *dl, acDoubleComplex *d,
                              acDoubleComplex *du, acDoubleComplex *x,
                              int batchCount, void *pBuffer) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBuffer;
  fprintf(stderr, "acsparseZgtsvInterleavedBatch is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZgtsvInterleavedBatch_bufferSizeExt(
    acsparseHandle_t handle, int algo, int m, const acDoubleComplex *dl,
    const acDoubleComplex *d, const acDoubleComplex *du,
    const acDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
  (void)handle;
  (void)algo;
  (void)m;
  (void)dl;
  (void)d;
  (void)du;
  (void)x;
  (void)batchCount;
  (void)pBufferSizeInBytes;
  fprintf(stderr,
          "acsparseZgtsvInterleavedBatch_bufferSizeExt is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t
acsparseZnnz(acsparseHandle_t handle, acsparseDirection_t dirA, int m, int n,
             const acsparseMatDescr_t descrA, const acDoubleComplex *A, int lda,
             int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
  (void)handle;
  (void)dirA;
  (void)m;
  (void)n;
  (void)descrA;
  (void)A;
  (void)lda;
  (void)nnzPerRowCol;
  (void)nnzTotalDevHostPtr;
  fprintf(stderr, "acsparseZnnz is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

static inline acsparseStatus_t acsparseZnnz_compress(
    acsparseHandle_t handle, int m, const acsparseMatDescr_t descr,
    const acDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    int *nnzPerRow, int *nnzC, acDoubleComplex tol) {
  (void)handle;
  (void)m;
  (void)descr;
  (void)csrSortedValA;
  (void)csrSortedRowPtrA;
  (void)nnzPerRow;
  (void)nnzC;
  (void)tol;
  fprintf(stderr, "acsparseZnnz_compress is not supported.\n");
  exit(1);
  return (acsparseStatus_t)0; /* unreachable */
}

/* ── nvidia-ml: 4 unsupported APIs ── */

static inline hgmlReturn_t
hgmlDeviceGetGspFirmwareMode(hgmlDevice_t device, unsigned int *isEnabled,
                             unsigned int *defaultMode) {
  (void)device;
  (void)isEnabled;
  (void)defaultMode;
  fprintf(stderr, "hgmlDeviceGetGspFirmwareMode is not supported.\n");
  exit(1);
  return (hgmlReturn_t)0; /* unreachable */
}

static inline hgmlReturn_t hgmlDeviceGetGspFirmwareVersion(hgmlDevice_t device,
                                                           char *version) {
  (void)device;
  (void)version;
  fprintf(stderr, "hgmlDeviceGetGspFirmwareVersion is not supported.\n");
  exit(1);
  return (hgmlReturn_t)0; /* unreachable */
}

static inline hgmlReturn_t
hgmlSystemGetIcnlinkBwMode(unsigned int *nvlinkBwMode) {
  (void)nvlinkBwMode;
  fprintf(stderr, "hgmlSystemGetIcnlinkBwMode is not supported.\n");
  exit(1);
  return (hgmlReturn_t)0; /* unreachable */
}

static inline hgmlReturn_t
hgmlSystemSetIcnlinkBwMode(unsigned int nvlinkBwMode) {
  (void)nvlinkBwMode;
  fprintf(stderr, "hgmlSystemSetIcnlinkBwMode is not supported.\n");
  exit(1);
  return (hgmlReturn_t)0; /* unreachable */
}

/* ── nvrtc: 3 unsupported APIs ── */

static inline hgrtcResult hgrtcGetOptiXIR(hgrtcProgram prog, char *optixir) {
  (void)prog;
  (void)optixir;
  fprintf(stderr, "hgrtcGetOptiXIR is not supported.\n");
  exit(1);
  return (hgrtcResult)0; /* unreachable */
}

static inline hgrtcResult hgrtcGetOptiXIRSize(hgrtcProgram prog,
                                              size_t *optixirSizeRet) {
  (void)prog;
  (void)optixirSizeRet;
  fprintf(stderr, "hgrtcGetOptiXIRSize is not supported.\n");
  exit(1);
  return (hgrtcResult)0; /* unreachable */
}

static inline hgrtcResult hgrtcSetFlowCallback(hgrtcProgram prog,
                                               int (*callback)(void *, void *),
                                               void *payload) {
  (void)prog;
  (void)callback;
  (void)payload;
  fprintf(stderr, "hgrtcSetFlowCallback is not supported.\n");
  exit(1);
  return (hgrtcResult)0; /* unreachable */
}

#endif /* PPU_UNSUPPORTED_STUBS_H */
