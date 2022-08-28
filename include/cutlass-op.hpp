#ifndef CUTLASS_OP_H_   
#define CUTLASS_OP_H_

/* CUTLASS v2.X template abstractions library. */
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"

/* Used for accessing tensor cores to perform tensorized fused AND+POPC (Ampere) or XOR+POPC (Ampere or Turing) operations. */
cudaError_t cutlass_U1_MmaMatOpTN(int m, int n, int k, cutlass::uint1b_t *A, int lda, cutlass::uint1b_t *B, int ldb, int32_t *C, int ldc, cudaStream_t stream);

#endif
