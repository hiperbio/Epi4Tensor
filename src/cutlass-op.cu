#include "cutlass-op.hpp"

#include <iostream>


/* Both Ampere and Turing are supported in CUTLASS 2.X.
 * 
 * OpXorPopc            : performs XOR+POPC (supported on Turing and Ampere)
 * OpMultiplyAdd        : performs AND+POPC when operating on binary inputs (supported on Ampere) 
 * 
 * The following settings have been experimentally found to result in high performance for Turing (SM70) and Ampere GPUs (SM80/SM86).
 * */

#if defined(AMPERE_86_AND)
#define CUDA_ARCH       cutlass::arch::Sm80     // Sm86 not recognized in CUTLASS v2.5
#define TENSOR_OP       cutlass::arch::OpMultiplyAdd
#define MMA_OP_SIZE     cutlass::gemm::GemmShape<16, 8, 256>
#define MMA_WARP_SIZE   cutlass::gemm::GemmShape<64, 64, 512>   
#define MMA_TBLOCK_SIZE cutlass::gemm::GemmShape<128, 256, 512> 
#define MMA_NUM_STAGES  3       

#elif defined(AMPERE_86_XOR)
#define CUDA_ARCH       cutlass::arch::Sm80     // Sm86 not recognized in CUTLASS V2.5
#define TENSOR_OP       cutlass::arch::OpXorPopc
#define MMA_OP_SIZE     cutlass::gemm::GemmShape<16, 8, 256>
#define MMA_WARP_SIZE   cutlass::gemm::GemmShape<64, 64, 512>   
#define MMA_TBLOCK_SIZE cutlass::gemm::GemmShape<128, 256, 512> 
#define MMA_NUM_STAGES  3       

#elif defined(AMPERE_80_AND)
#define CUDA_ARCH       cutlass::arch::Sm80
#define TENSOR_OP       cutlass::arch::OpMultiplyAdd
#define MMA_OP_SIZE     cutlass::gemm::GemmShape<16, 8, 256>
#define MMA_WARP_SIZE   cutlass::gemm::GemmShape<64, 64, 1024>
#define MMA_TBLOCK_SIZE cutlass::gemm::GemmShape<128, 256, 1024>
#define MMA_NUM_STAGES  3

#elif defined(AMPERE_80_XOR)
#define CUDA_ARCH       cutlass::arch::Sm80
#define TENSOR_OP       cutlass::arch::OpXorPopc
#define MMA_OP_SIZE     cutlass::gemm::GemmShape<16, 8, 256>
#define MMA_WARP_SIZE   cutlass::gemm::GemmShape<64, 64, 1024>
#define MMA_TBLOCK_SIZE cutlass::gemm::GemmShape<128, 256, 1024>
#define MMA_NUM_STAGES  3

#else
#define CUDA_ARCH cutlass::arch::Sm75
#define TENSOR_OP cutlass::arch::OpXorPopc
#define MMA_OP_SIZE     cutlass::gemm::GemmShape<8, 8, 128>             
#define MMA_WARP_SIZE   cutlass::gemm::GemmShape<64, 32, 1024>          
#define MMA_TBLOCK_SIZE cutlass::gemm::GemmShape<128, 128, 1024>
#define MMA_NUM_STAGES  2                                               
#endif


/* Checks for CUTLASS errors and prints error information.
 * from:  'cutlass/examples/common/helper.h' */

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/* Setup of AND+POPC (or XOR+POPC) 1-bit tensor core accelerated kernel for compatible Ampere (or Turing) GPUs. */

using LinearCombinationOp = cutlass::epilogue::thread::LinearCombination<
	int32_t,					// Numerical type of output matrix
	128 / cutlass::sizeof_bits<int32_t>::value,  	// Elements per memory access
	int32_t,                                	// Numerical type of accumulator
	int32_t>;  					// Numerical type of alpha and beta

using BinaryMatrixOpKernel = cutlass::gemm::device::Gemm<
	cutlass::uint1b_t,		// Numerical type of input Matrix A
	cutlass::layout::RowMajor,	// Matrix A is row-major
	cutlass::uint1b_t,		// Numerical type of input Matrix B
	cutlass::layout::ColumnMajor,	// Matrix B is column-major
	int32_t,			// Numerical type of output matrix
      	cutlass::layout::ColumnMajor,	// Matrix C (output) is column-major
      	int32_t,			// Numerical type of accumulator
      	cutlass::arch::OpClassTensorOp,	// Tensor cores or CUDA cores
      	CUDA_ARCH,			// SM architecture
      	MMA_TBLOCK_SIZE,		// Tile shape that a thread-block computes
      	MMA_WARP_SIZE,			// Tile shape that a warp computes
      	MMA_OP_SIZE,			// Tile instruction size of MMA operation
      	LinearCombinationOp,
      	cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,	// Type of scheduling of thread blocks
      	MMA_NUM_STAGES,			// Number of pipelines
      	128, 
      	128,
      	false, 
      	TENSOR_OP>;			// Operation to perform on tensor cores



/* Relies on CUTLASS 2.X to perform matrix-matrix operation.
 * C := alpha * op(A) * op(B) + beta * C
 */
cudaError_t cutlass_U1_MmaMatOpTN(int m, int n, int k, cutlass::uint1b_t *A, int lda, cutlass::uint1b_t *B, int ldb, int32_t *C, int ldc, cudaStream_t stream) {

	cutlass::layout::RowMajor layoutMatrixA(lda);
	cutlass::TensorRef< cutlass::uint1b_t, cutlass::layout::RowMajor> A_ref(A, layoutMatrixA);

	cutlass::layout::ColumnMajor layoutMatrixB(ldb);
	cutlass::TensorRef< cutlass::uint1b_t, cutlass::layout::ColumnMajor> B_ref(B, layoutMatrixB);

	cutlass::layout::ColumnMajor layoutMatrixC(ldc);
	cutlass::TensorRef< int32_t, cutlass::layout::ColumnMajor> C_ref(C, layoutMatrixC);

	/* Creates GEMM-like kernel arguments tuple. */
	typename BinaryMatrixOpKernel::Arguments kernelArguments{
		cutlass::gemm::GemmCoord(m,n,k),	// matrix-matrix operation problem shape
		A_ref,					// first input matrix
		B_ref,					// second input matrix
		C_ref,					// 'beta' equals 0 --> 'beta * C' equals 0
		C_ref,					// output matrix
		{1, 0},          			// alpha and beta scale factors
		1};        				// split factor of 'k' dimension


	/* Queries and allocates memory for matrix-matrix operation */
	size_t workspaceSize = BinaryMatrixOpKernel::get_workspace_size(kernelArguments);
	cutlass::device_memory::allocation<cutlass::uint1b_t> kernelOpWorkspace(workspaceSize);

	/* Instantiates matrix-matrix operation kernel and assesses if problem shape is supported */
	BinaryMatrixOpKernel matrixOpKernel;
	cutlass::Status status = matrixOpKernel.can_implement(kernelArguments);
	CUTLASS_CHECK(status);

	/* Initializes kernel passing the created arguments tuple and a pointer to the allocated workspace */
	status = matrixOpKernel.initialize(kernelArguments, kernelOpWorkspace.get(), stream);	
	CUTLASS_CHECK(status);

	/* Launches matrix-matrix operation kernel */
	status = matrixOpKernel(stream);
	CUTLASS_CHECK(status);

	return cudaGetLastError();

}



