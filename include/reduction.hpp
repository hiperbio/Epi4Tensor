#ifndef REDUCTION_H_   
#define REDUCTION_H_

#include <cfloat>


/* GPU kernel code for MIN reduction. Based on: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler */

__inline__ __device__ float shfl_xor_32(float scalarValue, const int n) {
        return __shfl_xor_sync(0xFFFFFFFF, scalarValue, n);
}

__inline__ __device__ float warpReduceMin(float val) {
        val = min(val, shfl_xor_32(val, 1));
        val = min(val, shfl_xor_32(val, 2));
        val = min(val, shfl_xor_32(val, 4));
        val = min(val, shfl_xor_32(val, 8));
        val = min(val, shfl_xor_32(val, 16));
        return val;
}

__inline__ __device__ float blockReduceMin_2D(float val) {
	static __shared__ float shared[32];
	val = warpReduceMin(val);
	if (threadIdx.x == 0) shared[threadIdx.y]=val;
	__syncthreads();
	val = (threadIdx.x < blockDim.y) ? shared[threadIdx.x] : FLT_MAX;

	float val_min = warpReduceMin(val);
	if (threadIdx.x==0 && threadIdx.y==0) shared[0] = val_min;

	__syncthreads();
	val = shared[0];
	return val;
}


/* GPU kernel code for saving best cadidate score. Based on: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/ */

__inline__ __device__ void atomicMin_g_f(float *addr, float val)
{
	union {
		unsigned int u32;
		float        f32;
	} next, expected, current;
	current.f32    = *addr;
	do {
		expected.f32 = current.f32;
		next.f32     = min(expected.f32,  val);
		current.u32  = atomicCAS( (unsigned int *)addr,
				expected.u32, next.u32);
	} while( current.u32 != expected.u32 );
}


/* GPU kernel code for saving indexes (encoded in 64-bit integer) of set of SNPs with best cadidate score */

__inline__ __device__ void atomicMinGetIndex(float *addr, float val, unsigned long long int *index, unsigned long long int nextPackedIndices)
{
	unsigned long long int expected, current;
	current = *index;

        do {
                expected = current;
                float global_minVal = *addr;
                if(val <= global_minVal) {
                        current  = atomicCAS( index, expected, nextPackedIndices);
                }
        } while( current != expected );
}


#endif
