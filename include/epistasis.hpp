#ifndef EPISTASIS_H_   
#define EPISTASIS_H_

/* The controls matrix and the cases matrix are expected to be padded to multiples of PADDING_SAMPLES in regard to samples */
#define PADDING_SAMPLES 1024

/* Number of SNPs per block */
#define SNP_BLOCK       32

/* Number of streams for concurrent execution of evaluation rounds per GPU device */
#define NUM_STREAMS     8       // set to 1 to disable

/* Settings related to fourth-order epistasis detection (k=4 interation order).
 * Only 2 out of the possible three alleles are represented (2 bits per tuple of {SNP, sample}).
 * There are 81 (= 3^4) genotype combinations to take into account in fourth-order detection.
 * Matrix operations are used to process 16 (= 2^4) genotypes, while the remaining 65 are analytically derived. */
#define SNP_CALC        2
#define INTER_OR        4
#define SNP_COMB        81
#define SNP_COMB_CALC   16


/* K2 Bayesian scoring function */

__inline__ __device__ float scoreQuadsK2(int *casesContTab, int *controlsContTab, float *tablePrecalc, int numCases, int numControls) {

        float score = 0.0f;

        for(int i = 0; i<3; i++) {
                for(int j = 0; j<3; j++) {
                        for(int k = 0; k<3; k++) {
                                for(int f = 0; f<3; f++) {
                                        score += __ldg(&tablePrecalc[controlsContTab[i*27 + j*9 + k*3 + f]]) + __ldg(&tablePrecalc[casesContTab[i*27 + j*9 + k*3 + f]]) - __ldg(&tablePrecalc[controlsContTab[i*27 + j*9 + k*3 + f] + casesContTab[i*27 + j*9 + k*3 + f] + 1]);
                                }
                        }
                }
        }
        score = fabs(score);

        return score;
}


/* Functions used as part of main loop for performing epistasis detection searches. */

__global__ void individualPopcount(int start_SNP_idx, uint *datasetCases, uint *datasetControls, uint *output_individualSNP_popcountsForCases, uint *output_individualSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal);

__global__ void pairwisePopcount(uint *datasetCases, uint *datasetControls, uint *output_pairwiseSNP_popcountsForCases, uint *output_pairwiseSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, uint SNP_A_start);

__global__ void combine(uint *datasetCases, uint *datasetControls, uint *outputCasesXY, uint *outputControlsXY, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, int startIndex_X, int startIndex_Y);

__global__ void applyScore_and_FindGloballyBestSol(int *C_ptrGPU_cases, int *C_ptrGPU_controls, int *C_ptrGPU_cases_XYZ, int *C_ptrGPU_controls_XYZ, int *C_ptrGPU_cases_WYZ, int *C_ptrGPU_controls_WYZ, int *C_ptrGPU_cases_WXY, int *C_ptrGPU_controls_WXY, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, uint* d_output_pairwiseSNP_popcountsForCases, uint* d_output_pairwiseSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_W, int start_X, int start_Y, int start_Z, int numSNPs, int numCases, int numControls);


#endif
