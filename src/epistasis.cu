#include "epistasis.hpp"

#include <cfloat>
#include <iostream>

#include "cutlass-op.hpp"
#include "reduction.hpp"


/* MACROS for deriving the genotype counts for 65 out of 81 possible genotypes in fourth-order contingency tables. */

#define CALC_MACRO_A_B_C(a, b, c, snpIndexA, snpIndexB, snpIndexC); {\
	casesContTab[(a * 27 + b * 9 + c * 3 + 2)] = casesContTab_inferTripletsWXY[a*9 + b*3 + c] - (casesContTab[(a * 27 + b * 9 + c * 3 + 0)] + casesContTab[(a * 27 + b * 9 + c * 3 + 1)]); \
	controlsContTab[(a * 27 + b * 9 + c * 3 + 2)] = controlsContTab_inferTripletsWXY[a*9 + b*3 + c] - (controlsContTab[(a * 27 + b * 9 + c * 3 + 0)] + controlsContTab[(a * 27 + b * 9 + c * 3 + 1)]); \
}


#define CALC_MACRO_A_B_D(a, b, d, snpIndexA, snpIndexB, snpIndexD); {\
	casesContTab[(a * 27 + b * 9 + 2 * 3 + d)] = casesContTab_inferTriplets_WXZ[a*9 + b*3 + d] - (casesContTab[(a * 27 + b * 9 + 0 * 3 + d)] + casesContTab[(a * 27 + b * 9 + 1 * 3 + d)]); \
	controlsContTab[(a * 27 + b * 9 + 2 * 3 + d)] = controlsContTab_inferTriplets_WXZ[a*9 + b*3 + d] - (controlsContTab[(a * 27 + b * 9 + 0 * 3 + d)] + controlsContTab[(a * 27 + b * 9 + 1 * 3 + d)]); \
}


#define CALC_MACRO_A_C_D(a, c, d, snpIndexA, snpIndexC, snpIndexD); {\
	casesContTab[(a * 27 + 2 * 9 + c * 3 + d)] = casesContTab_inferTriplets_W_Y_Z[a*9 + c*3 + d] - (casesContTab[(a * 27 + 0 * 9 + c * 3 + d)] + casesContTab[(a * 27 + 1 * 9 + c * 3 + d)]); \
	controlsContTab[(a * 27 + 2 * 9 + c * 3 + d)] = controlsContTab_inferTriplets_W_Y_Z[a*9 + c*3 + d] - (controlsContTab[(a * 27 + 0 * 9 + c * 3 + d)] + controlsContTab[(a * 27 + 1 * 9 + c * 3 + d)]); \
}

#define CALC_MACRO_B_C_D(b, c, d, snpIndexB, snpIndexC, snpIndexD); {\
	casesContTab[(2 * 27 + b * 9 + c * 3 + d)] = casesContTab_inferTriplets_X_Y_Z[b*9 + c*3 + d] - (casesContTab[(0 * 27 + b * 9 + c * 3 + d)] + casesContTab[(1 * 27 + b * 9 + c * 3 + d)]); \
	controlsContTab[(2 * 27 + b * 9 + c * 3 + d)] = controlsContTab_inferTriplets_X_Y_Z[b*9 + c*3 + d] - (controlsContTab[(0 * 27 + b * 9 + c * 3 + d)] + controlsContTab[(1 * 27 + b * 9 + c * 3 + d)]); \
}


/* MACROS for deriving the genotype counts for 19 out of 27 possible genotypes in third-order contingency tables. */

#define CALC_MACRO_W_X_Y(x, y, z, snpIndexA, snpIndexB, a, b, x1,y1,z1, x2,y2,z2) {\
	casesContTab_inferTripletsWXY[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForCases[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (casesContTab_inferTripletsWXY[x1*9+y1*3+z1] + casesContTab_inferTripletsWXY[x2*9+y2*3+z2]);\
	controlsContTab_inferTripletsWXY[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForControls[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (controlsContTab_inferTripletsWXY[x1*9+y1*3+z1] + controlsContTab_inferTripletsWXY[x2*9+y2*3+z2]);\
}


#define CALC_MACRO_W_X_Z(x, y, z, snpIndexA, snpIndexB, a, b, x1,y1,z1, x2,y2,z2) {\
	casesContTab_inferTriplets_WXZ[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForCases[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (casesContTab_inferTriplets_WXZ[x1*9+y1*3+z1] + casesContTab_inferTriplets_WXZ[x2*9+y2*3+z2]);\
	controlsContTab_inferTriplets_WXZ[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForControls[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (controlsContTab_inferTriplets_WXZ[x1*9+y1*3+z1] + controlsContTab_inferTriplets_WXZ[x2*9+y2*3+z2]);\
}


#define CALC_MACRO_W_Y_Z(x, y, z, snpIndexA, snpIndexB, a, b, x1,y1,z1, x2,y2,z2) {\
	casesContTab_inferTriplets_W_Y_Z[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForCases[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (casesContTab_inferTriplets_W_Y_Z[x1*9+y1*3+z1] + casesContTab_inferTriplets_W_Y_Z[x2*9+y2*3+z2]);\
	controlsContTab_inferTriplets_W_Y_Z[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForControls[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (controlsContTab_inferTriplets_W_Y_Z[x1*9+y1*3+z1] + controlsContTab_inferTriplets_W_Y_Z[x2*9+y2*3+z2]);\
}


#define CALC_MACRO_X_Y_Z(x, y, z, snpIndexA, snpIndexB, a, b, x1,y1,z1, x2,y2,z2) {\
	casesContTab_inferTriplets_X_Y_Z[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForCases[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (casesContTab_inferTriplets_X_Y_Z[x1*9+y1*3+z1] + casesContTab_inferTriplets_X_Y_Z[x2*9+y2*3+z2]);\
	controlsContTab_inferTriplets_X_Y_Z[x*9+y*3+z] = d_output_pairwiseSNP_popcountsForControls[(a*3+b) * (numSNPs * numSNPs) + ( snpIndexA ) * numSNPs + snpIndexB] - (controlsContTab_inferTriplets_X_Y_Z[x1*9+y1*3+z1] + controlsContTab_inferTriplets_X_Y_Z[x2*9+y2*3+z2]);\
}



/* Individual {SNP, allele} population count calculation. Used by applyScore_and_FindGloballyBestSol(). 
   Counts for third genotype (homozygous minor) are derived from the other two genotypes.
 */
__global__ void individualPopcount(int start_SNP_idx, uint *datasetCases, uint *datasetControls, uint *output_individualSNP_popcountsForCases, uint *output_individualSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal)
{

	uint SNP_i = start_SNP_idx + blockDim.x * blockIdx.x + threadIdx.x;
	int cases_i, controls_i;

	int casesSizeNoPadding = ceil(((float) casesSizeOriginal) / 32.0f);
	int controlsSizeNoPadding = ceil(((float) controlsSizeOriginal) / 32.0f);

	int casesSize = ceil(((float) casesSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) controlsSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

	int casesZerosAcc = 0;
	int casesOnesAcc = 0;

	int controlsZerosAcc = 0;
	int controlsOnesAcc = 0;

	if(SNP_i < numSNPs) {           // To ensure processing is inside bounds.

		for(cases_i = 0; cases_i < casesSizeNoPadding; cases_i++) {
			casesZerosAcc += __popc(datasetCases[SNP_i * SNP_CALC * casesSize + cases_i]);
			casesOnesAcc += __popc(datasetCases[SNP_i * SNP_CALC * casesSize + casesSize + cases_i]);
		}

		output_individualSNP_popcountsForCases[0 * numSNPs + SNP_i] = casesZerosAcc;
		output_individualSNP_popcountsForCases[1 * numSNPs + SNP_i] = casesOnesAcc;
		output_individualSNP_popcountsForCases[2 * numSNPs + SNP_i] = casesSizeOriginal - (casesZerosAcc + casesOnesAcc);       	

		for(controls_i = 0; controls_i < controlsSizeNoPadding; controls_i++) {
			controlsZerosAcc += __popc(datasetControls[SNP_i * SNP_CALC * controlsSize + controls_i]);
			controlsOnesAcc += __popc(datasetControls[SNP_i * SNP_CALC * controlsSize + controlsSize + controls_i]);
		}

		output_individualSNP_popcountsForControls[0 * numSNPs + SNP_i] = controlsZerosAcc;
		output_individualSNP_popcountsForControls[1 * numSNPs + SNP_i] = controlsOnesAcc;
		output_individualSNP_popcountsForControls[2 * numSNPs + SNP_i] = controlsSizeOriginal - (controlsZerosAcc + controlsOnesAcc);  

	}
}


/* Construction of contingency tables for pair-wise interactions.
   Data for third genotype is derived from the other two genotypes. */  

__global__ void pairwisePopcount(uint *datasetCases, uint *datasetControls, uint *output_pairwiseSNP_popcountsForCases, uint *output_pairwiseSNP_popcountsForControls, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, uint SNP_A_start)
{
	uint SNP_A_i_fromBlockStart = blockDim.x * blockIdx.x + threadIdx.x;
	uint SNP_A_i = SNP_A_start + SNP_A_i_fromBlockStart;
	uint SNP_B_i_fromBlockStart = blockDim.y * blockIdx.y + threadIdx.y;
	uint SNP_B_i = 0 + SNP_B_i_fromBlockStart;

	int cases_i, controls_i;

	int casesSizeNoPadding = ceil(((float) casesSizeOriginal) / 32.0f);
	int controlsSizeNoPadding = ceil(((float) controlsSizeOriginal) / 32.0f);

	int casesSize = ceil(((float) casesSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) controlsSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

	uint maskRelevantBitsSetCases = (~0u) << (casesSizeNoPadding * 32 - casesSizeOriginal); // Mask where the only bits not set are the 'casesSize * 32 - casesSizeOriginal' less significant bits.
	uint maskRelevantBitsSetControls = (~0u) << (controlsSizeNoPadding * 32 - controlsSizeOriginal); // Mask where the only bits not set are the 'controlsSize * 32 - controlsSizeOriginal' less significant bits.

	if((SNP_A_i < numSNPs) && (SNP_B_i < numSNPs)) {       // This is because there may be more threads launched than combinations.

		int casesCountsArr[9];
		for(int i=0; i<9; i++) {
			casesCountsArr[i] = 0;
		}

		int controlsCountsArr[9];
		for(int i=0; i<9; i++) {
			controlsCountsArr[i] = 0;
		}

		unsigned int cases_0_A, cases_1_A, cases_2_A, cases_0_B, cases_1_B, cases_2_B;
		for(cases_i = 0; cases_i < (casesSizeNoPadding - 1); cases_i++) {

			cases_0_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + cases_i];
			cases_1_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + casesSize + cases_i];
			cases_2_A = ~(cases_0_A | cases_1_A);

			cases_0_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + cases_i];
			cases_1_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + casesSize + cases_i];
			cases_2_B = ~(cases_0_B | cases_1_B);

			casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
			casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
			casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
			casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
			casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
			casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
			casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
			casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
			casesCountsArr[8] += __popc(cases_2_A & cases_2_B);
		}

		/* Processes last 32-bit bit-pack in order to take into acount when number of cases is not multiple of 32. */

		cases_0_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + cases_i];
		cases_1_A = datasetCases[SNP_A_i * SNP_CALC * casesSize + casesSize + cases_i];
		cases_2_A = (~(cases_0_A | cases_1_A)) & maskRelevantBitsSetCases;

		cases_0_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + cases_i];
		cases_1_B = datasetCases[SNP_B_i * SNP_CALC * casesSize + casesSize + cases_i];
		cases_2_B = (~(cases_0_B | cases_1_B)) & maskRelevantBitsSetCases;

		casesCountsArr[0] += __popc(cases_0_A & cases_0_B);
		casesCountsArr[1] += __popc(cases_0_A & cases_1_B);
		casesCountsArr[2] += __popc(cases_0_A & cases_2_B);
		casesCountsArr[3] += __popc(cases_1_A & cases_0_B);
		casesCountsArr[4] += __popc(cases_1_A & cases_1_B);
		casesCountsArr[5] += __popc(cases_1_A & cases_2_B);
		casesCountsArr[6] += __popc(cases_2_A & cases_0_B);
		casesCountsArr[7] += __popc(cases_2_A & cases_1_B);
		casesCountsArr[8] += __popc(cases_2_A & cases_2_B);


		output_pairwiseSNP_popcountsForCases[0 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[0];	
		output_pairwiseSNP_popcountsForCases[1 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[1];
		output_pairwiseSNP_popcountsForCases[2 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[2];
		output_pairwiseSNP_popcountsForCases[3 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[3];
		output_pairwiseSNP_popcountsForCases[4 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[4];
		output_pairwiseSNP_popcountsForCases[5 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[5];
		output_pairwiseSNP_popcountsForCases[6 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[6];
		output_pairwiseSNP_popcountsForCases[7 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[7];
		output_pairwiseSNP_popcountsForCases[8 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = casesCountsArr[8];


		unsigned int controls_0_A, controls_1_A, controls_2_A, controls_0_B, controls_1_B, controls_2_B;
		for(controls_i = 0; controls_i < (controlsSizeNoPadding - 1); controls_i++) {

			controls_0_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controls_i];
			controls_1_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controlsSize + controls_i];
			controls_2_A = ~(controls_0_A | controls_1_A);

			controls_0_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controls_i];
			controls_1_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controlsSize + controls_i];
			controls_2_B = ~(controls_0_B | controls_1_B);

			controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
			controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
			controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
			controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
			controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
			controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
			controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
			controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
			controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);
		}

		/* Processes last 32-bit bit-pack in order to take into acount when number of controls is not multiple of 32. */

		controls_0_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controls_i];
		controls_1_A = datasetControls[SNP_A_i * SNP_CALC * controlsSize + controlsSize + controls_i];
		controls_2_A = (~(controls_0_A | controls_1_A)) & maskRelevantBitsSetControls;

		controls_0_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controls_i];
		controls_1_B = datasetControls[SNP_B_i * SNP_CALC * controlsSize + controlsSize + controls_i];
		controls_2_B = (~(controls_0_B | controls_1_B)) & maskRelevantBitsSetControls;

		controlsCountsArr[0] += __popc(controls_0_A & controls_0_B);
		controlsCountsArr[1] += __popc(controls_0_A & controls_1_B);
		controlsCountsArr[2] += __popc(controls_0_A & controls_2_B);
		controlsCountsArr[3] += __popc(controls_1_A & controls_0_B);
		controlsCountsArr[4] += __popc(controls_1_A & controls_1_B);
		controlsCountsArr[5] += __popc(controls_1_A & controls_2_B);
		controlsCountsArr[6] += __popc(controls_2_A & controls_0_B);
		controlsCountsArr[7] += __popc(controls_2_A & controls_1_B);
		controlsCountsArr[8] += __popc(controls_2_A & controls_2_B);


		output_pairwiseSNP_popcountsForControls[0 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[0];
		output_pairwiseSNP_popcountsForControls[1 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[1];
		output_pairwiseSNP_popcountsForControls[2 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[2];
		output_pairwiseSNP_popcountsForControls[3 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[3];
		output_pairwiseSNP_popcountsForControls[4 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[4];
		output_pairwiseSNP_popcountsForControls[5 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[5];
		output_pairwiseSNP_popcountsForControls[6 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[6];
		output_pairwiseSNP_popcountsForControls[7 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[7];
		output_pairwiseSNP_popcountsForControls[8 * (numSNPs * numSNPs) + SNP_A_i * numSNPs + SNP_B_i] = controlsCountsArr[8];
	}
}




/* Combines genotypic data from a set of SNPs (X) with genotypic data from a set of other SNPs (Y). 
   This computation phase enables the use of fused AND+POPC (or XOR+POPC) on tensor cores in the context of fourth-order searches. */

__global__ void combine(uint *datasetCases, uint *datasetControls, uint *outputCasesXY, uint *outputControlsXY, int numSNPs, int casesSizeOriginal, int controlsSizeOriginal, int startIndex_X, int startIndex_Y)
{

	uint snp_X_fromBlockStart = blockDim.x * blockIdx.x + threadIdx.x;
	uint snp_X_index = startIndex_X + snp_X_fromBlockStart;

	uint snp_Y_fromBlockStart = blockDim.y * blockIdx.y + threadIdx.y;
	uint snp_Y_index = startIndex_Y + snp_Y_fromBlockStart;

	uint patient_idx_thread = threadIdx.z;


	int cases_i, controls_i;

	int casesSize = ceil(((float) casesSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;
	int controlsSize = ceil(((float) controlsSizeOriginal) / ((float) PADDING_SAMPLES)) * PADDING_SAMPLES / 32;

	if((snp_X_index < numSNPs) && (snp_Y_index < numSNPs)) {       // To ensure processing is within bounds.

		for(int i=0; i<SNP_CALC; i++) {
			for(int j=0; j<SNP_CALC; j++) {

				// cases
				for(cases_i = 0; cases_i < casesSize; cases_i += blockDim.z) {
					outputCasesXY[snp_X_fromBlockStart * SNP_BLOCK * (SNP_CALC * SNP_CALC) * casesSize + snp_Y_fromBlockStart * (SNP_CALC * SNP_CALC) * casesSize + (i * SNP_CALC + j) * casesSize + cases_i + patient_idx_thread] = datasetCases[snp_X_index * SNP_CALC * casesSize + i * casesSize + cases_i + patient_idx_thread] & datasetCases[snp_Y_index * SNP_CALC * casesSize + j * casesSize + cases_i + patient_idx_thread];
				}

				// controls
				for(controls_i = 0; controls_i < controlsSize; controls_i += blockDim.z) {
					outputControlsXY[snp_X_fromBlockStart * SNP_BLOCK * (SNP_CALC * SNP_CALC) * controlsSize + snp_Y_fromBlockStart * (SNP_CALC * SNP_CALC) * controlsSize + (i * SNP_CALC + j) * controlsSize + controls_i + patient_idx_thread] = datasetControls[snp_X_index * SNP_CALC * controlsSize + i * controlsSize + controls_i + patient_idx_thread] & datasetControls[snp_Y_index * SNP_CALC * controlsSize + j * controlsSize + controls_i + patient_idx_thread];
				}
			}
		}
	}
}



/* Computes and reduces scores, identifying the best SNP combination. */

__global__ void applyScore_and_FindGloballyBestSol(int *C_ptrGPU_cases, int *C_ptrGPU_controls, int *C_ptrGPU_cases_XYZ, int *C_ptrGPU_controls_XYZ, int *C_ptrGPU_cases_WYZ, int *C_ptrGPU_controls_WYZ, int *C_ptrGPU_cases_WXY, int *C_ptrGPU_controls_WXY, uint * d_output_individualSNP_popcountsForCases, uint *d_output_individualSNP_popcountsForControls, uint* d_output_pairwiseSNP_popcountsForCases, uint* d_output_pairwiseSNP_popcountsForControls, float *tablePrecalc, float *output, unsigned long long int *output_packedIndices, int start_W, int start_X, int start_Y, int start_Z, int numSNPs, int numCases, int numControls)
{
	int local_id = threadIdx.x;

	int SNP_X = blockDim.x * blockIdx.x + threadIdx.x;	
	int SNP_Y = blockDim.y * blockIdx.y + threadIdx.y;
	int SNP_Z = blockDim.z * blockIdx.z + threadIdx.z;

	int SNP_W_withBestScore;		// Stores the index of the SNP W that results in minimum score.

	float score = FLT_MAX;

	if( ((start_X + SNP_X) < numSNPs) && ((start_Y + SNP_Y) < numSNPs) && ((start_Z + SNP_Z) < numSNPs)) {      


		// X_Y_Z

		int casesContTab_inferTriplets_X_Y_Z[27];		
		int controlsContTab_inferTriplets_X_Y_Z[27];


		#if defined(AMPERE_80_AND) || defined(AMPERE_86_AND)
		for(int i = 0; i<SNP_CALC; i++) {
			for(int j = 0; j<SNP_CALC; j++) {
				for(int k = 0; k<SNP_CALC; k++) {


					casesContTab_inferTriplets_X_Y_Z[i*9 + j*3 + k] = C_ptrGPU_cases_XYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_X * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)];        
					controlsContTab_inferTriplets_X_Y_Z[i*9 + j*3 + k] = C_ptrGPU_controls_XYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_X * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)];    
				}
			}
		}

		// FOR WHEN USING XOR+POPC
		#else	
		for(int i = 0; i<SNP_CALC; i++) {
			for(int j = 0; j<SNP_CALC; j++) {
				for(int k = 0; k<SNP_CALC; k++) {

					casesContTab_inferTriplets_X_Y_Z[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForCases[(i*3+j) * (numSNPs * numSNPs) + ( start_X + SNP_X ) * numSNPs + start_Y + SNP_Y] + d_output_individualSNP_popcountsForCases[k * numSNPs + start_Z + SNP_Z]) - C_ptrGPU_cases_XYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_X * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)])) >> 1;        
					controlsContTab_inferTriplets_X_Y_Z[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForControls[(i*3+j) * (numSNPs * numSNPs) + ( start_X + SNP_X ) * numSNPs + start_Y + SNP_Y] + d_output_individualSNP_popcountsForControls[k * numSNPs + start_Z + SNP_Z]) - C_ptrGPU_controls_XYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_X * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)])) >> 1;    
				}
			}
		}
		#endif


		// $\{0,0,2\}$ & $\{0,0,:\} - (\{0,0,0\} + \{0,0,1\})$
		CALC_MACRO_X_Y_Z(0,0,2, start_X + SNP_X, start_Y + SNP_Y, 0,0, 0,0,0, 0,0,1);

		// $\{0,1,2\}$ & $\{0,1,:\} - (\{0,1,0\} + \{0,1,1\})$
		CALC_MACRO_X_Y_Z(0,1,2, start_X + SNP_X, start_Y + SNP_Y, 0,1, 0,1,0, 0,1,1);

		// $\{0,2,0\}$ & $\{0,:,0\} - (\{0,0,0\} + \{0,1,0\})$
		CALC_MACRO_X_Y_Z(0,2,0, start_X + SNP_X, start_Z + SNP_Z, 0,0, 0,0,0, 0,1,0);

		// $\{0,2,1\}$ & $\{0,:,1\} - (\{0,0,1\} + \{0,1,1\})$
		CALC_MACRO_X_Y_Z(0,2,1, start_X + SNP_X, start_Z + SNP_Z, 0,1, 0,0,1, 0,1,1);

		// $\{0,2,2\}$ & $\{0,:,2\} - (\{0,0,2\} + \{0,1,2\})$
		CALC_MACRO_X_Y_Z(0,2,2, start_X + SNP_X, start_Z + SNP_Z, 0,2, 0,0,2, 0,1,2);

		// $\{1,0,2\}$ & $\{1,0,:\} - (\{1,0,0\} + \{1,0,1\})$
		CALC_MACRO_X_Y_Z(1,0,2, start_X + SNP_X, start_Y + SNP_Y, 1,0, 1,0,0, 1,0,1);

		// $\{1,1,2\}$ & $\{1,1,:\} - (\{1,1,0\} + \{1,1,1\})$
		CALC_MACRO_X_Y_Z(1,1,2, start_X + SNP_X, start_Y + SNP_Y, 1,1, 1,1,0, 1,1,1);

		// $\{1,2,0\}$ & $\{1,:,0\} - (\{1,0,0\} + \{1,1,0\})$
		CALC_MACRO_X_Y_Z(1,2,0, start_X + SNP_X, start_Z + SNP_Z, 1,0, 1,0,0, 1,1,0);

		// $\{1,2,1\}$ & $\{1,:,1\} - (\{1,0,1\} + \{1,1,1\})$
		CALC_MACRO_X_Y_Z(1,2,1, start_X + SNP_X, start_Z + SNP_Z, 1,1, 1,0,1, 1,1,1);

		// $\{1,2,2\}$ & $\{1,:,2\} - (\{1,0,2\} + \{1,1,2\})$
		CALC_MACRO_X_Y_Z(1,2,2, start_X + SNP_X, start_Z + SNP_Z, 1,2, 1,0,2, 1,1,2);

		// $\{2,0,0\}$ & $\{:,0,0\} - (\{0,0,0\} + \{1,0,0\})$
		CALC_MACRO_X_Y_Z(2,0,0, start_Y + SNP_Y, start_Z + SNP_Z, 0,0, 0,0,0, 1,0,0);

		// $\{2,0,1\}$ & $\{:,0,1\} - (\{0,0,1\} + \{1,0,1\})$
		CALC_MACRO_X_Y_Z(2,0,1, start_Y + SNP_Y, start_Z + SNP_Z, 0,1, 0,0,1, 1,0,1);

		// $\{2,0,2\}$ & $\{2,0,:\} - (\{2,0,0\} + \{2,0,1\})$
		CALC_MACRO_X_Y_Z(2,0,2, start_X + SNP_X, start_Y + SNP_Y, 2,0, 2,0,0, 2,0,1);	

		// $\{2,1,0\}$ & $\{:,1,0\} - (\{0,1,0\} + \{1,1,0\})$
		CALC_MACRO_X_Y_Z(2,1,0, start_Y + SNP_Y, start_Z + SNP_Z, 1,0, 0,1,0, 1,1,0);

		// $\{2,1,1\}$ & $\{:,1,1\} - (\{0,1,1\} + \{1,1,1\})$
		CALC_MACRO_X_Y_Z(2,1,1, start_Y + SNP_Y, start_Z + SNP_Z, 1,1, 0,1,1, 1,1,1);

		// $\{2,1,2\}$ & $\{2,1,:\} - (\{2,1,0\} + \{2,1,1\})$
		CALC_MACRO_X_Y_Z(2,1,2, start_X + SNP_X, start_Y + SNP_Y, 2,1, 2,1,0, 2,1,1);

		// $\{2,2,0\}$ & $\{2,:,0\} - (\{2,0,0\} + \{2,1,0\})$
		CALC_MACRO_X_Y_Z(2,2,0, start_X + SNP_X, start_Z + SNP_Z, 2,0, 2,0,0, 2,1,0);

		// $\{2,2,1\}$ & $\{2,:,1\} - (\{2,0,1\} + \{2,1,1\})$
		CALC_MACRO_X_Y_Z(2,2,1, start_X + SNP_X, start_Z + SNP_Z, 2,1, 2,0,1, 2,1,1);

		// $\{2,2,2\}$ & $\{2,:,2\} - (\{2,0,2\} + \{2,1,2\})$
		CALC_MACRO_X_Y_Z(2,2,2, start_X + SNP_X, start_Z + SNP_Z, 2,2, 2,0,2, 2,1,2);


		for(int SNP_W = 0; SNP_W < SNP_BLOCK; SNP_W++) { 	


			// W_X_Y

			int casesContTab_inferTripletsWXY[27];		
			int controlsContTab_inferTripletsWXY[27];


			#if defined(AMPERE_80_AND) || defined(AMPERE_86_AND)
			for(int i = 0; i<SNP_CALC; i++) {
				for(int j = 0; j<SNP_CALC; j++) {
					for(int k = 0; k<SNP_CALC; k++) {

						casesContTab_inferTripletsWXY[i*9 + j*3 + k] = C_ptrGPU_cases_WXY[(start_Y + SNP_Y) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)];        
						controlsContTab_inferTripletsWXY[i*9 + j*3 + k] = C_ptrGPU_controls_WXY[(start_Y + SNP_Y) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)];    
					}
				}
			}


			// FOR WHEN USING XOR+POPC
			#else	
			for(int i = 0; i<SNP_CALC; i++) {
				for(int j = 0; j<SNP_CALC; j++) {
					for(int k = 0; k<SNP_CALC; k++) {

						casesContTab_inferTripletsWXY[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForCases[(i*3+j) * (numSNPs * numSNPs) + ( start_W + SNP_W ) * numSNPs + start_X + SNP_X] + d_output_individualSNP_popcountsForCases[k * numSNPs + start_Y + SNP_Y]) - C_ptrGPU_cases_WXY[(start_Y + SNP_Y) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)])) >> 1;        
						controlsContTab_inferTripletsWXY[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForControls[(i*3+j) * (numSNPs * numSNPs) + ( start_W + SNP_W ) * numSNPs + start_X + SNP_X] + d_output_individualSNP_popcountsForControls[k * numSNPs + start_Y + SNP_Y]) - C_ptrGPU_controls_WXY[(start_Y + SNP_Y) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)])) >> 1;    


					}
				}
			}
			#endif


			// $\{0,0,2\}$ & $\{0,0,:\} - (\{0,0,0\} + \{0,0,1\})$
			CALC_MACRO_W_X_Y(0,0,2, start_W + SNP_W, start_X + SNP_X, 0,0, 0,0,0, 0,0,1);

			// $\{0,1,2\}$ & $\{0,1,:\} - (\{0,1,0\} + \{0,1,1\})$
			CALC_MACRO_W_X_Y(0,1,2, start_W + SNP_W, start_X + SNP_X, 0,1, 0,1,0, 0,1,1);

			// $\{0,2,0\}$ & $\{0,:,0\} - (\{0,0,0\} + \{0,1,0\})$
			CALC_MACRO_W_X_Y(0,2,0, start_W + SNP_W, start_Y + SNP_Y, 0,0, 0,0,0, 0,1,0);

			// $\{0,2,1\}$ & $\{0,:,1\} - (\{0,0,1\} + \{0,1,1\})$
			CALC_MACRO_W_X_Y(0,2,1, start_W + SNP_W, start_Y + SNP_Y, 0,1, 0,0,1, 0,1,1);

			// $\{0,2,2\}$ & $\{0,:,2\} - (\{0,0,2\} + \{0,1,2\})$
			CALC_MACRO_W_X_Y(0,2,2, start_W + SNP_W, start_Y + SNP_Y, 0,2, 0,0,2, 0,1,2);

			// $\{1,0,2\}$ & $\{1,0,:\} - (\{1,0,0\} + \{1,0,1\})$
			CALC_MACRO_W_X_Y(1,0,2, start_W + SNP_W, start_X + SNP_X, 1,0, 1,0,0, 1,0,1);

			// $\{1,1,2\}$ & $\{1,1,:\} - (\{1,1,0\} + \{1,1,1\})$
			CALC_MACRO_W_X_Y(1,1,2, start_W + SNP_W, start_X + SNP_X, 1,1, 1,1,0, 1,1,1);

			// $\{1,2,0\}$ & $\{1,:,0\} - (\{1,0,0\} + \{1,1,0\})$
			CALC_MACRO_W_X_Y(1,2,0, start_W + SNP_W, start_Y + SNP_Y, 1,0, 1,0,0, 1,1,0);

			// $\{1,2,1\}$ & $\{1,:,1\} - (\{1,0,1\} + \{1,1,1\})$
			CALC_MACRO_W_X_Y(1,2,1, start_W + SNP_W, start_Y + SNP_Y, 1,1, 1,0,1, 1,1,1);

			// $\{1,2,2\}$ & $\{1,:,2\} - (\{1,0,2\} + \{1,1,2\})$
			CALC_MACRO_W_X_Y(1,2,2, start_W + SNP_W, start_Y + SNP_Y, 1,2, 1,0,2, 1,1,2);

			// $\{2,0,0\}$ & $\{:,0,0\} - (\{0,0,0\} + \{1,0,0\})$
			CALC_MACRO_W_X_Y(2,0,0, start_X + SNP_X, start_Y + SNP_Y, 0,0, 0,0,0, 1,0,0);

			// $\{2,0,1\}$ & $\{:,0,1\} - (\{0,0,1\} + \{1,0,1\})$
			CALC_MACRO_W_X_Y(2,0,1, start_X + SNP_X, start_Y + SNP_Y, 0,1, 0,0,1, 1,0,1);

			// $\{2,0,2\}$ & $\{2,0,:\} - (\{2,0,0\} + \{2,0,1\})$
			CALC_MACRO_W_X_Y(2,0,2, start_W + SNP_W, start_X + SNP_X, 2,0, 2,0,0, 2,0,1);	

			// $\{2,1,0\}$ & $\{:,1,0\} - (\{0,1,0\} + \{1,1,0\})$
			CALC_MACRO_W_X_Y(2,1,0, start_X + SNP_X, start_Y + SNP_Y, 1,0, 0,1,0, 1,1,0);

			// $\{2,1,1\}$ & $\{:,1,1\} - (\{0,1,1\} + \{1,1,1\})$
			CALC_MACRO_W_X_Y(2,1,1, start_X + SNP_X, start_Y + SNP_Y, 1,1, 0,1,1, 1,1,1);

			// $\{2,1,2\}$ & $\{2,1,:\} - (\{2,1,0\} + \{2,1,1\})$
			CALC_MACRO_W_X_Y(2,1,2, start_W + SNP_W, start_X + SNP_X, 2,1, 2,1,0, 2,1,1);

			// $\{2,2,0\}$ & $\{2,:,0\} - (\{2,0,0\} + \{2,1,0\})$
			CALC_MACRO_W_X_Y(2,2,0, start_W + SNP_W, start_Y + SNP_Y, 2,0, 2,0,0, 2,1,0);

			// $\{2,2,1\}$ & $\{2,:,1\} - (\{2,0,1\} + \{2,1,1\})$
			CALC_MACRO_W_X_Y(2,2,1, start_W + SNP_W, start_Y + SNP_Y, 2,1, 2,0,1, 2,1,1);

			// $\{2,2,2\}$ & $\{2,:,2\} - (\{2,0,2\} + \{2,1,2\})$
			CALC_MACRO_W_X_Y(2,2,2, start_W + SNP_W, start_Y + SNP_Y, 2,2, 2,0,2, 2,1,2);


			// W_X_Z

			int casesContTab_inferTriplets_WXZ[27];		
			int controlsContTab_inferTriplets_WXZ[27];			

			#if defined(AMPERE_80_AND) || defined(AMPERE_86_AND)
			for(int i = 0; i<SNP_CALC; i++) {
				for(int j = 0; j<SNP_CALC; j++) {
					for(int k = 0; k<SNP_CALC; k++) {

						casesContTab_inferTriplets_WXZ[i*9 + j*3 + k] = C_ptrGPU_cases_WXY[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)];        
						controlsContTab_inferTriplets_WXZ[i*9 + j*3 + k] = C_ptrGPU_controls_WXY[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)];    
					}
				}
			}

			// FOR WHEN USING XOR+POPC
			#else	
			for(int i = 0; i<SNP_CALC; i++) {
				for(int j = 0; j<SNP_CALC; j++) {
					for(int k = 0; k<SNP_CALC; k++) {

						casesContTab_inferTriplets_WXZ[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForCases[(i*3+j) * (numSNPs * numSNPs) + ( start_W + SNP_W ) * numSNPs + start_X + SNP_X] + d_output_individualSNP_popcountsForCases[k * numSNPs + start_Z + SNP_Z]) - C_ptrGPU_cases_WXY[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)])) >> 1;        
						controlsContTab_inferTriplets_WXZ[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForControls[(i*3+j) * (numSNPs * numSNPs) + ( start_W + SNP_W ) * numSNPs + start_X + SNP_X] + d_output_individualSNP_popcountsForControls[k * numSNPs + start_Z + SNP_Z]) - C_ptrGPU_controls_WXY[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_X * 4 + (2 * i + j)])) >> 1;    

					}
				}
			}
			#endif


			// $\{0,0,2\}$ & $\{0,0,:\} - (\{0,0,0\} + \{0,0,1\})$
			CALC_MACRO_W_X_Z(0,0,2, start_W + SNP_W, start_X + SNP_X, 0,0, 0,0,0, 0,0,1);

			// $\{0,1,2\}$ & $\{0,1,:\} - (\{0,1,0\} + \{0,1,1\})$
			CALC_MACRO_W_X_Z(0,1,2, start_W + SNP_W, start_X + SNP_X, 0,1, 0,1,0, 0,1,1);

			// $\{0,2,0\}$ & $\{0,:,0\} - (\{0,0,0\} + \{0,1,0\})$
			CALC_MACRO_W_X_Z(0,2,0, start_W + SNP_W, start_Z + SNP_Z, 0,0, 0,0,0, 0,1,0);

			// $\{0,2,1\}$ & $\{0,:,1\} - (\{0,0,1\} + \{0,1,1\})$
			CALC_MACRO_W_X_Z(0,2,1, start_W + SNP_W, start_Z + SNP_Z, 0,1, 0,0,1, 0,1,1);

			// $\{0,2,2\}$ & $\{0,:,2\} - (\{0,0,2\} + \{0,1,2\})$
			CALC_MACRO_W_X_Z(0,2,2, start_W + SNP_W, start_Z + SNP_Z, 0,2, 0,0,2, 0,1,2);

			// $\{1,0,2\}$ & $\{1,0,:\} - (\{1,0,0\} + \{1,0,1\})$
			CALC_MACRO_W_X_Z(1,0,2, start_W + SNP_W, start_X + SNP_X, 1,0, 1,0,0, 1,0,1);

			// $\{1,1,2\}$ & $\{1,1,:\} - (\{1,1,0\} + \{1,1,1\})$
			CALC_MACRO_W_X_Z(1,1,2, start_W + SNP_W, start_X + SNP_X, 1,1, 1,1,0, 1,1,1);

			// $\{1,2,0\}$ & $\{1,:,0\} - (\{1,0,0\} + \{1,1,0\})$
			CALC_MACRO_W_X_Z(1,2,0, start_W + SNP_W, start_Z + SNP_Z, 1,0, 1,0,0, 1,1,0);

			// $\{1,2,1\}$ & $\{1,:,1\} - (\{1,0,1\} + \{1,1,1\})$
			CALC_MACRO_W_X_Z(1,2,1, start_W + SNP_W, start_Z + SNP_Z, 1,1, 1,0,1, 1,1,1);

			// $\{1,2,2\}$ & $\{1,:,2\} - (\{1,0,2\} + \{1,1,2\})$
			CALC_MACRO_W_X_Z(1,2,2, start_W + SNP_W, start_Z + SNP_Z, 1,2, 1,0,2, 1,1,2);

			// $\{2,0,0\}$ & $\{:,0,0\} - (\{0,0,0\} + \{1,0,0\})$
			CALC_MACRO_W_X_Z(2,0,0, start_X + SNP_X, start_Z + SNP_Z, 0,0, 0,0,0, 1,0,0);

			// $\{2,0,1\}$ & $\{:,0,1\} - (\{0,0,1\} + \{1,0,1\})$
			CALC_MACRO_W_X_Z(2,0,1, start_X + SNP_X, start_Z + SNP_Z, 0,1, 0,0,1, 1,0,1);

			// $\{2,0,2\}$ & $\{2,0,:\} - (\{2,0,0\} + \{2,0,1\})$
			CALC_MACRO_W_X_Z(2,0,2, start_W + SNP_W, start_X + SNP_X, 2,0, 2,0,0, 2,0,1);	

			// $\{2,1,0\}$ & $\{:,1,0\} - (\{0,1,0\} + \{1,1,0\})$
			CALC_MACRO_W_X_Z(2,1,0, start_X + SNP_X, start_Z + SNP_Z, 1,0, 0,1,0, 1,1,0);

			// $\{2,1,1\}$ & $\{:,1,1\} - (\{0,1,1\} + \{1,1,1\})$
			CALC_MACRO_W_X_Z(2,1,1, start_X + SNP_X, start_Z + SNP_Z, 1,1, 0,1,1, 1,1,1);

			// $\{2,1,2\}$ & $\{2,1,:\} - (\{2,1,0\} + \{2,1,1\})$
			CALC_MACRO_W_X_Z(2,1,2, start_W + SNP_W, start_X + SNP_X, 2,1, 2,1,0, 2,1,1);

			// $\{2,2,0\}$ & $\{2,:,0\} - (\{2,0,0\} + \{2,1,0\})$
			CALC_MACRO_W_X_Z(2,2,0, start_W + SNP_W, start_Z + SNP_Z, 2,0, 2,0,0, 2,1,0);

			// $\{2,2,1\}$ & $\{2,:,1\} - (\{2,0,1\} + \{2,1,1\})$
			CALC_MACRO_W_X_Z(2,2,1, start_W + SNP_W, start_Z + SNP_Z, 2,1, 2,0,1, 2,1,1);

			// $\{2,2,2\}$ & $\{2,:,2\} - (\{2,0,2\} + \{2,1,2\})$
			CALC_MACRO_W_X_Z(2,2,2, start_W + SNP_W, start_Z + SNP_Z, 2,2, 2,0,2, 2,1,2);


			// W_Y_Z

			int casesContTab_inferTriplets_W_Y_Z[27];		
			int controlsContTab_inferTriplets_W_Y_Z[27];


			#if defined(AMPERE_80_AND) || defined(AMPERE_86_AND)
			for(int i = 0; i<SNP_CALC; i++) {
				for(int j = 0; j<SNP_CALC; j++) {
					for(int k = 0; k<SNP_CALC; k++) {

						casesContTab_inferTriplets_W_Y_Z[i*9 + j*3 + k] = C_ptrGPU_cases_WYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)];        
						controlsContTab_inferTriplets_W_Y_Z[i*9 + j*3 + k] = C_ptrGPU_controls_WYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)];    
					}
				}
			}

			// FOR WHEN USING XOR+POPC
			#else	
			for(int i = 0; i<SNP_CALC; i++) {
				for(int j = 0; j<SNP_CALC; j++) {
					for(int k = 0; k<SNP_CALC; k++) {

						casesContTab_inferTriplets_W_Y_Z[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForCases[(i*3+j) * (numSNPs * numSNPs) + ( start_W + SNP_W ) * numSNPs + start_Y + SNP_Y] + d_output_individualSNP_popcountsForCases[k * numSNPs + start_Z + SNP_Z]) - C_ptrGPU_cases_WYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)])) >> 1;        
						controlsContTab_inferTriplets_W_Y_Z[i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForControls[(i*3+j) * (numSNPs * numSNPs) + ( start_W + SNP_W ) * numSNPs + start_Y + SNP_Y] + d_output_individualSNP_popcountsForControls[k * numSNPs + start_Z + SNP_Z]) - C_ptrGPU_controls_WYZ[(start_Z + SNP_Z) * (4 * SNP_BLOCK * SNP_BLOCK) * 2  +  k * (4 * SNP_BLOCK * SNP_BLOCK)  +  SNP_W * (4 * SNP_BLOCK)  + SNP_Y * 4 + (2 * i + j)])) >> 1;    
					}
				}
			}
			#endif


			// $\{0,0,2\}$ & $\{0,0,:\} - (\{0,0,0\} + \{0,0,1\})$
			CALC_MACRO_W_Y_Z(0,0,2, start_W + SNP_W, start_Y + SNP_Y, 0,0, 0,0,0, 0,0,1);

			// $\{0,1,2\}$ & $\{0,1,:\} - (\{0,1,0\} + \{0,1,1\})$
			CALC_MACRO_W_Y_Z(0,1,2, start_W + SNP_W, start_Y + SNP_Y, 0,1, 0,1,0, 0,1,1);

			// $\{0,2,0\}$ & $\{0,:,0\} - (\{0,0,0\} + \{0,1,0\})$
			CALC_MACRO_W_Y_Z(0,2,0, start_W + SNP_W, start_Z + SNP_Z, 0,0, 0,0,0, 0,1,0);

			// $\{0,2,1\}$ & $\{0,:,1\} - (\{0,0,1\} + \{0,1,1\})$
			CALC_MACRO_W_Y_Z(0,2,1, start_W + SNP_W, start_Z + SNP_Z, 0,1, 0,0,1, 0,1,1);

			// $\{0,2,2\}$ & $\{0,:,2\} - (\{0,0,2\} + \{0,1,2\})$
			CALC_MACRO_W_Y_Z(0,2,2, start_W + SNP_W, start_Z + SNP_Z, 0,2, 0,0,2, 0,1,2);

			// $\{1,0,2\}$ & $\{1,0,:\} - (\{1,0,0\} + \{1,0,1\})$
			CALC_MACRO_W_Y_Z(1,0,2, start_W + SNP_W, start_Y + SNP_Y, 1,0, 1,0,0, 1,0,1);

			// $\{1,1,2\}$ & $\{1,1,:\} - (\{1,1,0\} + \{1,1,1\})$
			CALC_MACRO_W_Y_Z(1,1,2, start_W + SNP_W, start_Y + SNP_Y, 1,1, 1,1,0, 1,1,1);

			// $\{1,2,0\}$ & $\{1,:,0\} - (\{1,0,0\} + \{1,1,0\})$
			CALC_MACRO_W_Y_Z(1,2,0, start_W + SNP_W, start_Z + SNP_Z, 1,0, 1,0,0, 1,1,0);

			// $\{1,2,1\}$ & $\{1,:,1\} - (\{1,0,1\} + \{1,1,1\})$
			CALC_MACRO_W_Y_Z(1,2,1, start_W + SNP_W, start_Z + SNP_Z, 1,1, 1,0,1, 1,1,1);

			// $\{1,2,2\}$ & $\{1,:,2\} - (\{1,0,2\} + \{1,1,2\})$
			CALC_MACRO_W_Y_Z(1,2,2, start_W + SNP_W, start_Z + SNP_Z, 1,2, 1,0,2, 1,1,2);

			// $\{2,0,0\}$ & $\{:,0,0\} - (\{0,0,0\} + \{1,0,0\})$
			CALC_MACRO_W_Y_Z(2,0,0, start_Y + SNP_Y, start_Z + SNP_Z, 0,0, 0,0,0, 1,0,0);

			// $\{2,0,1\}$ & $\{:,0,1\} - (\{0,0,1\} + \{1,0,1\})$
			CALC_MACRO_W_Y_Z(2,0,1, start_Y + SNP_Y, start_Z + SNP_Z, 0,1, 0,0,1, 1,0,1);

			// $\{2,0,2\}$ & $\{2,0,:\} - (\{2,0,0\} + \{2,0,1\})$
			CALC_MACRO_W_Y_Z(2,0,2, start_W + SNP_W, start_Y + SNP_Y, 2,0, 2,0,0, 2,0,1);	

			// $\{2,1,0\}$ & $\{:,1,0\} - (\{0,1,0\} + \{1,1,0\})$
			CALC_MACRO_W_Y_Z(2,1,0, start_Y + SNP_Y, start_Z + SNP_Z, 1,0, 0,1,0, 1,1,0);

			// $\{2,1,1\}$ & $\{:,1,1\} - (\{0,1,1\} + \{1,1,1\})$
			CALC_MACRO_W_Y_Z(2,1,1, start_Y + SNP_Y, start_Z + SNP_Z, 1,1, 0,1,1, 1,1,1);

			// $\{2,1,2\}$ & $\{2,1,:\} - (\{2,1,0\} + \{2,1,1\})$
			CALC_MACRO_W_Y_Z(2,1,2, start_W + SNP_W, start_Y + SNP_Y, 2,1, 2,1,0, 2,1,1);

			// $\{2,2,0\}$ & $\{2,:,0\} - (\{2,0,0\} + \{2,1,0\})$
			CALC_MACRO_W_Y_Z(2,2,0, start_W + SNP_W, start_Z + SNP_Z, 2,0, 2,0,0, 2,1,0);

			// $\{2,2,1\}$ & $\{2,:,1\} - (\{2,0,1\} + \{2,1,1\})$
			CALC_MACRO_W_Y_Z(2,2,1, start_W + SNP_W, start_Z + SNP_Z, 2,1, 2,0,1, 2,1,1);

			// $\{2,2,2\}$ & $\{2,:,2\} - (\{2,0,2\} + \{2,1,2\})$
			CALC_MACRO_W_Y_Z(2,2,2, start_W + SNP_W, start_Z + SNP_Z, 2,2, 2,0,2, 2,1,2);


			int casesContTab[81];
			int controlsContTab[81];

			/* The frequency counts for the following 8 genotypes are determined from the output of the binarized tensor operations. */
			// {0,0,0}, {0,0,1}, {0,1,0}, {0,1,1}, {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}

			#if defined(AMPERE_80_AND) || defined(AMPERE_86_AND)
			for(int w = 0; w<SNP_CALC; w++) {
				for(int i = 0; i<SNP_CALC; i++) {
					for(int j = 0; j<SNP_CALC; j++) {
						for(int k = 0; k<SNP_CALC; k++) {

							casesContTab[w*27 + i*9 + j*3 + k] = C_ptrGPU_cases[

								SNP_Y * (16 * SNP_BLOCK * SNP_BLOCK * SNP_BLOCK)
									+       SNP_Z * (16 * SNP_BLOCK * SNP_BLOCK)              
									+       (2*j + k) * (SNP_BLOCK * SNP_BLOCK * 4)
									+       SNP_W * (SNP_BLOCK * 4)
									+       SNP_X * (4) 
									+       2 * w + i
							];

							controlsContTab[w*27 + i*9 + j*3 + k] = C_ptrGPU_controls[
								SNP_Y * (16 * SNP_BLOCK * SNP_BLOCK * SNP_BLOCK)
									+       SNP_Z * (16 * SNP_BLOCK * SNP_BLOCK)              
									+       (2*j + k) * (SNP_BLOCK * SNP_BLOCK * 4)
									+       SNP_W * (SNP_BLOCK * 4)
									+       SNP_X * (4) 
									+       2 * w + i
							];

						}
					}
				}
			}
			#else	
			for(int w = 0; w<SNP_CALC; w++) {
				for(int i = 0; i<SNP_CALC; i++) {
					for(int j = 0; j<SNP_CALC; j++) {
						for(int k = 0; k<SNP_CALC; k++) {

							casesContTab[w*27 + i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForCases[(w*3 + i) * (numSNPs * numSNPs) + (start_W + SNP_W) * numSNPs + (start_X + SNP_X)] + d_output_pairwiseSNP_popcountsForCases[(j*3 + k) * (numSNPs * numSNPs) + (start_Y + SNP_Y) * numSNPs + (start_Z + SNP_Z)]) - C_ptrGPU_cases[ SNP_Y * (16 * SNP_BLOCK * SNP_BLOCK * SNP_BLOCK) +       SNP_Z * (16 * SNP_BLOCK * SNP_BLOCK) +       (2*j + k) * (SNP_BLOCK * SNP_BLOCK * 4) +       SNP_W * (SNP_BLOCK * 4) +       SNP_X * (4) +       2 * w + i])) >> 1;        
							controlsContTab[w*27 + i*9 + j*3 + k] = ((int)((d_output_pairwiseSNP_popcountsForControls[(w*3 + i) * (numSNPs * numSNPs) + (start_W + SNP_W) * numSNPs + (start_X + SNP_X)] + d_output_pairwiseSNP_popcountsForControls[(j*3 + k) * (numSNPs * numSNPs) + (start_Y + SNP_Y) * numSNPs + (start_Z + SNP_Z)]) - C_ptrGPU_controls[ SNP_Y * (16 * SNP_BLOCK * SNP_BLOCK * SNP_BLOCK) +       SNP_Z * (16 * SNP_BLOCK * SNP_BLOCK) +       (2*j + k) * (SNP_BLOCK * SNP_BLOCK * 4) +       SNP_W * (SNP_BLOCK * 4) +       SNP_X * (4) +       2 * w + i])) >> 1;    


						}
					}
				}
			}
			#endif


			// Genotype counts derivation in 4-way

			// 8 genotypes

			for(int a = 0; a < 2; a++) {
				for(int b = 0; b < 2; b++) {
					for(int c = 0; c < 2; c++) {
						CALC_MACRO_A_B_C(a,b,c, SNP_W, SNP_X, start_Y + SNP_Y);
					}
				}
			}

			// 8 genotypes

			for(int a = 0; a < 2; a++) {
				for(int b = 0; b < 2; b++) {
					for(int d = 0; d < 2; d++) {
						CALC_MACRO_A_B_D(a,b,d, SNP_W, SNP_X, start_Z + SNP_Z);
					}
				}
			}

			// 8 genotypes

			for(int a = 0; a < 2; a++) {
				for(int c = 0; c < 2; c++) {
					for(int d = 0; d < 2; d++) {
						CALC_MACRO_A_C_D(a,c,d, SNP_W, SNP_Y, start_Z + SNP_Z);
					}
				}
			}

			// 8 genotypes

			for(int b = 0; b < 2; b++) {
				for(int c = 0; c < 2; c++) {
					for(int d = 0; d < 2; d++) {
						CALC_MACRO_B_C_D(b,c,d, SNP_X, SNP_Y, start_Z + SNP_Z);
					}
				}
			}


			// for when there are two 2's

			// 4 genotypes

			for(int a = 0; a < 2; a++) {
				for(int b = 0; b < 2; b++) {
					CALC_MACRO_A_B_C(a,b,2, SNP_W, SNP_X, start_Y + SNP_Y);
				}
			}

			// 4 genotypes

			for(int a = 0; a < 2; a++) {
				for(int c = 0; c < 2; c++) {
					CALC_MACRO_A_B_C(a,2,c, SNP_W, SNP_X, start_Y + SNP_Y);
				}
			}

			// 4 genotypes

			for(int b = 0; b < 2; b++) {
				for(int c = 0; c < 2; c++) {
					CALC_MACRO_A_B_C(2,b,c, SNP_W, SNP_X, start_Y + SNP_Y);
				}
			}

			// 4 genotypes

			for(int a = 0; a < 2; a++) {
				for(int d = 0; d < 2; d++) {
					CALC_MACRO_A_B_D(a,2,d, SNP_W, SNP_X, start_Z);
				}
			}

			// 4 genotypes

			for(int b = 0; b < 2; b++) {
				for(int d = 0; d < 2; d++) {
					CALC_MACRO_A_B_D(2,b,d, SNP_W, SNP_X, start_Z + SNP_Z);
				}
			}

			// 4 genotypes

			for(int c = 0; c < 2; c++) {
				for(int d = 0; d < 2; d++) {
					CALC_MACRO_A_C_D(2,c,d, SNP_W, SNP_Y, start_Z + SNP_Z);
				}
			}


			// for when there are three 2's


			for(int a = 0; a < 2; a++) {
				CALC_MACRO_A_B_C(a,2,2, SNP_W, SNP_X, start_Y + SNP_Y);
			}

			for(int b = 0; b < 2; b++) {
				CALC_MACRO_A_B_C(2,b,2, SNP_W, SNP_X, start_Y + SNP_Y);
			}

			for(int c = 0; c < 2; c++) {
				CALC_MACRO_A_B_C(2,2,c, SNP_W, SNP_X, start_Y + SNP_Y);
			}

			for(int d = 0; d < 2; d++) {
				CALC_MACRO_B_C_D(2,2,d, SNP_X, SNP_Y, start_Z + SNP_Z);
			}


			CALC_MACRO_A_B_C(2,2,2, SNP_W, SNP_X, start_Y + SNP_Y);


			float score_new = scoreQuadsK2(casesContTab, controlsContTab, tablePrecalc, numCases, numControls);


			if ((score_new <= score) && ((start_W + SNP_W) < (start_X + SNP_X)) && ((start_X + SNP_X) < (start_Y + SNP_Y)) && ((start_Y + SNP_Y) < (start_Z + SNP_Z))) {	
				SNP_W_withBestScore = (start_W + SNP_W);
				score = score_new;
			}
		}
	}

	float min_score =  blockReduceMin_2D(score);	
	if(local_id == 0 && threadIdx.y == 0) {		
		atomicMin_g_f(output, min_score);
	}


	if(score == min_score) {
		if((SNP_W_withBestScore < (start_Y + SNP_Y)) && ((start_X + SNP_X) < (start_Y + SNP_Y)) &&  ((start_Y + SNP_Y) < (start_Z + SNP_Z))) {	
			unsigned long long int packedIndices = (((unsigned long long int)SNP_W_withBestScore) << 0) | (((unsigned long long int)(start_X + SNP_X)) << 16) | (((unsigned long long int)(start_Y + SNP_Y)) << 32) | (((unsigned long long int)(start_Z + SNP_Z)) << 48);
			atomicMinGetIndex(output, min_score, output_packedIndices, packedIndices);
		}
	}

}



