#include "BackPropagation.cuh"

int CheckCudaError1(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		return 1;
	}

	return 0;
}

struct Variables
{
	float learningRate;
	float momentum;
	float weightDecay;
	int batchSize;
};

Variables variables;

//Sets Constant values after userinput
void BackPropagation::initializeConstants(float learningRate, float momentum, float weightDecay, int batchSize)
{
	variables.learningRate = learningRate;
	variables.momentum = momentum;
	variables.weightDecay = weightDecay;
	variables.batchSize = batchSize;
}

void BackPropagation::updateLearningRate(float learningRate)
{
	variables.learningRate = learningRate;
}

//Sets Values of Image depending Variables
void BackPropagation::initializeVariables(ResultVariables res_Variables, LayerCalculation l_Calculation)
{
	resultVariables = res_Variables;
	layerCalculation = l_Calculation;
}

//Calculates Error of each Result value
__global__
void cudaResult_ErrorCalculation(int* categoryCount, float* results, float* softMaxResults, float* newTotalError, float* pRelu, int offset, int batchDim)
{
	int d_n = threadIdx.x;
	int BatchCount = blockIdx.x;
	int CategorySize = blockDim.x;

	//if the Category is the correctly labeled Image
	if (d_n == categoryCount[BatchCount])
	{
		//Result aims to be 1
		newTotalError[BatchCount * batchDim + offset + d_n] = softMaxResults[BatchCount * CategorySize + d_n] - 1;
		//NewTotalError[BatchCount * BatchDim + offset + d_n] = Results[BatchCount * BatchDim + offset + d_n] - 1;
	}
	else
	{
		//Result aims to be 0
		newTotalError[BatchCount * batchDim + offset + d_n] = softMaxResults[BatchCount * CategorySize + d_n];
		//NewTotalError[BatchCount * BatchDim + offset + d_n] = Results[BatchCount * BatchDim + offset + d_n];
	}

	if (results[BatchCount * batchDim + offset + d_n] == 0)
	{
		newTotalError[BatchCount * batchDim + offset + d_n] = 0;
	}
}

int BackPropagation::results_ErrorCalculation(dim3* kernelSizes)
{
	//Calculates offset of ErrorArray
	int offset = layerCalculation.offsetCalculation(resultVariables.filterMatrix.h_HiddenDimensions, resultVariables.filterMatrix.poolingLayers, 0, resultVariables.filterMatrix.layers + 1, 0);
	//Executes Kernel
	cudaResult_ErrorCalculation << <kernelSizes[0], kernelSizes[1] >> > (resultVariables.categoryCounts, resultVariables.d_LayerResults, resultVariables.d_SoftMaxResults, resultVariables.d_TotalErrors,
		resultVariables.filterMatrix.d_PRelu[resultVariables.filterMatrix.layers], offset, layerCalculation.batchDim);
	//Checks for Errors
	cudaError_t error = cudaGetLastError();
	return CheckCudaError1(error);
}

//Calculates Errors for the Fully-Connected-Layer
__global__
void cudaPoolingFCLayer_ErrorCalculation(float* filterMatrix, float* totalError, float* layerResults, float* poolingResults, float* pRelu, int layerNum, 
	int* arraySizes, int poolingDim, int l_offset, int lasterror_offset, int p_offset, int batchDim, int poolingBatchDim)
{
	//LayerDepht
	register int LD = arraySizes[layerNum * 4 + 1];
	//ResultDepht
	register int RD = arraySizes[(layerNum + 1) * 4 + 1];
	//LayerWidth
	register int LW = arraySizes[layerNum * 4];
	register int PW = LW / poolingDim;
	//Batchoffset
	register int BO = LD * __powf(LW, 2);

	register int d_n = blockIdx.x * blockDim.x + threadIdx.x;

	register int BatchCount = d_n / BO;
	//Image of the Layer (Depht)
	register int i = (d_n - BatchCount * BO) / __powf(LW, 2);
	//Inputpixel of the above Layer (Width)
	register int j = (d_n - BatchCount * BO - i * __powf(LW, 2)) / LW;
	//Inputpixel of the above Layer (Height)
	register int k = d_n - BatchCount * BO - i * __powf(LW, 2) - j * LW;

	register int pj = j / 2;
	register int pk = k / 2;

	//Sets Error equal to 0, in case it has a value from the last image
	totalError[batchDim * BatchCount + l_offset + i * (int)__powf(LW, 2) + j * LW + k] = 0;

	//only if the pixelvalue is bigger or equal to 0, because of the Derivative of the Relu-Activationfunction
	if (layerResults[batchDim * BatchCount + l_offset + i * (int)__powf(LW, 2) + j * LW + k] == poolingResults[poolingBatchDim * BatchCount + p_offset + i * (int)__powf(PW, 2) + pj * PW + pk])
	{
		//For each Result
		for (int l = 0; l < arraySizes[(layerNum + 1) * 4 + 1]; l++)
		{
			totalError[batchDim * BatchCount + l_offset + i * (int)__powf(LW, 2) + j * LW + k] += totalError[batchDim * BatchCount + lasterror_offset + l] *
				filterMatrix[(i * RD + l) * (int)__powf(PW, 2) + pj * PW + pk];
		}
	}
	if (poolingResults[poolingBatchDim * BatchCount + p_offset + i * (int)__powf(PW, 2) + pj * PW + pk] == 0)
	{
		totalError[batchDim * BatchCount + l_offset + i * (int)__powf(LW, 2) + j * LW + k] *= 0;
	}
}

int BackPropagation::poolingFCLayer_ErrorCalculation(int layerNum, dim3* kernelSizes)
{
	//Calculates position of wanted values in the array
	int l_offset = layerCalculation.offsetCalculation(resultVariables.filterMatrix.h_HiddenDimensions, resultVariables.filterMatrix.poolingLayers, 0, layerNum + 1, 0);
	int p_offset = layerCalculation.poolingOffsetCalculation(resultVariables.filterMatrix.h_HiddenDimensions, resultVariables.filterMatrix.poolingLayers, 0, layerNum + 1);
	int lasterror_offset = layerCalculation.offsetCalculation(resultVariables.filterMatrix.h_HiddenDimensions, resultVariables.filterMatrix.poolingLayers, 0, layerNum + 2, 0);
	//Executes Kernel
	cudaPoolingFCLayer_ErrorCalculation << <kernelSizes[0], kernelSizes[1] >> > (resultVariables.filterMatrix.d_FilterMatrices[layerNum + 1], resultVariables.d_TotalErrors,
		resultVariables.d_LayerResults,	resultVariables.d_PoolingResults, resultVariables.filterMatrix.d_PRelu[layerNum], layerNum, resultVariables.filterMatrix.d_HiddenDimensions,
		resultVariables.filterMatrix.poolingLayers[layerNum * 3 + 1], l_offset, lasterror_offset, p_offset, layerCalculation.batchDim, layerCalculation.poolingBatchDim);
	//Checks for Errors
	cudaError_t error = cudaGetLastError();
	return CheckCudaError1(error);
}
