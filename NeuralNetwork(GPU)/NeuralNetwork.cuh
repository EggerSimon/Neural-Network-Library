#pragma once
#pragma comment(lib,"cublas.lib")

#include "LayerCalculation.cuh"
#include "BackPropagation.cuh"
#include "Resultvariables.h"
#include "CudaErrors.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdio>
#include <ctime>

class CudaNeuralNetwork {

public:
	CudaNeuralNetwork(int batchSizes, int inputDimensions[], int hiddenDimensions[], int poolingLayers[], int layers, bool useCudNN);

	float* runNeuralNetwork(float input[], int categoryCount, int batchCount, bool backPropagationActive);
	void updateLearningRate(float learningRate);
	void updateFilterWeights(float** filterMatrixPointers, float** biasMatrixPointer);
	void initializeVariables(float** filterMatrices, float** bias, float** pRelu, int layers, float learningRate, float momentum, float weightDecay, int batchSize);
	void freeMemorySpace();

private:
	BackPropagation backPropagation;
	LayerCalculation layerCalculation;
	ResultVariables variables;
	CudaErrors cudaErrors;
	
	float* backProp_variables;
	dim3* kernelSizes;
	int category = 0;

	float constCalculator(float l[]);
	void setKernelSizes(int size);
};
