#pragma once
#include "ResultVariables.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

class LayerCalculation {

public:
	int cudaSoftMaxCalculation(float* results, float* softMaxResults, int* h_ArraySizes, int* poolingLayers, int batchCount, int layerNum, int softMaxOffset, dim3 kernelSizes[]);

	int offsetCalculation(int *arraySizes, int* poolingLayers, int batchCount, int layerNum, int layers);
	int poolingOffsetCalculation(int *arraySizes, int* poolingLayers, int batchCount, int layerNum);

	int batchDim;
	int poolingBatchDim;
};