#pragma once
#include "LayerCalculation.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

class BackPropagation {
public:
	void initializeConstants(float learningRate, float momentum, float weightDecay, int batchSize);
	void initializeVariables(ResultVariables res_Variables, LayerCalculation l_calculation);
	void updateLearningRate(float learningRate);

	int results_ErrorCalculation(dim3* kernelSizes);
	int poolingFCLayer_ErrorCalculation(int layerNum, dim3* kernelSizes);

private:
	LayerCalculation layerCalculation;
	ResultVariables resultVariables;
};