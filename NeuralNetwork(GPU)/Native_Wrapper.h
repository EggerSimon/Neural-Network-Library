#pragma once

#include "NeuralNetwork.cuh"

#define LIBRARY_EXPORT __declspec(dllexport)

extern "C" LIBRARY_EXPORT CudaNeuralNetwork* new_CudaNeuralNetwork(int BatchSizes, int InputDimensions[], int HiddenDimensions[], int PoolingLayers[], int Layers, bool useCudNN);

extern "C" LIBRARY_EXPORT void delete_NeuralNetwork(CudaNeuralNetwork* instance);

extern "C" LIBRARY_EXPORT void updateLearningRate(CudaNeuralNetwork* instance, float LearningRate);

extern "C" LIBRARY_EXPORT float* runNeuralNetwork(CudaNeuralNetwork* instance, float input[], int categoryCount, int batchCount, bool backPropagationActive);

extern "C" LIBRARY_EXPORT void initializeVariables(CudaNeuralNetwork* instance, float** FilterMatrices, float** Bias, float** PRelu, int Layers, float LearningRate, 
	float Momentum, float WeightDecay, int BatchSize);

extern "C" LIBRARY_EXPORT void freeMemorySpace(CudaNeuralNetwork* instance);

extern "C" LIBRARY_EXPORT void updateFilterWeights(CudaNeuralNetwork* instance, float** FilterMatrixPointer, float** BiasMatrixPointer);

