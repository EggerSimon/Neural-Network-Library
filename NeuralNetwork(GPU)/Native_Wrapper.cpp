#include "Native_Wrapper.h"

extern "C" LIBRARY_EXPORT CudaNeuralNetwork* new_CudaNeuralNetwork(int batchSizes, int inputDimensions[], int hiddenDimensions[], int poolingLayers[], int layers, bool useCudNN)
{
	return new CudaNeuralNetwork(batchSizes, inputDimensions, hiddenDimensions, poolingLayers, layers, useCudNN);
}

extern "C" LIBRARY_EXPORT void delete_NeuralNetwork(CudaNeuralNetwork* instance)
{
	delete instance;
}

extern "C" LIBRARY_EXPORT void updateLearningRate(CudaNeuralNetwork* instance, float learningRate)
{
	instance->updateLearningRate(learningRate);
}

extern "C" LIBRARY_EXPORT float* runNeuralNetwork(CudaNeuralNetwork* instance, float input[], int categoryCount, int batchCount, bool backPropagationActive)
{
	float* returnValue = instance->runNeuralNetwork(input, categoryCount, batchCount, backPropagationActive);

	return returnValue;
}

extern "C" LIBRARY_EXPORT void initializeVariables(CudaNeuralNetwork* instance, float** filterMatrices, float** bias, float** pRelu, int layers, float learningRate,
	float momentum, float weightDecay, int batchSize)
{
	instance->initializeVariables(filterMatrices, bias, pRelu, layers, learningRate, momentum, weightDecay, batchSize);
}

extern "C" LIBRARY_EXPORT void freeMemorySpace(CudaNeuralNetwork* instance)
{
	instance->freeMemorySpace();
}

extern "C" LIBRARY_EXPORT void updateFilterWeights(CudaNeuralNetwork* instance, float** filterMatrixPointer, float** biasMatrixPointer)
{
	instance->updateFilterWeights(filterMatrixPointer, biasMatrixPointer);
}

