#pragma once
#include "cudnn.h";
#include "cublas_v2.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

#pragma comment(lib,"cublas.lib")

class ResultVariables {
public:
	int allocateGPUMemory(int inputDimensions[], int hiddenDimensions[], int poolingLayers[], int batchSize, int layers, bool useCudNN);
	int initializeVariables(float** filterMatrices, float** bias, float** pRelu, int layers);
	int updateFilterWeights(float** filterMatrixPointer, float** biasMatrixPointer);
	int freeMemorySpace();

	float* d_LayerOutputs;
	float* d_LayerResults;
	float* d_SoftMaxResults;
	float* d_TotalErrors;
	float* d_ActivationErrors;
	float* d_PoolingResults;
	float* d_PoolingErrors;
	float* d_ImageParts;
	int* categoryCounts;
	float* h_Results;

	bool useCudNN;
	size_t* workspace_bytes;
	size_t* bwdWorkspace_bytes;
	size_t* bwdfilterWorkspace_bytes;

	cudnnHandle_t cudnn;
	cudnnTensorDescriptor_t* layer_descriptor;
	cudnnTensorDescriptor_t* poolinglayer_descriptor;
	cudnnPoolingDescriptor_t* pooling_descriptor;
	cudnnConvolutionDescriptor_t* convolution_descriptor;
	cudnnFilterDescriptor_t* filter_descriptor;
	cudnnTensorDescriptor_t* bias_descriptor;
	cudnnActivationDescriptor_t* activation_descriptor;
	cudnnConvolutionFwdAlgo_t* convolution_algorithm;
	cudnnConvolutionBwdDataAlgo_t* bwd_convolution_algorithm;
	cudnnConvolutionBwdFilterAlgo_t* bwd_filter_algorithm;

	cublasHandle_t cublas;

	void** d_Workspace;
	void** d_BwdWorkspace;
	void** d_BwdFilterWorkspace;

	struct FilterMatrix {
		float** d_FilterMatrices;
		float** d_WeightMomentum;
		float** d_Bias;
		float** d_PRelu;
		float** d_BiasMomentum;
		int* h_InputDimensions;
		int* d_HiddenDimensions;
		int* h_HiddenDimensions;
		int* poolingLayers;
		int layers;
		int batchSize;

		float** layerResultsPointer;
	};
	struct LayerResultVariables {
		float* d_TempResults;
	};

	FilterMatrix filterMatrix;
	LayerResultVariables* resVariables;

private:
	void setUpCudNN(int PoolingLayers);
};