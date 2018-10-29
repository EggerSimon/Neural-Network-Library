#include "ResultVariables.h"

//Checks if an Error has occured
int CheckCudaError2(cudaError_t CudaError)
{
	if (CudaError != cudaSuccess)
	{
		return 1;
	}

	return 0;
}

int ResultVariables::allocateGPUMemory(int inputDimensions[], int hiddenDimensions[], int poolingLayers[], int batchSize, int layers, bool cudnn)
{
	filterMatrix.h_HiddenDimensions = new int[(layers + 1) * 4];
	filterMatrix.h_InputDimensions = new int[2];
	useCudNN = cudnn;

	for (int i = 0; i < (layers + 1) * 4; i++)
	{
		filterMatrix.h_HiddenDimensions[i] = hiddenDimensions[i];

		if (i < 2)
		{
			filterMatrix.h_InputDimensions[i] = inputDimensions[i];
		}
	}

	filterMatrix.layers = layers;
	filterMatrix.batchSize = batchSize;
	filterMatrix.poolingLayers = new int[layers * 3];

	resVariables = new ResultVariables::LayerResultVariables[batchSize];

	cudaMalloc((void **)&categoryCounts, batchSize * sizeof(int));

	//Copies the given ArraySize from Host to Device, to make calculations on device code possible
	cudaMalloc((void **)&filterMatrix.d_HiddenDimensions, (layers + 1) * 4 * sizeof(int));
	cudaMemcpy(filterMatrix.d_HiddenDimensions, hiddenDimensions, (layers + 1) * 4 * sizeof(int), cudaMemcpyHostToDevice);

	//Calculates the needed Space for the LayerResults and TotalErrors
	int ArraySizeCount = 0;
	int PoolingSizeCount = 0;
	int PoolingCounter = 0;
	for (int i = 0; i < layers + 1; i++)
	{
		ArraySizeCount += hiddenDimensions[i * 4 + 1] * pow(hiddenDimensions[i * 4], 2);
		if (i < layers)
		{
			if (poolingLayers[i * 3] == 1)
			{
				PoolingCounter++;
				PoolingSizeCount += hiddenDimensions[i * 4 + 1] * pow(hiddenDimensions[i * 4], 2) / pow(poolingLayers[i * 3 + 1], 2);
			}

			for (int j = 0; j < 3; j++)
			{
				filterMatrix.poolingLayers[i * 3 + j] = poolingLayers[i * 3 + j];
			}
		}
	}

	//Allocates the Variables
	cudaMalloc((void **)&d_ImageParts, inputDimensions[1] * batchSize * pow(inputDimensions[0], 2) * sizeof(float));
	cudaMalloc((void **)&d_SoftMaxResults, batchSize * hiddenDimensions[layers * 4 + 1] * pow(hiddenDimensions[layers * 4], 2) * sizeof(float));
	cudaMalloc((void **)&d_LayerOutputs, batchSize * ArraySizeCount * sizeof(float));
	cudaMalloc((void **)&d_LayerResults, batchSize * ArraySizeCount * sizeof(float));
	cudaMalloc((void **)&d_TotalErrors, batchSize * ArraySizeCount * sizeof(float));
	cudaMalloc((void **)&d_ActivationErrors, batchSize * ArraySizeCount * sizeof(float));

	cudaMalloc((void **)&d_PoolingResults, batchSize * PoolingSizeCount * sizeof(float));
	cudaMalloc((void **)&d_PoolingErrors, batchSize * PoolingSizeCount * sizeof(float));

	cudaMallocHost((void**)&h_Results, hiddenDimensions[4 * layers + 1] * (int)pow(hiddenDimensions[4 * layers], 2) * sizeof(float));

	//Allocates MemorySpace for the temporary Results
	for (int i = 0; i < batchSize; i++)
	{
		//Temporary Space is needed for the Cuda-Reduction Method
		cudaMalloc((void **)&resVariables[i].d_TempResults, 5 * hiddenDimensions[4 * layers + 1] * (int)pow(hiddenDimensions[4 * layers], 2) * sizeof(float));
	}
	//Allocates Memoryspace for the FilterMatrices and Bias
	filterMatrix.d_FilterMatrices = new float*[layers + 1];
	filterMatrix.d_WeightMomentum = new float*[layers + 1];
	filterMatrix.d_Bias = new float*[layers + 1];
	filterMatrix.d_PRelu = new float*[layers + 1];
	filterMatrix.d_BiasMomentum = new float*[layers + 1];

	cudaMalloc((void **)&filterMatrix.d_FilterMatrices[0], hiddenDimensions[1] * inputDimensions[1] * pow(hiddenDimensions[2], 2) * sizeof(float));
	cudaMalloc((void **)&filterMatrix.d_WeightMomentum[0], hiddenDimensions[1] * inputDimensions[1] * pow(hiddenDimensions[2], 2) * sizeof(float));
	cudaMemset(filterMatrix.d_WeightMomentum[0], 0, hiddenDimensions[1] * inputDimensions[1] * pow(hiddenDimensions[2], 2) * sizeof(float));
	cudaMalloc((void **)&filterMatrix.d_Bias[0], hiddenDimensions[1] * sizeof(float));
	cudaMalloc((void **)&filterMatrix.d_PRelu[0], hiddenDimensions[1] * sizeof(float));
	cudaMalloc((void **)&filterMatrix.d_BiasMomentum[0], hiddenDimensions[1] * sizeof(float));
	cudaMemset(filterMatrix.d_BiasMomentum[0], 0, hiddenDimensions[1] * sizeof(float));

	for (int i = 1; i < layers + 1; i++)
	{
		cudaMalloc((void **)&filterMatrix.d_FilterMatrices[i], hiddenDimensions[i * 4 + 1] * hiddenDimensions[(i - 1) * 4 + 1] * pow(hiddenDimensions[i * 4 + 2], 2) * sizeof(float));
		cudaMalloc((void **)&filterMatrix.d_WeightMomentum[i], hiddenDimensions[i * 4 + 1] * hiddenDimensions[(i - 1) * 4 + 1] * pow(hiddenDimensions[i * 4 + 2], 2) * sizeof(float));
		cudaMemset(filterMatrix.d_WeightMomentum[i], 0, hiddenDimensions[i * 4 + 1] * hiddenDimensions[(i - 1) * 4 + 1] * pow(hiddenDimensions[i * 4 + 2], 2) * sizeof(float));
		cudaMalloc((void **)&filterMatrix.d_Bias[i], hiddenDimensions[i * 4 + 1] * sizeof(float));
		cudaMalloc((void **)&filterMatrix.d_PRelu[i], hiddenDimensions[i * 4 + 1] * sizeof(float));
		cudaMalloc((void **)&filterMatrix.d_BiasMomentum[i], hiddenDimensions[i * 4 + 1] * sizeof(float));
		cudaMemset(filterMatrix.d_BiasMomentum[i], 0, hiddenDimensions[i * 4 + 1] * sizeof(float));
	}

	if (useCudNN)
	{
		setUpCudNN(PoolingCounter);
	}

	//Checks if the MemorySpace has been successfully allocated
	if (CheckCudaError2(cudaGetLastError()) == 1)
	{
		return 1;
	}

	return 0;
}

void ResultVariables::setUpCudNN(int poolingLayers)
{
	//Create Cudnn handle and declaration of Cudnn descriptors
	cudnnCreate(&cudnn);
	cublasCreate(&cublas);
	layer_descriptor = new cudnnTensorDescriptor_t[filterMatrix.layers + 2];
	pooling_descriptor = new cudnnPoolingDescriptor_t[filterMatrix.layers + 1];
	poolinglayer_descriptor = new cudnnTensorDescriptor_t[filterMatrix.layers + 1];
	convolution_descriptor = new cudnnConvolutionDescriptor_t[filterMatrix.layers + 1];
	filter_descriptor = new cudnnFilterDescriptor_t[filterMatrix.layers + 1];
	bias_descriptor = new cudnnTensorDescriptor_t[filterMatrix.layers + 1];
	activation_descriptor = new cudnnActivationDescriptor_t[filterMatrix.layers + 1];
	convolution_algorithm = new cudnnConvolutionFwdAlgo_t[filterMatrix.layers + 1];
	bwd_convolution_algorithm = new cudnnConvolutionBwdDataAlgo_t[filterMatrix.layers + 1];
	bwd_filter_algorithm = new cudnnConvolutionBwdFilterAlgo_t[filterMatrix.layers + 1];
	d_Workspace = new void*[filterMatrix.layers + 1];
	workspace_bytes = new size_t[filterMatrix.layers + 1];
	d_BwdWorkspace = new void*[filterMatrix.layers + 1];
	bwdWorkspace_bytes = new size_t[filterMatrix.layers + 1];
	d_BwdFilterWorkspace = new void*[filterMatrix.layers + 1];
	bwdfilterWorkspace_bytes = new size_t[filterMatrix.layers + 1];

	//Creates tensor of the image data and the first filter descriptor
	cudnnCreateTensorDescriptor(&layer_descriptor[0]);
	cudnnSetTensor4dDescriptor(layer_descriptor[0], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, filterMatrix.batchSize, filterMatrix.h_InputDimensions[1],
		filterMatrix.h_InputDimensions[0], filterMatrix.h_InputDimensions[0]);
	cudnnCreateFilterDescriptor(&filter_descriptor[0]);
	cudnnSetFilter4dDescriptor(filter_descriptor[0], CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterMatrix.h_HiddenDimensions[1], filterMatrix.h_InputDimensions[1],
		filterMatrix.h_HiddenDimensions[2], filterMatrix.h_HiddenDimensions[2]);

	//Creates tensors, filter, convolution and activation descriptors for the cudnn lib
	for (int i = 0; i < filterMatrix.layers + 1; i++)
	{
		int padding = 1;
		if (filterMatrix.h_HiddenDimensions[i * 4 + 3] == 0)
		{
			padding = 0;
		}
		cudnnCreateConvolutionDescriptor(&convolution_descriptor[i]);
		if (i == filterMatrix.layers)
		{
			cudnnSetConvolution2dDescriptor(convolution_descriptor[i], padding, padding, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
		}
		else
		{
			cudnnSetConvolution2dDescriptor(convolution_descriptor[i], padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		}
		cudnnCreateActivationDescriptor(&activation_descriptor[i]);
		cudnnSetActivationDescriptor(activation_descriptor[i], CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);
		cudnnCreateTensorDescriptor(&layer_descriptor[i + 1]);
		cudnnSetTensor4dDescriptor(layer_descriptor[i + 1], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, filterMatrix.batchSize, filterMatrix.h_HiddenDimensions[i * 4 + 1],
			filterMatrix.h_HiddenDimensions[i * 4], filterMatrix.h_HiddenDimensions[i * 4]);
		cudnnCreateTensorDescriptor(&bias_descriptor[i]);
		cudnnSetTensor4dDescriptor(bias_descriptor[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filterMatrix.h_HiddenDimensions[i * 4 + 1], 1, 1);
		if (i < filterMatrix.layers)
		{
			cudnnCreateFilterDescriptor(&filter_descriptor[i + 1]);
			cudnnSetFilter4dDescriptor(filter_descriptor[i + 1], CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterMatrix.h_HiddenDimensions[(i + 1) * 4 + 1],
				filterMatrix.h_HiddenDimensions[i * 4 + 1], filterMatrix.h_HiddenDimensions[(i + 1) * 4 + 2], filterMatrix.h_HiddenDimensions[(i + 1) * 4 + 2]);

			if (filterMatrix.poolingLayers[i * 3] == 1)
			{
				cudnnCreatePoolingDescriptor(&pooling_descriptor[i]);
				cudnnSetPooling2dDescriptor(pooling_descriptor[i], CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, 2, 0, 0, filterMatrix.poolingLayers[i * 3 + 2],
					filterMatrix.poolingLayers[i * 3 + 2]);
				cudnnCreateTensorDescriptor(&poolinglayer_descriptor[i]);
				cudnnSetTensor4dDescriptor(poolinglayer_descriptor[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, filterMatrix.batchSize, filterMatrix.h_HiddenDimensions[i * 4 + 1],
					filterMatrix.h_HiddenDimensions[i * 4] / filterMatrix.poolingLayers[i * 3 + 1], filterMatrix.h_HiddenDimensions[i * 4] / filterMatrix.poolingLayers[i * 3 + 1]);
			}
		}

		if (i > 0)
		{
			if (filterMatrix.poolingLayers[(i - 1) * 3] == 1)
			{
				cudnnGetConvolutionForwardAlgorithm(cudnn, poolinglayer_descriptor[i - 1], filter_descriptor[i], convolution_descriptor[i], layer_descriptor[i + 1],
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm[i]);
				cudnnStatus_t  s = cudnnGetConvolutionForwardWorkspaceSize(cudnn, poolinglayer_descriptor[i - 1], filter_descriptor[i], convolution_descriptor[i],
					layer_descriptor[i + 1], convolution_algorithm[i], &workspace_bytes[i]);
				d_Workspace[i] = new void*{ nullptr };
				cudaMalloc(&d_Workspace[i], workspace_bytes[i]);

				s = cudnnGetConvolutionBackwardDataAlgorithm(cudnn, filter_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], poolinglayer_descriptor[i - 1],
					CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_convolution_algorithm[i]);
				s = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filter_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], poolinglayer_descriptor[i - 1],
					bwd_convolution_algorithm[i], &bwdWorkspace_bytes[i]);
				d_BwdWorkspace[i] = new void*{ nullptr };
				cudaMalloc(&d_BwdWorkspace[i], bwdWorkspace_bytes[i]);

				s = cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, poolinglayer_descriptor[i - 1], layer_descriptor[i + 1], convolution_descriptor[i], filter_descriptor[i],
					CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_filter_algorithm[i]);
				s = cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, poolinglayer_descriptor[i - 1], layer_descriptor[i + 1], convolution_descriptor[i], filter_descriptor[i],
					bwd_filter_algorithm[i], &bwdfilterWorkspace_bytes[i]);
				d_BwdFilterWorkspace[i] = new void*{ nullptr };
				cudaMalloc(&d_BwdFilterWorkspace[i], bwdfilterWorkspace_bytes[i]);
			}
			else
			{
				cudnnGetConvolutionForwardAlgorithm(cudnn, layer_descriptor[i], filter_descriptor[i], convolution_descriptor[i], layer_descriptor[i + 1],
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm[i]);
				cudnnStatus_t  s = cudnnGetConvolutionForwardWorkspaceSize(cudnn, layer_descriptor[i], filter_descriptor[i], convolution_descriptor[i], layer_descriptor[i + 1],
					convolution_algorithm[i], &workspace_bytes[i]);
				d_Workspace[i] = new void*{ nullptr };
				cudaMalloc(&d_Workspace[i], workspace_bytes[i]);

				s = cudnnGetConvolutionBackwardDataAlgorithm(cudnn, filter_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], layer_descriptor[i], 
					CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_convolution_algorithm[i]);
				s = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filter_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], layer_descriptor[i],
					bwd_convolution_algorithm[i], &bwdWorkspace_bytes[i]);
				d_BwdWorkspace[i] = new void*{ nullptr };
				cudaMalloc(&d_BwdWorkspace[i], bwdWorkspace_bytes[i]);

				s = cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, layer_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], filter_descriptor[i],
					CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_filter_algorithm[i]);
				s = cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, layer_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], filter_descriptor[i],
					bwd_filter_algorithm[i], &bwdfilterWorkspace_bytes[i]);
				d_BwdFilterWorkspace[i] = new void*{ nullptr };
				cudaMalloc(&d_BwdFilterWorkspace[i], bwdfilterWorkspace_bytes[i]);
			}

		}
		else
		{
			cudnnGetConvolutionForwardAlgorithm(cudnn, layer_descriptor[i], filter_descriptor[i], convolution_descriptor[i], layer_descriptor[i + 1],
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm[i]);
			cudnnStatus_t  s = cudnnGetConvolutionForwardWorkspaceSize(cudnn, layer_descriptor[i], filter_descriptor[i], convolution_descriptor[i], 
				layer_descriptor[i + 1], convolution_algorithm[i], &workspace_bytes[i]);
			d_Workspace[i] = new void*{ nullptr };
			cudaMalloc(&d_Workspace[i], workspace_bytes[i]);

			s = cudnnGetConvolutionBackwardDataAlgorithm(cudnn, filter_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], layer_descriptor[i],
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_convolution_algorithm[i]);
			s = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filter_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], layer_descriptor[i],
				bwd_convolution_algorithm[i], &bwdWorkspace_bytes[i]);
			d_BwdWorkspace[i] = new void*{ nullptr };
			cudaMalloc(&d_BwdWorkspace[i], bwdWorkspace_bytes[i]);

			s = cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, layer_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], filter_descriptor[i],
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_filter_algorithm[i]);
			s = cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, layer_descriptor[i], layer_descriptor[i + 1], convolution_descriptor[i], filter_descriptor[i],
				bwd_filter_algorithm[i], &bwdfilterWorkspace_bytes[i]);
			d_BwdFilterWorkspace[i] = new void*{ nullptr };
			cudaMalloc(&d_BwdFilterWorkspace[i], bwdfilterWorkspace_bytes[i]);
		}

	}
}

int ResultVariables::initializeVariables(float** filterMatrices, float** bias, float** pRelu, int layers)
{
	//Copies the constant Variables from Host to Device only once, since it is an time intensive operation
	cudaMemcpy(filterMatrix.d_FilterMatrices[0], filterMatrices[0], filterMatrix.h_HiddenDimensions[1] * filterMatrix.h_InputDimensions[1] * 
		pow(filterMatrix.h_HiddenDimensions[2], 2) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(filterMatrix.d_Bias[0], bias[0], filterMatrix.h_HiddenDimensions[1] * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(filterMatrix.d_PRelu[0], pRelu[0], filterMatrix.h_HiddenDimensions[1] * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 1; i < layers; i++)
	{
		cudaMemcpy(filterMatrix.d_FilterMatrices[i], filterMatrices[i], filterMatrix.h_HiddenDimensions[i * 4 + 1] * filterMatrix.h_HiddenDimensions[(i - 1) * 4 + 1] *
			pow(filterMatrix.h_HiddenDimensions[i * 4 + 2], 2) * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(filterMatrix.d_Bias[i], bias[i], filterMatrix.h_HiddenDimensions[i * 4 + 1] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(filterMatrix.d_PRelu[i], pRelu[i], filterMatrix.h_HiddenDimensions[i * 4 + 1] * sizeof(float), cudaMemcpyHostToDevice);
	}
	//Checks if the Variables have been successfully copied
	if (CheckCudaError2(cudaGetLastError()) == 1)
	{
		return 1;
	}

	return 0;
}

int ResultVariables::updateFilterWeights(float** FilterMatrixPointer, float** BiasMatrixPointer)
{
	int* HiddenDimensions = filterMatrix.h_HiddenDimensions;

	//Copies FilterMatrix from Device to Host-Pointer
	cudaMemcpy(FilterMatrixPointer[0], filterMatrix.d_FilterMatrices[0], 3 * HiddenDimensions[1] * pow(HiddenDimensions[2], 2) * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 1; i <= filterMatrix.layers; i++)
	{
		cudaMemcpy(FilterMatrixPointer[i], filterMatrix.d_FilterMatrices[i], HiddenDimensions[(i - 1) * 4 + 1] * HiddenDimensions[i * 4 + 1] * 
			pow(HiddenDimensions[i * 4 + 2], 2) * sizeof(float), cudaMemcpyDeviceToHost);
	}
	//Copies Bias from Device to Host-Pointer
	for (int i = 0; i <= filterMatrix.layers; i++)
	{
		cudaMemcpy(BiasMatrixPointer[i], filterMatrix.d_Bias[i], HiddenDimensions[i * 4 + 1] * sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaError_t error = cudaGetLastError();

	return 0;
}

int ResultVariables::freeMemorySpace()
{
	//Frees Memory
	for (int i = 0; i < filterMatrix.batchSize; i++)
	{
		cudaFree(resVariables[i].d_TempResults);
	}

	cudaFree(d_ImageParts);
	cudaFree(d_LayerResults);
	cudaFree(d_SoftMaxResults);
	cudaFree(d_PoolingResults);
	cudaFree(d_TotalErrors);
	cudaFree(filterMatrix.d_HiddenDimensions);
	cudaFree(filterMatrix.d_Bias);
	cudaFree(filterMatrix.d_FilterMatrices);
	cudaFree(filterMatrix.d_WeightMomentum);
	cudaFree(filterMatrix.d_BiasMomentum);

	cudaFreeHost(h_Results);

	if (useCudNN)
	{
		for (int i = 0; i < filterMatrix.layers + 1; i++)
		{
			cudnnDestroyTensorDescriptor(layer_descriptor[i]);
			cudnnDestroyActivationDescriptor(activation_descriptor[i]);
			cudnnDestroyFilterDescriptor(filter_descriptor[i]);
			cudnnDestroyConvolutionDescriptor(convolution_descriptor[i]);

			if (filterMatrix.poolingLayers[i * 3] == 1)
			{
				cudnnDestroyPoolingDescriptor(pooling_descriptor[i]);
			}
		}

		cudnnDestroyTensorDescriptor(layer_descriptor[filterMatrix.layers + 1]);
		cudnnDestroy(cudnn);
	}

	return 0;
}
