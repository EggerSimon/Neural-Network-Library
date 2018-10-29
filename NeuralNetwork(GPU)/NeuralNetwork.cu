#include "NeuralNetwork.cuh"

/*ArraySizes
0: Layer1Depht;
1: Layer1Dimension (eg. 32x32) = ImageSize
2: ConvMatrixDimension1 (eg. 3x3)
3: Layer2Depht;
.
.
.
3 * (n-1): LayerNDepht
3 * (n-1) + 1: LayerNDimension
3 * (n-1) + 2: ConvMatrixDimensionN (eg. 3x3)
3 * (n-1) + 3: ResultsDepht (eg. 10)
3 * (n-1) + 3: ResultsDimension (eg. 1x1)
3 * (n-1) + 3: ResultsMatrixDimension (eg. 8x8)
*/

float CudaNeuralNetwork::constCalculator(float l[]) {
	if (l[0] > 0) {
		backProp_variables = new float[2];
		backProp_variables[0] = l[0];
		backProp_variables[1] = l[1];
	}
	else if (l[0] == -1) {
		if (category == 6) {
			return backProp_variables[0] * 100;
		}
		return backProp_variables[0];
	}
	else if (l[1] == -1) {
		return backProp_variables[1];
	}
}

CudaNeuralNetwork::CudaNeuralNetwork(int batchSize, int inputDimensions[], int hiddenDimensions[], int poolingLayers[], int layers, bool useCudNN, bool regression) {
	variables.allocateGPUMemory(inputDimensions, hiddenDimensions, poolingLayers, batchSize, layers, useCudNN, regression);
	kernelSizes = new dim3[2];
}

void CudaNeuralNetwork::freeMemorySpace() {
	variables.freeMemorySpace();
}

void CudaNeuralNetwork::updateFilterWeights(float** filterMatrixPointer, float** biasMatrixPointer) {
	int Error = variables.updateFilterWeights(filterMatrixPointer, biasMatrixPointer);
}

void CudaNeuralNetwork::updateLearningRate(float learningRate) {
	backPropagation.updateLearningRate(learningRate);
}

//Initialize FilterMatrix-Variables
void CudaNeuralNetwork::initializeVariables(float** filterMatrices, float** bias, float** pRelu, int layers, float learningRate, float momentum, float weightDecay, int batchSize) {
	backPropagation.initializeConstants(learningRate, momentum, weightDecay, batchSize);
	variables.initializeVariables(filterMatrices, bias, pRelu, layers);
	constCalculator(new float[2]{ learningRate, momentum });
}

//Calculates the needed size of the Kernels, depending on the users NeuralNetwork-Size
void CudaNeuralNetwork::setKernelSizes(int size) {
	for (int i = 256; i >= 8; i /= 2) {
		if (size % i == 0) {
			kernelSizes[0].x = size / i;
			kernelSizes[1].x = i;
			i = 7;
		}
	}
}

float* CudaNeuralNetwork::runNeuralNetwork(float input[], int categoryCount, int batchCount, bool backPropagationActive) {
	cublasStatus_t cublasStatus;
	cudnnStatus_t cudnnStatus;

	const float alpha = 1, beta = 0;
	const float learningrate = -1 * constCalculator(new float[2]{ -1,0 });
	const float momentum = constCalculator(new float[2]{ 0, -1 });
	int l_offset;
	int last_offset;
	int p_offset;

	cudaMemcpy(&variables.d_ImageParts[batchCount * variables.filterMatrix.h_InputDimensions[1] * (int)pow(variables.filterMatrix.h_InputDimensions[0], 2)],
		input, variables.filterMatrix.h_InputDimensions[1] * pow(variables.filterMatrix.h_InputDimensions[0], 2) * sizeof(float), cudaMemcpyHostToDevice);

	l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, 0, variables.filterMatrix.layers);

	cudnnStatus = cudnnConvolutionBiasActivationForward(variables.cudnn, &alpha, variables.layer_descriptor[0], variables.d_ImageParts, variables.filter_descriptor[0],
		variables.filterMatrix.d_FilterMatrices[0], variables.convolution_descriptor[0], variables.convolution_algorithm[0], variables.d_Workspace[0], variables.workspace_bytes[0],
		&beta, variables.layer_descriptor[1], &variables.d_LayerOutputs[l_offset], variables.bias_descriptor[0], variables.filterMatrix.d_Bias[0], variables.activation_descriptor[0],
		variables.layer_descriptor[1], &variables.d_LayerResults[l_offset]);
	cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Layer)");

	if (variables.filterMatrix.poolingLayers[0] == 1) {
		p_offset = layerCalculation.poolingOffsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, 0);
		cudnnStatus = cudnnPoolingForward(variables.cudnn, variables.pooling_descriptor[0], &alpha, variables.layer_descriptor[1], variables.d_LayerResults, &beta,
			variables.poolinglayer_descriptor[0], variables.d_PoolingResults + p_offset);
		cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Pooling)");
	}

	for (int i = 1; i <= variables.filterMatrix.layers; i++) {
		//Fully Connected (if Weight Dimensions is equal to Layer Width of previous Layer
		if (variables.filterMatrix.h_HiddenDimensions[i * 4 + 2] == variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4]) {
			int* dim = new int[3]{ variables.filterMatrix.h_HiddenDimensions[i * 4 + 1], variables.filterMatrix.h_HiddenDimensions[i * 4],
				variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4 + 1] * (int)pow(variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4],2) };

			l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
			last_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);

			cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_T, dim[0], dim[1], dim[2], &alpha, variables.filterMatrix.d_FilterMatrices[i], dim[2],
				variables.d_LayerResults + last_offset, dim[1], &beta, variables.d_LayerResults + l_offset, dim[0]);
			cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Fully Connected)");

			cublasStatus = cublasSgeam(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, dim[0], dim[1], &alpha, variables.d_LayerResults + l_offset, dim[0], &beta,
				variables.filterMatrix.d_Bias[i], dim[0], variables.d_LayerOutputs + l_offset, dim[0]);
			cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Fully Connected)");

			cudnnStatus = cudnnActivationForward(variables.cudnn, variables.activation_descriptor[i], &alpha, variables.layer_descriptor[i + 1],
				variables.d_LayerOutputs + l_offset, &beta, variables.layer_descriptor[i + 1], variables.d_LayerResults + l_offset);
			cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Fully Connected)");
		}
		//Convolutional Layer
		else {
			//If the previous Layer is a Pooling Layer
			if (variables.filterMatrix.poolingLayers[(i - 1) * 3] == 1) {
				p_offset = layerCalculation.poolingOffsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i);
				l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);

				cudnnStatus = cudnnConvolutionBiasActivationForward(variables.cudnn, &alpha, variables.poolinglayer_descriptor[i - 1], &variables.d_PoolingResults[p_offset],
					variables.filter_descriptor[i], variables.filterMatrix.d_FilterMatrices[i], variables.convolution_descriptor[i], variables.convolution_algorithm[i],
					variables.d_Workspace[i], variables.workspace_bytes[i], &beta, variables.layer_descriptor[i + 1], &variables.d_LayerOutputs[l_offset],
					variables.bias_descriptor[i], variables.filterMatrix.d_Bias[i], variables.activation_descriptor[i], variables.layer_descriptor[i + 1], &variables.d_LayerResults[l_offset]);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Convolution)");
			}
			//If the previous Layer isnt a Pooling Layer
			else {
				last_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);
				l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);

				cudnnStatus = cudnnConvolutionBiasActivationForward(variables.cudnn, &alpha, variables.layer_descriptor[i], &variables.d_LayerResults[last_offset],
					variables.filter_descriptor[i], variables.filterMatrix.d_FilterMatrices[i], variables.convolution_descriptor[i], variables.convolution_algorithm[i],
					variables.d_Workspace[i], variables.workspace_bytes[i], &beta, variables.layer_descriptor[i + 1], &variables.d_LayerOutputs[l_offset],
					variables.bias_descriptor[i], variables.filterMatrix.d_Bias[i], variables.activation_descriptor[i], variables.layer_descriptor[i + 1], &variables.d_LayerResults[l_offset]);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Convolution)");
			}
		}

		//If the Layer isn't the Output Layer
		if (i < variables.filterMatrix.layers) {
			//Max Pooling Layer
			if (variables.filterMatrix.poolingLayers[i * 3] == 1) {
				p_offset = layerCalculation.poolingOffsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1);
				l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
				cudnnStatus = cudnnPoolingForward(variables.cudnn, variables.pooling_descriptor[i], &alpha, variables.layer_descriptor[i + 1], &variables.d_LayerResults[l_offset],
					&beta, variables.poolinglayer_descriptor[i], &variables.d_PoolingResults[p_offset]);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_FORWARD (Pooling)");
			}
		}
	}

	if (!variables.regression) {
		//SoftMax Calculation (Exponential Probability)
		int softMaxOffset = variables.filterMatrix.h_HiddenDimensions[variables.filterMatrix.layers * 4 + 1] *
			pow(variables.filterMatrix.h_HiddenDimensions[variables.filterMatrix.layers * 4], 2) * batchCount;

		kernelSizes[0].x = variables.filterMatrix.h_HiddenDimensions[variables.filterMatrix.layers * 4 + 1] * 5;
		kernelSizes[1].x = variables.filterMatrix.h_HiddenDimensions[(variables.filterMatrix.layers - 1) * 4 + 1] *
			pow(variables.filterMatrix.h_HiddenDimensions[variables.filterMatrix.layers * 4 + 2], 2) / 5;

		layerCalculation.cudaSoftMaxCalculation(variables.d_LayerResults, variables.d_SoftMaxResults, variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers,
			batchCount, variables.filterMatrix.layers + 1, softMaxOffset, kernelSizes);
	}

	if (batchCount + 1 == variables.filterMatrix.batchSize && backPropagationActive) {
		int error_offset;

		//Initializes Variables used in all BackPropagation Methods
		backPropagation.initializeVariables(variables, layerCalculation);

		/*Error Calculation*/

		//Result Layer
		kernelSizes[0].x = variables.filterMatrix.batchSize;
		kernelSizes[1] = variables.filterMatrix.h_HiddenDimensions[variables.filterMatrix.layers * 4 + 1];
		backPropagation.results_ErrorCalculation(kernelSizes);

		for (int i = variables.filterMatrix.layers; i > 0; i--) {
			//FullyConnected Layer with Pooling
			if (variables.filterMatrix.poolingLayers[(i - 1) * 3] == 1 &&
				variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4] == variables.filterMatrix.h_HiddenDimensions[i * 4 + 2] * variables.filterMatrix.poolingLayers[(i - 1) * 3 + 1]) {
				setKernelSizes(variables.filterMatrix.h_HiddenDimensions[i * 4 + 1] * pow(variables.filterMatrix.h_HiddenDimensions[i * 4], 2) * variables.filterMatrix.batchSize);
				backPropagation.poolingFCLayer_ErrorCalculation((i - 1), kernelSizes);
			}
			//FullyConnected Layer
			else if (variables.filterMatrix.poolingLayers[(i - 1) * 3] == 0 && variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4] == variables.filterMatrix.h_HiddenDimensions[i * 4 + 2]) {
				error_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
				l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);

				cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_N, CUBLAS_OP_N, pow(variables.filterMatrix.h_HiddenDimensions[i * 4 + 2], 2) *
					variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4], 1, 10, &alpha, variables.filterMatrix.d_FilterMatrices[3], pow(variables.filterMatrix.h_HiddenDimensions[i * 4 + 2], 2)
					* variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4], variables.d_TotalErrors + error_offset, 10, &beta, variables.d_ActivationErrors + l_offset,
					pow(variables.filterMatrix.h_HiddenDimensions[i * 4 + 2], 2) * variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4]);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_GRADIENT (Fully Connected)");

				cudnnStatus = cudnnActivationBackward(variables.cudnn, variables.activation_descriptor[i], &alpha, variables.layer_descriptor[i], variables.d_LayerResults + l_offset,
					variables.layer_descriptor[i], variables.d_ActivationErrors + l_offset,
					variables.layer_descriptor[i], variables.d_LayerResults + l_offset, &beta, variables.layer_descriptor[i], variables.d_TotalErrors + l_offset);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_GRADIENT (Fully Connected)");
			}
			//Convolutional Layer with Pooling
			else if (variables.filterMatrix.poolingLayers[(i - 1) * 3] == 1) {
				error_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
				l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);
				p_offset = layerCalculation.poolingOffsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i);
				cudnnStatus = cudnnConvolutionBackwardData(variables.cudnn, &alpha, variables.filter_descriptor[i], variables.filterMatrix.d_FilterMatrices[i], variables.layer_descriptor[i + 1],
					variables.d_TotalErrors + error_offset,
					variables.convolution_descriptor[i], variables.bwd_convolution_algorithm[i], variables.d_BwdWorkspace[i], variables.bwdWorkspace_bytes[i], &beta,
					variables.poolinglayer_descriptor[i - 1], variables.d_PoolingErrors + p_offset);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_GRADIENT (Convolution)");

				cudnnStatus = cudnnPoolingBackward(variables.cudnn, variables.pooling_descriptor[i - 1], &alpha, variables.poolinglayer_descriptor[i - 1], variables.d_PoolingResults + p_offset,
					variables.poolinglayer_descriptor[i - 1], variables.d_PoolingErrors + p_offset,
					variables.layer_descriptor[i], variables.d_LayerResults + l_offset, &beta, variables.layer_descriptor[i], variables.d_ActivationErrors + l_offset);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_GRADIENT (Convolution)");

				cudnnStatus = cudnnActivationBackward(variables.cudnn, variables.activation_descriptor[i], &alpha, variables.layer_descriptor[i], variables.d_LayerResults + l_offset,
					variables.layer_descriptor[i], variables.d_ActivationErrors + l_offset,
					variables.layer_descriptor[i], variables.d_LayerResults + l_offset, &beta, variables.layer_descriptor[i], variables.d_TotalErrors + l_offset);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_GRADIENT (Convolution)");
			}
			//Convolutional Layer without Pooling
			else {
				error_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
				l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);

				cudnnStatus = cudnnConvolutionBackwardData(variables.cudnn, &alpha, variables.filter_descriptor[i], variables.filterMatrix.d_FilterMatrices[i],
					variables.layer_descriptor[i + 1], variables.d_TotalErrors + error_offset, variables.convolution_descriptor[i], variables.bwd_convolution_algorithm[i],
					variables.d_BwdWorkspace[i], variables.bwdWorkspace_bytes[i], &beta, variables.layer_descriptor[i], variables.d_ActivationErrors + l_offset);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_GRADIENT (Convolution)");

				cudnnStatus = cudnnActivationBackward(variables.cudnn, variables.activation_descriptor[i], &alpha, variables.layer_descriptor[i], variables.d_LayerResults + l_offset,
					variables.layer_descriptor[i], variables.d_ActivationErrors + l_offset,
					variables.layer_descriptor[i], variables.d_LayerResults + l_offset, &beta, variables.layer_descriptor[i], variables.d_TotalErrors + l_offset);
				cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_GRADIENT (Convolution)");
			}
		}

		/*BackPropagation*/
		for (int i = variables.filterMatrix.layers - 1; i >= 0; i--) {
			if (i > 0) {
				//Fully Connected
				if (variables.filterMatrix.h_HiddenDimensions[i * 4 + 2] == variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4]) {
					int* dim = new int[3]{ variables.filterMatrix.h_HiddenDimensions[i * 4 + 1], variables.filterMatrix.h_HiddenDimensions[i * 4],
						variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4 + 1] * (int)pow(variables.filterMatrix.h_HiddenDimensions[(i - 1) * 4],2) };

					//With Pooling Input
					if (variables.filterMatrix.poolingLayers[(i - 1) * 3] == 1) {
						last_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
						p_offset = layerCalculation.poolingOffsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i);
						cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_T, dim[2], dim[0], dim[1], &learningrate, variables.d_LayerResults + l_offset,
							dim[1], variables.d_TotalErrors + last_offset, dim[0], &alpha, variables.filterMatrix.d_FilterMatrices[i], dim[2]);
						cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_BACKWARD (Fully Connected & Pooling)");
					}
					//Without Pooling Input
					else {
						l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);
						last_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
						cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_T, dim[2], dim[0], dim[1], &learningrate, variables.d_LayerResults + l_offset,
							dim[1], variables.d_TotalErrors + last_offset, dim[0], &alpha, variables.filterMatrix.d_FilterMatrices[i], dim[2]);
						cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_BACKWARD (Fully Connected)");
					}
				}
				//Convolution
				else {
					//With Pooling Input
					if (variables.filterMatrix.poolingLayers[(i - 1) * 3] == 1) {
						last_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
						p_offset = layerCalculation.poolingOffsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i);
						cudnnStatus = cudnnConvolutionBackwardFilter(variables.cudnn, &learningrate, variables.poolinglayer_descriptor[i - 1], variables.d_PoolingResults + p_offset,
							variables.layer_descriptor[i + 1], variables.d_TotalErrors + last_offset, variables.convolution_descriptor[i], variables.bwd_filter_algorithm[i],
							variables.d_BwdFilterWorkspace[i], variables.bwdfilterWorkspace_bytes[i], &alpha, variables.filter_descriptor[i], variables.filterMatrix.d_FilterMatrices[i]);
						cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_BACKWARD (Convolution & Pooling)");
					}
					//Without Pooling Input
					else {
						last_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
						l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);
						cudnnStatus = cudnnConvolutionBackwardFilter(variables.cudnn, &learningrate, variables.layer_descriptor[i], variables.d_LayerResults + l_offset,
							variables.layer_descriptor[i + 1], variables.d_TotalErrors + last_offset, variables.convolution_descriptor[i], variables.bwd_filter_algorithm[i],
							variables.d_BwdFilterWorkspace[i], variables.bwdfilterWorkspace_bytes[i], &alpha, variables.filter_descriptor[i], variables.filterMatrix.d_FilterMatrices[i]);
						cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_BACKWARD (Convolution)");
					}
				}
			}
			//Input Layer
			else {
				//Fully Connected Layer
				if (variables.filterMatrix.h_HiddenDimensions[i * 4 + 2] == variables.filterMatrix.h_InputDimensions[0]) {
					int* dim = new int[3]{ variables.filterMatrix.h_HiddenDimensions[i * 4 + 1], variables.filterMatrix.h_HiddenDimensions[i * 4],
						variables.filterMatrix.h_InputDimensions[1] * (int)pow(variables.filterMatrix.h_InputDimensions[0],2) };

					l_offset = dim[2] * batchCount;
					last_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i + 1, 0);
					cublasStatus = cublasSgemm(variables.cublas, CUBLAS_OP_T, CUBLAS_OP_T, dim[2], dim[0], dim[1], &learningrate, variables.d_ImageParts + l_offset, dim[1],
						variables.d_TotalErrors + last_offset, dim[0], &alpha, variables.filterMatrix.d_FilterMatrices[i], dim[2]);
					cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_BACKWARD (Fully Connected)");
				}
				//Convolutional Layer
				else {
					l_offset = layerCalculation.offsetCalculation(variables.filterMatrix.h_HiddenDimensions, variables.filterMatrix.poolingLayers, batchCount, i, 0);
					cudnnStatus = cudnnConvolutionBackwardFilter(variables.cudnn, &learningrate, variables.layer_descriptor[i], variables.d_ImageParts, variables.layer_descriptor[i + 1],
						variables.d_TotalErrors + l_offset, variables.convolution_descriptor[i], variables.bwd_filter_algorithm[i], variables.d_BwdFilterWorkspace[i],
						variables.bwdfilterWorkspace_bytes[i], &alpha, variables.filter_descriptor[i], variables.filterMatrix.d_FilterMatrices[i]);
					cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_BACKWARD (Convolution)");
				}
			}

			//Default Bias 
			cudnnStatus = cudnnConvolutionBackwardBias(variables.cudnn, &learningrate, variables.layer_descriptor[i + 1], variables.d_TotalErrors + l_offset, &alpha,
				variables.bias_descriptor[i], variables.filterMatrix.d_Bias[i]);
			cudaErrors.checkCudnnStatus(cudnnStatus, "ERR_BACKWARD (Bias)");
		}
	}

	return 0;
}

