#include "LayerCalculation.cuh"

//Checks if an Error has occured
int CheckCudaError(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		return 1;
	}

	return 0;
}

int LayerCalculation::offsetCalculation(int *arraySizes, int* poolingLayers, int batchCount, int layerNum, int Layers)
{
	if (batchDim == 0)
	{
		for (int i = 0; i < Layers + 1; i++)
		{
			batchDim += arraySizes[i * 4 + 1] * (int)pow(arraySizes[i * 4], 2);

			if (poolingLayers[i * 3] == 1)
			{
				poolingBatchDim += arraySizes[i * 4 + 1] * (int)pow(arraySizes[i * 4], 2) / pow(poolingLayers[i * 3 + 1], 2);
			}
		}

		return 0;
	}

	int offset = 0;

	for (int i = 0; i < layerNum - 1; i++)
	{
		offset += arraySizes[i * 4 + 1] * (int)pow(arraySizes[i * 4], 2);
	}

	offset += batchCount * batchDim;
	return offset;
}

int LayerCalculation::poolingOffsetCalculation(int *arraySizes, int* poolingLayers, int batchCount, int layerNum)
{
	int offset = 0;

	for (int i = 0; i < layerNum - 1; i++)
	{
		if (poolingLayers[i * 3] == 1)
		{
			offset += arraySizes[i * 4 + 1] * (int)pow(arraySizes[i * 4], 2) / pow(poolingLayers[i * 3 + 1], 2);
		}
	}

	offset += poolingBatchDim * batchCount;

	return offset;
}

//SoftMax Function (Probability for each Category with an exponetial function)
__global__
void CudaSoftMaxCalculation1(float* results, float* softMaxResults, int s_offset, int r_offset)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;

	sdata[tid] = exp(results[r_offset + tid]);

	__syncthreads();

	//Cuda Reduction Method
	//Source: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf (page 7)
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		if (tid % (2 * s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	softMaxResults[s_offset + tid] = (exp(results[r_offset + tid])) / sdata[0];
}

int LayerCalculation::cudaSoftMaxCalculation(float* results, float* softMaxResults, int* h_ArraySizes, int* poolingLayers, int batchCount, int layerNum, int softMaxOffset, dim3 KernelSizes[])
{
	int r_offset = offsetCalculation(h_ArraySizes, poolingLayers, batchCount, layerNum, 0);
	CudaSoftMaxCalculation1 << <1, KernelSizes[0], KernelSizes[0].x * sizeof(float) >> > (results, softMaxResults, softMaxOffset, r_offset);
	int ret = CheckCudaError(cudaGetLastError());
	return ret;
}


