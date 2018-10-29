#pragma once
#pragma comment(lib,"cublas.lib")

#include "cublas_v2.h"
#include "cudnn.h"
#include "ResultVariables.h"
#include <iostream>


class CudaErrors {
public:
	void checkCublasStatus(cublasStatus_t cublasStatus, char* description);
	void checkCudnnStatus(cudnnStatus_t cudnnStatus, char* description);
	void checkCudaError(cudaError_t cudaError, char* description);
private:
	static const char* getCublasStatus(cublasStatus_t cublasStatus);
	static const char* getCudnnStatus(cudnnStatus_t cudnnStatus);

};