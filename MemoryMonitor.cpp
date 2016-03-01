#include "MemoryMonitor.h"

void MatData::Malloc__() {
	assert(size);
//	host = (float*) malloc(size);
//	memset(host, 0, rows * cols * sizeof(float));
	cudaError_t cudaStat = cudaMalloc((void**) &dev, size);
	if (cudaStat != cudaSuccess) {
		printf("MatData::cudaMalloc() failed\n");
		exit(0);
	}
	cudaStat = cudaMemset(dev, 0, size);
	if (cudaStat != cudaSuccess) {
		printf("MatData::cudaMemset() failed\n");
		exit(0);
	}
}
void MatData::Malloc() {
	assert( NULL == dev && NULL == host );
	Malloc__();
}

void MatData::CpuMalloc() {
	if (NULL == host && 0 != size) {
		host = (float*) malloc(size);
//		memset(host, 0, rows * cols * sizeof(float));
		memset(host, 0, size);
	}
}

void MatData::toCpu() {
	CpuMalloc();
	assert(host != NULL && dev != NULL);
	cudaError_t cudaStat;
	cudaStat = cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		printf("MatData::toCPU data download failed\n");
		exit(0);
	}
}

void MatData::setGpu(float* src) {
	assert(dev != NULL);
	cudaError_t cudaStat = cudaMemcpy(dev, src, size, cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) {
		printf("MatData::setGpu failed\n");
		exit(0);
	}
}

std::unordered_map<unsigned int, shared_ptr<MatData>> tmpMemory::M;

