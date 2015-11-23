#ifndef MEMORYMONITOR_H
#define MEMORYMONITOR_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <assert.h>
#include <stdio.h>
#include <algorithm>
//#include "Config.h"
using namespace std;
class MatData {
public:
	MatData(int r = 0, int c = 0) {
//		rows = r;
//		cols = c;
		size = r * c * sizeof(float);
		host = NULL;
		if (size == 0) {
			dev = NULL;
		} else {
			Malloc__();
		}
	}
	~MatData() {
		if (host != NULL)
			free(host);
		if (dev != NULL)
			cudaFree(dev);
	}
	void Malloc();
	void toCpu();
	void setGpu(float* src);
	float* getDev() {
		assert(dev != NULL);
		return dev;
	}
	float* getHost() {
		if (host == NULL) {
			toCpu();
		}
		return host;
	}
	int sizes(){
		return size;
	}
private:
	int size;
	float* host;
	float* dev;
	void Malloc__();
	void CpuMalloc();
};

#endif
