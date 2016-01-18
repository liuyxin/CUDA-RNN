#ifndef MEMORYMONITOR_H
#define MEMORYMONITOR_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <unordered_map>
//#include "Config.h"
using namespace std;
class MatData {
public:
	MatData(int r = 0, int c = 0) {
		size = r * c * sizeof(float);
		host = NULL;
		if (size == 0) {
			dev = NULL;
		} else {
			Malloc__();
		}
	}
	MatData(int r , int c ,int ch ,int t) {
		size = r * c * ch * t * sizeof(float);
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

class tmpMemory{
	public:
	static unordered_map<unsigned int, shared_ptr<MatData>> M;
	tmpMemory(unsigned int n){
		idx = getUsableIndex(n);
	}
	shared_ptr<MatData> getMem(){
		if(M[idx] == NULL){
			return NULL; 
		}
		return M[idx];
	}
	void set(shared_ptr<MatData> data){
		M[idx] = data;		
	}

	private:
	unsigned int getUsableIndex(unsigned int n){
		unsigned int t = n;
		if(M[t] == NULL || M[t].use_count() == 1){
			return t;
		}
		while(t++, t < n + sizeof(float)){
			if(M[t] == NULL || M[t].use_count() == 1){
				return t;
			}
		}
		printf("no available Memory for size %u\n",n);
		exit(0);
	}
	unsigned int idx;
};
#endif
