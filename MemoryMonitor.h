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
		if (0 == size) {
			dev = NULL;
		} else {
			Malloc__();
		}
	}
	MatData(int r , int c ,int ch ,int t) {
		size = r * c * ch * t * sizeof(float);
		host = NULL;
		if (0 == size) {
			dev = NULL;
		} else {
			Malloc__();
		}
	}
	~MatData() {
		if (NULL != host)
			free(host);
		if (NULL != dev)
			cudaFree(dev);
	}
	void Malloc();
	void toCpu();
	void setGpu(float* src);
	float* getDev() {
		assert(NULL != dev);
		return dev;
	}
	float* getHost() {
		if (NULL == host) {
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
		if( NULL == M[idx] ){
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
		if(NULL == M[t] || 1 == M[t].use_count() ){
			return t;
		}
		while(t++, t < n + sizeof(float)){
			if(NULL == M[t] || 1 == M[t].use_count()){
				return t;
			}
		}
		printf("no available Memory for size %u\n",n);
		exit(0);
	}
	unsigned int idx;
};
#endif
