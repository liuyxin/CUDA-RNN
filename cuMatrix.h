#ifndef CUMATRIX_H
#define CUMATRIX_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include "helper_cuda.h"
#include "MemoryMonitor.h"
#include "hardware.h"

using namespace std;
class cuMatrix {
public:
	cuMatrix(int r = 0, int c = 0) {
		row = r;
		col = c;
		size = r * c * sizeof(float);
		data = std::make_shared < MatData > (r, c);
	}
	cuMatrix(float *src, int r, int c) {
		row = r;
		col = c;
		size = r * c * sizeof(float);
		data = make_shared < MatData > (r, c);
		data->setGpu(src);
	}
	shared_ptr<MatData> data;
	int rows() {
		return row;
	}
	int cols() {
		return col;
	}
	int sizes() {
		return size;
	}
	float* getDev(){
		return data->getDev();
	}
	float* getHost(){
		return data->getHost();
	}
	void printMat(){
		data->toCpu();
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				printf("%f,",getHost()[i*cols() + j]);
			}
			printf("\n");
		}
	}
	float getSum(){
		data->toCpu();
		float sum = 0;
		float *tmp = getHost();
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				sum += tmp[i*cols() + j];
			}
		}
		return sum;
	}
	void copyTo(cuMatrix &dst){
		if(cols()!=dst.cols() || rows()!=dst.rows()){
			printf("cuMatrix::copyTo() size error\n");
			exit(0);
		}
		cudaError_t cudaStat;
		cudaStat = cudaMemcpy(dst.data->getDev(), data->getDev(), size, cudaMemcpyDeviceToDevice);
		if (cudaStat != cudaSuccess) {
			printf("cuMatrix::copyTo cudaMemcpy() failed\n");
			exit(0);
		}
	}
	cuMatrix t();
	cuMatrix Mul(cuMatrix cumat);      //per-element  mul
	cuMatrix operator +(cuMatrix cumat);
	cuMatrix operator +(float i);
	cuMatrix operator -(cuMatrix cumat);
	cuMatrix operator -(float i);
	cuMatrix operator *(cuMatrix cumat);//matrix mul
	cuMatrix operator *(float i);
	cuMatrix operator /(cuMatrix cumat);
	cuMatrix operator /(float i);
	friend cuMatrix operator /(float i,cuMatrix cumat);
private:
	int row;
	int col;
	int size;
};
#endif
