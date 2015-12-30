#ifndef CUMATRIX_H
#define CUMATRIX_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include <map>
#include "helper_cuda.h"
#include "MemoryMonitor.h"
#include "hardware.h"
//#include "cuMath.h";
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
	cuMatrix(shared_ptr<MatData> tmpd, int r, int c) {
		size = r * c * sizeof(float);
		assert(size == tmpd->sizes());
		row = r;
		col = c;
		data = tmpd;
	}
//	static map<int, shared_ptr<MatData> >& tmpMemory(){
//		static map<int,shared_ptr<MatData> > TmpMemory;
//		return TmpMemory;
//	}
	static std::map<unsigned int, shared_ptr<MatData>> tmpMemory;
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
	float* getDev() {
		return data->getDev();
	}
	float* getHost() {
		return data->getHost();
	}
	void printMat() {
		data->toCpu();
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				printf("%f,", getHost()[i * cols() + j]);
			}
			printf("\n");
		}
	}
	void copyTo(cuMatrix dst, cudaStream_t stream1 = 0) {
		if (cols() != dst.cols() || rows() != dst.rows()) {
			printf("cuMatrix::copyTo() size error\n");
			exit(0);
		}
		cudaError_t cudaStat;
		cudaStat = cudaMemcpyAsync(dst.data->getDev(), data->getDev(), size,
				cudaMemcpyDeviceToDevice, stream1);
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
	cuMatrix operator *(cuMatrix cumat);      //matrix mul
	cuMatrix operator *(float i);
	cuMatrix operator /(cuMatrix cumat);
	cuMatrix operator /(float i);
	void operator +=(cuMatrix cumat);
	void operator +=(float i);
	void operator -=(cuMatrix cumat);
	void operator -=(float i);
	void operator /=(cuMatrix cumat);
	void operator /=(float i);
	void operator *=(float i);
	void ReLU2(cuMatrix& cumat);
	void Square2(cuMatrix& cumat);
	void Mul2(cuMatrix cumat,cuMatrix& dst);
	void Mul2(float i ,cuMatrix& dst);
	float& getSum();
	friend cuMatrix operator /(float i, cuMatrix cumat);
	
private:
	float sum;
	int row;
	int col;
	int size;
};
cublasHandle_t& getHandle();

#endif
