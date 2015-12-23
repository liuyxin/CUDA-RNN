#ifndef CUMATH_H
#define CUMATH_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helper_cuda.h"
#include "curand.h"
#include "cuda.h"
#include "cuMatrix.h"
#include "hardware.h"


cuMatrix ReLU(cuMatrix& cumat);
cuMatrix dReLU(cuMatrix& cumat);
cuMatrix reduceMax(cuMatrix& src);
cuMatrix reduceSum(cuMatrix& src);
cuMatrix Exp(cuMatrix& src);
void Exp(cuMatrix& src,cuMatrix& res);
cuMatrix Log(cuMatrix& src);
cuMatrix Log(cuMatrix4d& src);
cuMatrix Pow(cuMatrix x,float y);
cuMatrix Pow(cuMatrix x,cuMatrix y);
void cuMultiplication(cuMatrix src1,cuMatrix src2,cuMatrix dst);
void cuPlus(cuMatrix src1,cuMatrix src2,cuMatrix dst);
void cuDec(cuMatrix src1, cuMatrix src2, cuMatrix dst);
void cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst);
void cuDiv(float src1, cuMatrix src2, cuMatrix dst);
void creatBnl(cuMatrix4d& bnl,float threshold);
void cuDec(cuMatrix4d src1, cuMatrix4d src2, cuMatrix4d dst);

//__global__ void ReLU_kernel(float* src, float* dst, int rows, int cols,
//		int maxt);
//__global__ void dReLU_kernel(float* src, float* dst, int cols, int maxt);
//__global__ void reduce_max_kernel(float* dev_x, float* dev_y, int rows,
//		int cols, int maxt);
//__global__ void reduce_sum_kernel(float* dev_x, float* dev_y, int rows,
//		int cols, int maxt);
//__global__ void log_kernel(float* dev_x, float* dev_y, int cols, int maxt);
//__global__ void exp_kernel(float* dev_x, float* dev_y, int cols, int maxt);
//__global__ void Pow_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
//		int maxt);
//__global__ void Pow_kernel(float* dev_x, float y_, float* dev_z, int cols,
//		int maxt);
//__global__ void cuPlus_kernel(float* dev_x, float* dev_y, float* dev_z,
//		int cols, int maxt);
//__global__ void cuDec_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
//		int maxt);
//__global__ void cuDiv_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
//		int maxt);
//__global__ void cuDiv_kernel(float x_, float* dev_y, float* dev_z, int cols,
//		int maxt);
//__global__ void creatBnl_kernel(float* dev, float threshold, int cols, int maxt);
#endif
