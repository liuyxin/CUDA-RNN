#ifndef COMPUTE_H
#define COMPUTE_H
#include "Base.h"
#include <assert.h>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__
//#include <math_functions.h>
#include <device_functions.h>
#include "hardware.h"
__global__ void ElementAdd_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt);
__global__ void ElementAdd_kernel(double* dev_x, double* y_, double* dev_z,
		int cols, int maxt);
__global__ void ElementMul_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt);
__global__ void ElementMul_kernel(double* dev_x, double* y_, double* dev_z,
		int cols, int maxt);
__global__ void ElementDec_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt);
__global__ void ElementDec_kernel(double* dev_x, double* y_, double* dev_z,
		int cols, int maxt);
__global__ void ElementDiv_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt);
__global__ void ElementDiv_kernel(double* dev_x, double* y_, double* dev_z,
		int cols, int maxt);
__global__ void ElementDiv_kernel(double* x_, double dev_y, double* dev_z,
		int cols, int maxt);
__global__ void ElementPow_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt);
__global__ void ElementPow_kernel(double* dev_x, double* y_, double* dev_z,
		int cols, int maxt);
__global__ void ReLU_kernel(double* dev_a, int rows, int cols, int maxt);
__global__ void dReLU_kernel(double* src,double*dst ,int cols, int maxt);
__global__ void reduce_max_kernel(double* dev_x, double* dev_y, int rows,
		int cols, int maxt);
__global__ void reduce_sum_kernel(double* dev_x, double* dev_y, int rows,
		int cols, int maxt);
__global__ void exp_mat_kernel(double* dev_x, double* dev_y, int cols,
		int maxt);
__global__ void exp_log_kernel(double* dev_x, double* dev_y, int cols,
		int maxt);
__global__ void maxtri_sum_kernel(double* src, double* res,int threadnum, int cols);
__global__ void drop_kernel(double* dev_x, int cols);

void ElementAdd(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z);
void ElementAdd(cuMatrix<double>* x, double y, cuMatrix<double>* z);
void ElementMul(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z);
void ElementMul(cuMatrix<double>* x, double y, cuMatrix<double>* z);
void ElementDec(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z);
void ElementDec(cuMatrix<double>* x, double y, cuMatrix<double>* z);
void ElementDiv(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z);
void ElementDiv(cuMatrix<double>* x, double y, cuMatrix<double>* z);
void ElementDiv(double x, cuMatrix<double>* y, cuMatrix<double>* z);
void ElementExp(cuMatrix<double>* src, cuMatrix<double>* dst);
void ElementLog(cuMatrix<double>* src, cuMatrix<double>* dst);
void ReLU(cuMatrix<double>* x);
cuMatrix<double>* dReLU(cuMatrix<double>* x);
void reduce_max(cuMatrix<double>* src, cuMatrix<double>* dst);
void reduce_sum(cuMatrix<double>* src, cuMatrix<double>* dst);
double matrix_sum(cuMatrix<double>* src);
void ElementPow(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z);
void ElementPow(cuMatrix<double>* x, double y, cuMatrix<double>* z);
void drop(cuMatrix<double>* x);
#endif
