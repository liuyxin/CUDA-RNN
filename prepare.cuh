#ifndef PREPARE_CUH
#define PREPARE_CUH
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "MemoryMonitor.h"
#include "Config.h"
#include "hardware.h"
#include "debug.h"
__global__ void set_acti0_kernel(double** acti0, int* src, int* dev_ran,
		int cols,int ngram);
__global__ void set_sampleY_kernel(double* sampleY, int* src, int* dev_ran,
		int cols,int ngram);
__global__ void set_gt_kernel(double** gt_, double* y,int rows, int cols);
__global__ void getDataMat_kernel(double** sampleX, int* src,int off ,int cols,
		int ngram);
__global__ void get_res_array_kernel(double* src,int* dev_res ,int rows, int cols);
__global__ void set_label_kernel(int* dst,int *src,int num , int threadnum , int mid);

void set_gpudata(int* host_, int* dev_, int size);
void set_acti0(cuMatrixVector<double>& acti0, int *inputx, int inputsizex,
		cuMatrix<double>& sampleY, int*inputy, int inputsizey);
void set_acti0(cuMatrixVector<double>& acti0, cuMatrix<double>& sampleY) ;
void set_groundtruth(cuMatrixVector<double>& gt, cuMatrix<double>& sampleY);
//void getDataMat(cuMatrixVector<double> &sampleX, int off , bool falg);
void getDataMat(cuMatrixVector<double> &sampleX, int off ,int bs,int n,const int size) ;
void get_res_array(cuMatrix<double> *src,int *res,int offset);
void set_label(int* label,const int size);
#endif
