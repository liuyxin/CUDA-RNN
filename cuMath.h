#ifndef CUMATH_H
#define CUMATH_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "helper_cuda.h"
#include "curand.h"
#include "cuda.h"
#include "cuMatrix.h"
#include "hardware.h"
#include "cuMatrix.h"
#include "cuMatrix4d.h"
#include "memory"
using namespace std;
cuMatrix ReLU(cuMatrix& cumat);
cuMatrix dReLU(cuMatrix& cumat);
cuMatrix reduceMax(cuMatrix& src);
cuMatrix reduceSum(cuMatrix& src);
cuMatrix Exp(cuMatrix& src);
void Exp(cuMatrix& src,cuMatrix& res);
cuMatrix Log(cuMatrix& src);
cuMatrix4d Log(cuMatrix4d& src);
cuMatrix Pow(cuMatrix x,float y);
cuMatrix Pow(cuMatrix x,cuMatrix y);
void cuMultiplication(cuMatrix src1,cuMatrix src2,cuMatrix dst);
void cuPlus(cuMatrix src1,cuMatrix src2,cuMatrix dst);
void cuDec(cuMatrix src1, cuMatrix src2, cuMatrix dst);
void cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst);
void cuDiv(float src1, cuMatrix src2, cuMatrix dst);
void creatBnl(cuMatrix4d& bnl,float threshold);
void cuDec(cuMatrix4d src1, cuMatrix4d src2, cuMatrix4d dst);

cublasHandle_t& getHandle();
void cuMatrix4d_Add(cuMatrix4d& src1,cuMatrix4d& src2, cuMatrix4d& dst);	
//dst = src1 * src2;
void cuMatrix4d_matMul(cuMatrix4d& src1,cuMatrix4d src2, cuMatrix4d& dst);	
void cuMatrix4d_matMul(cuMatrix src1, cuMatrix4d& src2, cuMatrix4d& dst);
//dst = src1.Mul(src2);
void cuMatrix4d_eleMul(cuMatrix4d& src1,cuMatrix4d& src2, cuMatrix4d& dst);	
void cuMatrix4dRightTrans(cuMatrix4d& src,cuMatrix& dst);
void cuMatrix4dRightInverseTrans(cuMatrix&src,cuMatrix4d& dst);
void extractMatrix(cuMatrix& src,cuMatrix4d& dst);
void square(cuMatrix4d& src,cuMatrix4d& dst);
#endif
