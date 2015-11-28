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
cuMatrix Pow(cuMatrix x,float y);
cuMatrix Pow(cuMatrix x,cuMatrix y);
void cuMultiplication(cuMatrix src1,cuMatrix src2,cuMatrix dst);
void cuPlus(cuMatrix src1,cuMatrix src2,cuMatrix dst);
void cuDec(cuMatrix src1, cuMatrix src2, cuMatrix dst);
void cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst);
void cuDiv(float src1, cuMatrix src2, cuMatrix dst);
void creatBnl(cuMatrix& bnl,float threshold);

#endif
