#ifndef CUMATH_H
#define CUMATH_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"
#include "cuMatrix.h"
#include "hardware.h"


cuMatrix ReLU(cuMatrix& cumat);
cuMatrix dReLU(cuMatrix& cumat);
cuMatrix reduceMax(cuMatrix src);
cuMatrix reduceSum(cuMatrix src);
cuMatrix Exp(cuMatrix src);
cuMatrix Log(cuMatrix src);
cuMatrix Pow(cuMatrix x,float y);
cuMatrix Pow(cuMatrix x,cuMatrix y);
#endif
