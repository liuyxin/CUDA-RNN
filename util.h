#ifndef UTIL_H
#define UTIL_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helper_cuda.h"
#include "cuda.h"
#include "cuMatrix.h"
#include "hardware.h"
#include "cuMatrix4d.h"
#include "cuMatrix.h"
#include "Config.h"
#ifndef TIMEFORWARD
#define TIMEFORWARD 0
#endif
#ifndef TIMEBACKWARD
#define TIMEBACKWARD 1
#endif

void hiddenForward(cuMatrix4d& nonlin, cuMatrix4d& acti,cuMatrix& Weight,cuMatrix4d& bnl,float dropoutRate, bool f);

void non2acti(cuMatrix4d& non, cuMatrix4d& acti,int t);
void acti2non2acti(cuMatrix4d& acti, cuMatrix4d& non,cuMatrix4d& bnl ,cuMatrix& w,int t,bool f);
void hiddenForwardAssist(cuMatrix4d& nonlin, cuMatrix4d& acti,cuMatrix& weight,cuMatrix4d& bnl, bool f);
void smrForward(cuMatrix& wr,cuMatrix4d& ar,cuMatrix& wl, cuMatrix4& al,cuMatrix4d &p);
void smrBP(SoftMax& smr, cuMatrix4d& acti_l,cuMatrix4d& acti_r,cuMatrix4d& acti_l2,cuMatrix4d& acti_r2,cuMatrix4d& dis, cuMatrix4d& dis2, int nSamples);
void hiddenBPTT(cuMatrix4d& delta, cuMatrix w, cuMatrix4d& non, cuMatrix4d& bnl, bool f);

#endif
