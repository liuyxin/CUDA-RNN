//#ifndef UTIL_H
//#define UTIL_H
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helper_cuda.h"
#include "cuda.h"
#include "hardware.h"
#include "cuMatrix4d.h"
#include "cuMatrix.h"
#include "Layer.h"
#include "Config.h"
#include "cuMath.h"
#ifndef TIMEFORWARD
#define TIMEFORWARD 0
#endif
#ifndef TIMEBACKWARD
#define TIMEBACKWARD 1
#endif

void hiddenForward(cuMatrix4d& nonlin, cuMatrix4d& acti,cuMatrix& Weight,cuMatrix4d& bnl, bool f);
void non2acti(cuMatrix4d& non, cuMatrix4d& acti, cuMatrix4d& bnl, int t);
void acti2non2acti(cuMatrix4d& acti, cuMatrix4d& non,cuMatrix4d& bnl ,cuMatrix& w,int t,bool f);

//void hiddenForwardAssist(cuMatrix4d& nonlin, cuMatrix4d& acti,cuMatrix& weight,cuMatrix4d& bnl, bool f);
void smrForward(cuMatrix& wr,cuMatrix4d& ar,cuMatrix& wl, cuMatrix4d& al,cuMatrix4d &p);
void smrBP(SoftMax& smr, cuMatrix4d& acti_l,cuMatrix4d& acti_r,cuMatrix4d& acti_l2,cuMatrix4d& acti_r2,cuMatrix4d& dis, cuMatrix4d& dis2, int nSamples);
void hiddenBPTT(cuMatrix4d& delta, cuMatrix w, cuMatrix4d& non, cuMatrix4d& bnl, bool f);
void hiddenGetUgrad(cuMatrix4d& delta_l, cuMatrix4d& delta_r, 
		    cuMatrix4d& delta_ld2, cuMatrix4d& delta_rd2,
		    cuMatrix4d& acti_sum, cuMatrix4d& acti2_sum, HiddenLayer& hidden , float WeightDecay);
void hiddenGetWgrad(cuMatrix4d& delta_l, cuMatrix4d& delta_r,
		cuMatrix4d& delta_ld2, cuMatrix4d& delta_rd2,
		cuMatrix4d& acti_l, cuMatrix4d& acti_r, 
		cuMatrix4d& acti_l2, cuMatrix4d& acti_r2, HiddenLayer& hidden, float WeightDecay);

void bpttInit(HiddenLayer& hidden, cuMatrix4d& delta_l1, cuMatrix4d& delta_r1,
		cuMatrix4d& delta_l, cuMatrix4d& delta_r,cuMatrix4d& delta_ld1, cuMatrix4d& delta_rd1,
		cuMatrix4d& delta_ld, cuMatrix4d& delta_rd);
//#endif
