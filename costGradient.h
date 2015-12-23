#ifndef COSTGRADIENT_H
#define COSTGRADIENT_H

#include <cuda_runtime.h>
#include "cuMatrix.h"
#include "cuMatrix4d.h"
#include "Config.h"
#include "cuMath.h"
#include "InputInit.h"
#include "util.h"
static vector<cuMatrix4d> acti_l;
static vector<cuMatrix4d> acti_r;
static vector<cuMatrix4d> acti_l2;
static vector<cuMatrix4d> acti_r2;
static vector<cuMatrix4d> acti_sum;
static vector<cuMatrix4d> acti2_sum;
static vector<cuMatrix4d> nonlin_l;
static vector<cuMatrix4d> nonlin_r;
static vector<cuMatrix4d> bernoulli_l;
static vector<cuMatrix4d> bernoulli_r;
static cuMatrix4d p;
static cuMatrix4d groundTruth;
static cuMatrix4d dis;
static cuMatrix4d dis2;

static vector<cuMatrix4d> delta_l;
static vector<cuMatrix4d> delta_ld2;
static vector<cuMatrix4d> delta_r;
static vector<cuMatrix4d> delta_rd2;


void costParamentInit(vector<HiddenLayer> &Hiddenlayers,SoftMax &SMR);
void getNetworkCost(cuMatrix4d &acti_0, cuMatrix &sampleY,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR);
#endif
