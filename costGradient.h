#ifndef COSTGRADIENT_H
#define COSTGRADIENT_H

#include <cuda_runtime.h>
#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "Config.h"
#include "cuMath.h"
#include "InputInit.h"

static vector<vector<cuMatrix> > acti_l;
static vector<vector<cuMatrix> > acti_r;
static vector<vector<cuMatrix> > acti_l2;
static vector<vector<cuMatrix> > acti_r2;
static vector<vector<cuMatrix> > acti_sum;
static vector<vector<cuMatrix> > acti2_sum;
static vector<vector<cuMatrix> > nonlin_l;
static vector<vector<cuMatrix> > nonlin_r;
static vector<vector<cuMatrix> > bernoulli_l;
static vector<vector<cuMatrix> > bernoulli_r;
static vector<cuMatrix> p;
static vector<cuMatrix> groundTruth;
static vector<cuMatrix> dis;
static vector<cuMatrix> dis2;

static vector<vector<cuMatrix> > delta_l;
static vector<vector<cuMatrix> > delta_ld2;
static vector<vector<cuMatrix> > delta_r;
static vector<vector<cuMatrix> > delta_rd2;


void costParamentInit(vector<HiddenLayer> &Hiddenlayers,SoftMax &SMR);
void getNetworkCost(cuMatrixVector &acti_0, cuMatrix &sampleY,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR);
#endif
