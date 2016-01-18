#ifndef RESULTPREDICT_H
#define RESULTPREDICT_H
#include <vector>
#include <string>
#include <iostream>
#include "Config.h"
#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "cuMath.h"
#include "InputInit.h"
#include "util.h"
#include "memory"
static	std::vector<cuMatrix4d > al;
static	std::vector<cuMatrix4d > ar;
static	std::vector<cuMatrix4d > as;
void testNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR,bool flag);
void predict(cuMatrix4d &sampleX, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR,int* output,int offset);
//void hiddenForward(cuMatrix4d& acti,cuMatrix& Weight,float dropoutRate);
void hiddenForward_(cuMatrix4d& acti,cuMatrix& weight, float dr, bool f);
void smrForward_(cuMatrix& M, cuMatrix4d& acti_l, cuMatrix4d& acti_r, SoftMax& SMR);
void non2acti_(cuMatrix4d& non, cuMatrix4d& acti,float bnl,int t);
void acti2non2acti_(cuMatrix4d& acti, cuMatrix4d& non,float bnl ,cuMatrix& w,int t,bool f);
void testInit(vector<HiddenLayer> &Hiddenlayers);
//void hiddenForward(cuMatrix4d& nonlin, cuMatrix4d& acti,cuMatrix& Weight,cuMatrix4d& bnl,float dropoutRate, bool f);
#endif
