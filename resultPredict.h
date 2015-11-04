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
void testNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR,bool flag);
void predict(cuMatrixVector &sampleX, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR,int* output,int offset);
#endif
