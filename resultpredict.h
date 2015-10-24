#ifndef RESULTPREDICT_H
#define RESULTPREDICT_H
#include <vector>
#include <string>
#include <iostream>
#include "Config.h"
#include "Base.h"
#include "cuMatrixVector.h"
#include "prepare.cuh"
#include "compute.h"
#include "debug.h"

void testNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR,
		vector<HiddenConfig> &HiddenConfigs, int n,const int size);
void predict(cuMatrixVector<double> &sampleX, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR, vector<HiddenConfig> &HiddenConfigs,int* output,int offset);
#endif
