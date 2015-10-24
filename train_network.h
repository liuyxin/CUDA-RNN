#ifndef TRAIN_NETWORK_H
#define TRAIN_NETWORK_H

#include "cuMatrixVector.h"
#include "Base.h"
#include "readdata.h"
#include "Layer.h"
#include "cuMatrixVector.h"
#include "Config.h"
#include "init.h"
#include "train_network.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include "compute.h"
#include "debug.h"
#include "resultpredict.h"
using namespace std;

void trainNetwork(vector<vector<int> > &trainX, vector<vector<int> > &trainY,
		vector<HiddenConfig> &HiddenConfigs, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR, vector<vector<int> > &testX,
		vector<vector<int> > &testY, vector<string> &re_word);
void getNetworkCost(cuMatrixVector<double> &acti_0, cuMatrix<double> &sampleY,
		vector<HiddenConfig> &HiddenConfigs, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR);

#endif
