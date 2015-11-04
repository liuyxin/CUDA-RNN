#ifndef TRAINNETWORK_H
#define TRAINNETWORK_H
#include "Config.h"
#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "Samples.h"
#include "InputInit.h"
#include "costGradient.h"
#include "resultPredict.h"
#include <stdio.h>
#include <iostream>
using namespace std;
void trainNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR,
		int reword_size);
#endif
