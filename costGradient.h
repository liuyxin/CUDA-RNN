#ifndef COSTGRADIENT_H
#define COSTGRADIENT_H

#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "Config.h"
#include "cuMath.h"
#include "InputInit.h"
void getNetworkCost(cuMatrixVector &acti_0, cuMatrix &sampleY,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR);
#endif
