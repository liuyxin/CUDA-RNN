#ifndef LAYERINIT_H
#define LAYERINIT_H
#include <stdlib.h>
#include "Config.h"
#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "Layer.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "MemoryMonitor.h"
using namespace cv;
//void init_acti(cuMatrixVector<double>& acti_0, vector<vector<int> >& trainX,
//		cuMatrix<double>& sampleY, vector<vector<int> >& trainY, int n);
void init_HLandSMR(vector<HiddenConfig>& HiddenConfigs,
		vector<HiddenLayer> &HiddenLayers, SoftMax &SMR, int word_vec_len);
void weightRandomInit(HiddenLayer &ntw, int inputsize, int hiddensize);
void weightRandomInit(SoftMax &SMR, int nclasses, int nfeatures);
//void init_testdata(vector<vector<int> > &testX,vector<vector<int> > &testY);

#endif
