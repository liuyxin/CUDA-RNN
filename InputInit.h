#ifndef INPUTINIT_H
#define INPUTINIT_H
#include "Samples.h"
#include "Config.h"
#include "hardware.h"
#include "cuMatrix4d.h"
void initTestdata(vector<vector<int> > &testX, vector<vector<int> > &testY);
void initTraindata(vector<vector<int> > &trainX, vector<vector<int> > &trainY);
void Data2GPU(vector<vector<int> > &trainX, vector<vector<int> > &trainY,
		vector<vector<int> > &testX, vector<vector<int> > &testY);
void init_acti0(cuMatrix4d& acti_0, cuMatrix& sampleY);
void set_groundtruth(cuMatrix4d& gt, cuMatrix& sampleY);
void getDataMat(cuMatrix4d &sampleX, int off, int bs, int n,
		bool flag);
void get_res_array(cuMatrix src, int *res, int offset) ;
void set_label(int* label, int size,bool flag);
#endif
