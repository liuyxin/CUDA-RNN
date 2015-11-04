#ifndef INPUTINIT_H
#define INPUTINIT_H
#include "Samples.h"
#include "Config.h"
#include "hardware.h"
void initTestdata(vector<vector<int> > &testX, vector<vector<int> > &testY);
void initTraindata(vector<vector<int> > &trainX, vector<vector<int> > &trainY);
void Data2GPU(vector<vector<int> > &trainX, vector<vector<int> > &trainY,
		vector<vector<int> > &testX, vector<vector<int> > &testY);

__global__ void set_acti0_kernel(double** acti0, int* src, int* dev_ran,
		int cols, int ngram);
__global__ void set_sampleY_kernel(double* sampleY, int* src, int* dev_ran,
		int cols, int ngram);
void init_acti0(cuMatrixVector& acti_0, cuMatrix& sampleY);

__global__ void set_gt_kernel(float** gt_, float* y, int rows, int cols);
void set_groundtruth(cuMatrixVector& gt, cuMatrix& sampleY);
void  getDataMat(cuMatrixVector &sampleX, int off, int bs, int n,
		bool flag);
void get_res_array(cuMatrix src, int *res, int offset) ;
void set_label(int* label, int size,bool flag);
#endif
