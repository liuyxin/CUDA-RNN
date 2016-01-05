#include "InputInit.h"

__global__ void set_sampleY_kernel(float* sampleY, int* src, int* dev_ran,
		int cols, int ngram) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	sampleY[tid * cols + bid] = src[dev_ran[bid] * ngram + tid];
}

__global__ void set_acti0_kernel(float* acti0, int* src, int* dev_ran,
		int cols, int ngram, int a2) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	float *p = acti0 + tid * a2;
	int n = src[dev_ran[bid] * ngram + tid];
	p[n * cols + bid] = 1;
}

void init_acti0(cuMatrix4d& acti_0, cuMatrix& sampleY) {
	int bs = Config::instance()->get_batch_size();
	int ngram = Config::instance()->get_ngram();
	int *dev_ran = NULL;
	Samples::instance()->randproductor_init();
	cudaError_t cudaStat = cudaMalloc((void**) &dev_ran, bs * sizeof(int));
	if (cudaStat != cudaSuccess) {
		printf("init_acti0 failed\n");
		exit(0);
	}
	checkCudaErrors(
			cudaMemcpyAsync(dev_ran, Samples::instance()->get_rand(1),
					bs * sizeof(int), cudaMemcpyHostToDevice, 0));
	dim3 block = dim3(bs);

	dim3 thread = dim3(ngram);
	set_acti0_kernel<<<block, thread>>>(acti_0.getDev(),
			Samples::instance()->get_trainX(), dev_ran, bs, ngram, acti_0.area2D());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("set_acti0_kernel-2");
	set_sampleY_kernel<<<block, thread,0,0>>>(sampleY.getDev(),
			Samples::instance()->get_trainY(), dev_ran, bs, ngram);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("set_sampleY_kernel-2");
	checkCudaErrors(cudaFree(dev_ran));
}

__global__ void set_gt_kernel(float* gt_, float* y , int a2) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int cols = gridDim.x;
	float* p = gt_ + a2 * tid;
	int i = y[tid * cols + bid];
	assert(i < 10);
	p[i * cols + bid] = 1.0;
}

void set_groundtruth(cuMatrix4d& gt, cuMatrix& sampleY) {
	dim3 block = dim3(sampleY.cols());
	dim3 thread = dim3(sampleY.rows());
	set_gt_kernel<<<block, thread>>>(gt.getDev(), sampleY.getDev(),gt.area2D());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("set_groundtruth ");
}


void initTestdata(vector<vector<int> > &testX, vector<vector<int> > &testY) {
	int *host_X = (int *) malloc(
			sizeof(int) * testX.size() * Config::instance()->get_ngram());
	int *host_Y = (int *) malloc(
			sizeof(int) * testY.size() * Config::instance()->get_ngram());
	for (int i = 0; i < testX.size(); i++) {
		memcpy(host_X + i * Config::instance()->get_ngram(), &testX[i][0], sizeof(int) * Config::instance()->get_ngram());
	}
	for (int i = 0; i < testY.size(); i++) {
		memcpy(host_Y + i * Config::instance()->get_ngram(), &testY[i][0], sizeof(int) * Config::instance()->get_ngram());
	}
	Samples::instance()->testX2gpu(host_X,
			sizeof(int) * testX.size() * Config::instance()->get_ngram());
	Samples::instance()->testY2gpu(host_Y,
			sizeof(int) * testY.size() * Config::instance()->get_ngram());
	free(host_X);
	free(host_Y);
}

void initTraindata(vector<vector<int> > &trainX, vector<vector<int> > &trainY) {
	int *host_X = (int *) malloc(
			sizeof(int) * trainX.size() * Config::instance()->get_ngram());
	int *host_Y = (int *) malloc(
			sizeof(int) * trainY.size() * Config::instance()->get_ngram());
	for (int i = 0; i < trainX.size(); i++) {
		memcpy(host_X + i * Config::instance()->get_ngram(), &trainX[i][0], sizeof(int) * Config::instance()->get_ngram());
	}
	for (int i = 0; i < trainY.size(); i++) {
		memcpy(host_Y + i * Config::instance()->get_ngram(), &trainY[i][0], sizeof(int) * Config::instance()->get_ngram());
	}
	Samples::instance()->trainX2gpu(host_X,
			sizeof(int) * trainX.size() * Config::instance()->get_ngram());
	Samples::instance()->trainY2gpu(host_Y,
			sizeof(int) * trainY.size() * Config::instance()->get_ngram());
	free(host_X);
	free(host_Y);
}

void Data2GPU(vector<vector<int> > &trainX, vector<vector<int> > &trainY,
		vector<vector<int> > &testX, vector<vector<int> > &testY) {
	initTestdata(testX, testY);
	initTraindata(trainX, trainY);
}

__global__ void getDataMat_kernel(float* sampleX, int* src, int off, int cols,
		int ngram, int a2) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	float *p = sampleX + tid * a2;
	int n = src[(off + bid) * ngram + tid];
	p[n * cols + bid] = 1.0;
}

void getDataMat(cuMatrix4d &sampleX, int off, int bs, int n, bool flag) {
	int ngram = Config::instance()->get_ngram();
	dim3 thread = dim3(ngram);
	dim3 block = dim3(bs);
	if (flag) {
		getDataMat_kernel<<<block, thread>>>(sampleX.getDev(),
				Samples::instance()->get_trainX(), off, bs, ngram, sampleX.area2D());
	} else {
		getDataMat_kernel<<<block, thread>>>(sampleX.getDev(),
				Samples::instance()->get_testX(), off, bs, ngram, sampleX.area2D());
	}
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("getDataMat_kernel ");

}

__global__ void get_res_array_kernel(float* src, int* dev_res, int rows,
		int cols) {
	int bid = blockIdx.x;
	float max = src[bid];
	dev_res[bid] = 0;
	for (int i = 1; i < rows; i++) {
		if (max < src[i * cols + bid]) {
			max = src[i * cols + bid];
			dev_res[bid] = i;
		}
	}
}

void get_res_array(cuMatrix src, int *res, int offset) {
	int *dev_res;
	checkCudaErrors(cudaMalloc((void** )&dev_res, sizeof(int) * src.cols()));
	get_res_array_kernel<<<src.cols(), 1>>>(src.getDev(), dev_res, src.rows(),
			src.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("get_res_array ");
	checkCudaErrors(
			cudaMemcpy(res + offset, dev_res, sizeof(int) * src.cols(),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaStreamSynchronize(0));
	checkCudaErrors(cudaFree(dev_res));
}

__global__ void set_label_kernel(int* dst, int *src, int num, int threadnum,
		int mid) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int off = bid * threadnum + tid;
	if (off < num) {
		dst[off] = src[off * (mid * 2 + 1) + mid];
	}
}

void set_label(int* label, int size, bool flag) {
	int *dev_label;
	int mid = Config::instance()->get_ngram() / 2;
	int num = size;
	checkCudaErrors(cudaMalloc((void** )&dev_label, sizeof(int) * num));
	int threadnum =
			Devices::instance()->maxThreadNum() > num ?
					num : Devices::instance()->maxThreadNum();
	int blocknum = num / threadnum + 1;
	dim3 blocks(blocknum);
	dim3 threads(threadnum);
	if (flag) {
		set_label_kernel<<<blocks, threads>>>(dev_label,
				Samples::instance()->get_trainY(), num, threadnum, mid);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("set_label");
	} else {
		set_label_kernel<<<blocks, threads>>>(dev_label,
				Samples::instance()->get_testY(), num, threadnum, mid);
		checkCudaErrors(cudaStreamSynchronize(0));
		getLastCudaError("set_label");
	}
	checkCudaErrors(
			cudaMemcpy(label, dev_label, sizeof(int) * num,
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaStreamSynchronize(0));
	checkCudaErrors(cudaFree(dev_label));
	getLastCudaError("set_label2");
}
