#include "prepare.cuh"
static int MAX_THREADNUM = Devices::instance()->max_ThreadsPerBlock();
void set_gpudata(int* host_, void** dev_, int size) {
	MemoryMonitor::instance()->gpuMalloc(dev_, size);
	cudaError_t cudaStat;
	cudaStat = cudaMemcpy(dev_, host_, size, cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) {
		printf("set_gpudata::toGPU data upload failed\n");
		MemoryMonitor::instance()->freeGpuMemory(*dev_);
		exit(0);
	}
}

__global__ void set_sampleY_kernel(double* sampleY, int* src, int* dev_ran,
		int cols, int ngram) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	sampleY[tid * cols + bid] = src[dev_ran[bid] * ngram + tid];
}

__global__ void set_acti0_kernel(double** acti0, int* src, int* dev_ran,
		int cols, int ngram) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	double *p = acti0[tid];
	int n = src[dev_ran[bid] * ngram + tid];
	p[n * cols + bid] = 1;
}

void set_acti0(cuMatrixVector<double>& acti0, int *inputx, int inputsizex,
		cuMatrix<double>& sampleY, int*inputy, int inputsizey) {
	int bs = Config::instance()->get_batch_size();
	int ngram = Config::instance()->get_ngram();
	int *dev_ran = NULL;
	Config::instance()->randproductor_init();
	MemoryMonitor::instance()->gpuMalloc((void**) &dev_ran, bs * sizeof(int));
	checkCudaErrors(
			cudaMemcpy(dev_ran, Config::instance()->get_rand(1),
					bs * sizeof(int), cudaMemcpyHostToDevice));
	Config::instance()->inputx2gpu(inputx, inputsizex);
	Config::instance()->inputy2gpu(inputy, inputsizey);
	dim3 block = dim3(bs);
	dim3 thread = dim3(ngram);
	set_acti0_kernel<<<block, thread>>>(acti0.get_devPoint(),
			Config::instance()->get_dev_inputX(), dev_ran, bs, ngram);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("set_acti0_kernel-1 ");
//	for (int i = 0; i < 5; i++) {
//		acti0[i]->toCpu();
//	}
	set_sampleY_kernel<<<block, thread>>>(sampleY.getDev(),
			Config::instance()->get_dev_inputY(), dev_ran, bs, ngram);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("set_sampleY_kernel-1 ");
	cudaFree(dev_ran);
//	sampleY.toCpu();
}

void set_acti0(cuMatrixVector<double>& acti0, cuMatrix<double>& sampleY) {
	int bs = Config::instance()->get_batch_size();
	int ngram = Config::instance()->get_ngram();
	int *dev_ran = NULL;
	Config::instance()->randproductor_init();
	MemoryMonitor::instance()->gpuMalloc((void**) &dev_ran, bs * sizeof(int));
	checkCudaErrors(
			cudaMemcpy(dev_ran, Config::instance()->get_rand(1),
					bs * sizeof(int), cudaMemcpyHostToDevice));
	dim3 block = dim3(bs);
	dim3 thread = dim3(ngram);
	set_acti0_kernel<<<block, thread>>>(acti0.get_devPoint(),
			Config::instance()->get_dev_inputX(), dev_ran, bs, ngram);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("set_acti0_kernel-2");
	set_sampleY_kernel<<<block, thread>>>(sampleY.getDev(),
			Config::instance()->get_dev_inputY(), dev_ran, bs, ngram);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("set_sampleY_kernel-2");
	checkCudaErrors(cudaFree(dev_ran));
}

__global__ void set_gt_kernel(double** gt_, double* y, int rows, int cols) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	assert(tid < rows && bid < cols);
	double* p = gt_[tid];
	int i = y[tid * cols + bid];
	assert(i < 10);
	p[i * cols + bid] = 1.0;
}

void set_groundtruth(cuMatrixVector<double>& gt, cuMatrix<double>& sampleY) {
	dim3 block = dim3(sampleY.cols);
	dim3 thread = dim3(sampleY.rows);
	set_gt_kernel<<<block, thread>>>(gt.get_devPoint(), sampleY.getDev(),
			sampleY.rows, sampleY.cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("set_groundtruth ");
}

__global__ void getDataMat_kernel(double** sampleX, int* src, int off, int cols,
		int ngram) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	double *p = sampleX[tid];
	int n = src[(off + bid) * ngram + tid];
	p[n * cols + bid] = 1.0;
}
void getDataMat(cuMatrixVector<double> &sampleX, int off, int bs, int n,
		const int size) {
	if (sampleX.size() != 0) {
		for (int i = 0; i < sampleX.size(); i++)
			delete sampleX[i];
	}
	sampleX.clear();
	int ngram = Config::instance()->get_ngram();
	for (int i = 0; i < Config::instance()->get_ngram(); i++) {
		cuMatrix<double> *tmp = new cuMatrix<double>(n, bs, 1);
		sampleX.push_back(tmp);
	}
	sampleX.toGpu();
	dim3 thread = dim3(ngram);
	dim3 block = dim3(bs);
	if (size > 5000) {
		getDataMat_kernel<<<block, thread>>>(sampleX.get_devPoint(),
				Config::instance()->get_dev_inputX(), off, bs, ngram);
	} else {
		getDataMat_kernel<<<block, thread>>>(sampleX.get_devPoint(),
				Config::instance()->get_dev_testX(), off, bs, ngram);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("getDataMat_kernel ");
}

__global__ void get_res_array_kernel(double* src, int* dev_res, int rows,
		int cols) {
	int bid = blockIdx.x;
	double max = src[bid];
	dev_res[bid] = 0;
	for (int i = 1; i < rows; i++) {
		if (max < src[i * cols + bid]) {
			max = src[i * cols + bid];
			dev_res[bid] = i;
		}
	}
}

void get_res_array(cuMatrix<double> *src, int *res, int offset) {
	int *dev_res;
	checkCudaErrors(cudaMalloc((void** )&dev_res, sizeof(int) * src->cols));
	get_res_array_kernel<<<src->cols, 1>>>(src->getDev(), dev_res, src->rows,
			src->cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("get_res_array ");
	checkCudaErrors(
			cudaMemcpy(res + offset, dev_res, sizeof(int) * src->cols,
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
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

void set_label(int* label, const int size) {
	int *dev_label;
	int mid = Config::instance()->get_ngram() / 2;
	int num = size;
	checkCudaErrors(cudaMalloc((void** )&dev_label, sizeof(int) * num));
	int threadnum = MAX_THREADNUM > num ? num : MAX_THREADNUM;
	int blocknum = num / threadnum + 1;
	dim3 blocks(blocknum);
	dim3 threads(threadnum);
	if (num > 5000) {
		set_label_kernel<<<blocks, threads>>>(dev_label,
				Config::instance()->get_dev_inputY(), num, threadnum, mid);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("set_label");
	} else {
		set_label_kernel<<<blocks, threads>>>(dev_label,
				Config::instance()->get_dev_testY(), num, threadnum, mid);
		checkCudaErrors(cudaDeviceSynchronize());
		getLastCudaError("set_label");
	}
	checkCudaErrors(
			cudaMemcpy(label, dev_label, sizeof(int) * num,
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(dev_label));
	getLastCudaError("set_label2");

}
