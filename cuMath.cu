#include "cuMath.h"
static int MAX_THREADNUM = Devices::instance()->max_ThreadsPerBlock();
__global__ void ReLU_kernel(float* src, float* dst, int rows, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		assert(x * cols + y < rows * cols);
		if (src[x * cols + y] <= 0) {
			dst[x * cols + y] = 0;
		} else {
			dst[x * cols + y] = src[x * cols + y];
		}
		y += maxt;
	}
}

cuMatrix ReLU(cuMatrix& cumat) {
//	cuMatrix res(cumat.rows(), cumat.cols());
	cuMatrix res;
	int size = cumat.sizes();
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], cumat.rows(), cumat.cols());
	} else {
		res = cuMatrix(cumat.rows(), cumat.cols());
		cuMatrix::tmpMemory[size] = cumat.data;
	}
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	ReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(),
			res.getDev(), cumat.rows(), cumat.cols(), threadnum);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ReLU ");
	return res;
}

__global__ void dReLU_kernel(float* src, float* dst, int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (src[x * cols + y] <= 0) {
			dst[x * cols + y] = 0;
		} else {
			dst[x * cols + y] = 1;
		}
		y += maxt;
	}
}

cuMatrix dReLU(cuMatrix& cumat) {
	cuMatrix res;
	int size = cumat.sizes();
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], cumat.rows(), cumat.cols());
	} else {
		res = cuMatrix(cumat.rows(), cumat.cols());
		cuMatrix::tmpMemory[size] = cumat.data;
	}
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	dReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(),
			res.getDev(), cumat.cols(), threadnum);
	getLastCudaError("dReLU ");
	return res;
}

__global__ void reduce_max_kernel(float* dev_x, float* dev_y, int rows,
		int cols, int maxt) {
	int tid = threadIdx.x;
	while (tid < cols) {
		float max = (float) LONG_MIN;
		for (int i = 0; i < rows; i++) {
			max = max > dev_x[i * cols + tid] ? max : dev_x[i * cols + tid];
		}
		for (int i = 0; i < rows; i++) {
			dev_y[i * cols + tid] = max;
		}
		tid += maxt;
	}
}

cuMatrix reduceMax(cuMatrix& src) {
	cuMatrix res;
	int size = src.sizes();
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		cuMatrix::tmpMemory[size] = res.data;
	}
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	reduce_max_kernel<<<dim3(1), dim3(threadnum)>>>(src.getDev(), res.getDev(),
			src.rows(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("reduce_max");
	return res;
}
//  share memory?
__global__ void reduce_sum_kernel(float* dev_x, float* dev_y, int rows,
		int cols, int maxt) {
	int tidx = blockIdx.x;
	int tidy = threadIdx.x;
	float sum = 0;
	while (tidy < cols) {
		for (int i = 0; i < rows; i++) {
			sum += dev_x[i * cols + tidy];
		}
		dev_y[tidx * cols + tidy] = sum;
		tidy += maxt;
	}
}

cuMatrix reduceSum(cuMatrix& src) {
	cuMatrix res;
	int size = src.sizes();
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		cuMatrix::tmpMemory[size] = res.data;
	}
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	reduce_sum_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.rows(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("reduce_sum");
	return res;
}

__global__ void log_kernel(float* dev_x, float* dev_y, int cols, int maxt) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols) {
		dev_y[bid * cols + tid] = log(dev_x[bid * cols + tid]);
		tid += maxt;
	}
}

cuMatrix Log(cuMatrix& src) {
	cuMatrix res;
	int size = src.sizes();
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		cuMatrix::tmpMemory[size] = res.data;
	}
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	log_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementLog");
	return res;
}

__global__ void exp_kernel(float* dev_x, float* dev_y, int cols, int maxt) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols) {
		dev_y[bid * cols + tid] = exp(dev_x[bid * cols + tid]);
		tid += maxt;
	}
}

cuMatrix Exp(cuMatrix& src) {
//	cuMatrix res(src.rows(), src.cols());
	cuMatrix res;
	int size = src.sizes();
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		cuMatrix::tmpMemory[size] = res.data;
	}
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	exp_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(), res.getDev(),
			src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementExp");
	return res;
}

void Exp(cuMatrix& src, cuMatrix& res) {
	assert(src.sizes() == res.sizes());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	exp_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(), res.getDev(),
			src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementExp");
}

__global__ void Pow_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], dev_y[x * cols + y]);
		y += maxt;
	}
}

__global__ void Pow_kernel(float* dev_x, float y_, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], y_);
		y += maxt;
	}
}

cuMatrix Pow(cuMatrix x, cuMatrix y) {
	if (!(x.rows() == y.rows())) {
		printf("cuMatrix Pow(cuMatrix x,cuMatrix y) error: rows!\n");
		exit(0);
	}
	if (!(x.cols() == y.cols())) {
		printf("cuMatrix Pow(cuMatrix x,cuMatrix y) error: cols!\n");
		exit(0);
	}
	int size = x.sizes();
	cuMatrix res;
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], x.rows(), x.cols());
	} else {
		res = cuMatrix(x.rows(), x.cols());
		cuMatrix::tmpMemory[size] = res.data;
	}
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	Pow_kernel<<<dim3(x.rows()), dim3(threadnum)>>>(x.getDev(),
			y.getDev(), res.getDev(), x.cols(), threadnum);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix Pow(cuMatrix x,cuMatrix y)");
	return res;
}

cuMatrix Pow(cuMatrix x, float y) {
	int size = x.sizes();
	cuMatrix res;
	if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[size], x.rows(), x.cols());
	} else {
		res = cuMatrix(x.rows(), x.cols());
		cuMatrix::tmpMemory[size] = res.data;
	}
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	Pow_kernel<<<dim3(x.rows()), dim3(threadnum)>>>(x.getDev(), y,
			res.getDev(), x.cols(), threadnum);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementPow matrix float matrix ");
	return res;
}

//dst = src1 * src2
void cuMultiplication(cuMatrix src1, cuMatrix src2, cuMatrix dst) {
	assert(src1.cols() == src2.rows());
	assert(src1.rows() == dst.rows() && src2.cols() == dst.cols());
	float alpha = 1.0;
	float beta = 0.0;
	cublasHandle_t handle = NULL;
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("init: CUBLAS initialization failed\n");
		exit(0);
	}
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, src2.cols(),
			src1.rows(), src2.rows(), &alpha, src2.getDev(), src2.cols(),
			src1.getDev(), src1.cols(), &beta, dst.getDev(), dst.cols());
	cudaStreamSynchronize(0);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("cuMatrix::Mul() error\n");
		exit(0);
	}
	stat = cublasDestroy(handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("init: CUBLAS destory failed\n");
		exit(0);
	}
}

__global__ void cuPlus_kernel(float* dev_x, float* dev_y, float* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] + dev_y[x * cols + y];
		y += maxt;
	}
}

//dst = src1 + src2
void cuPlus(cuMatrix src1, cuMatrix src2, cuMatrix dst) {
	assert(src1.sizes() == src2.sizes() && dst.sizes() == src1.sizes());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	cuPlus_kernel<<<dim3(src1.rows()), dim3(threadnum)>>>(src1.getDev(),
			src2.data->getDev(), dst.data->getDev(), src1.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
}

__global__ void cuDec_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] - dev_y[x * cols + y];
		y += maxt;
	}
}

//dst = src1 - src2
void cuDec(cuMatrix src1, cuMatrix src2, cuMatrix dst) {
	assert(src1.sizes() == src2.sizes() && dst.sizes() == src1.sizes());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	cuDec_kernel<<<dim3(src1.rows()), dim3(threadnum)>>>(src1.getDev(),
			src2.data->getDev(), dst.data->getDev(), src1.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
}

__global__ void cuDiv_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_y[x * cols + y] != 0) {
			dev_z[x * cols + y] = dev_x[x * cols + y] / dev_y[x * cols + y];
		}
		else{
			dev_z[x * cols + y] = 0;
		}
		y += maxt;
	}
}
void cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst){
	assert(src1.sizes() == src2.sizes());
	assert(src1.sizes() == dst.sizes());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	cuDiv_kernel<<<dim3(src1.rows()), dim3(threadnum)>>>(src1.getDev(),
			src2.getDev(), dst.getDev(), src1.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst)");
}

__global__ void cuDiv_kernel(float x_, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_y[x * cols + y] != 0) {
			dev_z[x * cols + y] = x_ / dev_y[x * cols + y];
		}
		else{
			dev_z[x * cols + y] = 0;
		}
		y += maxt;
	}
}
void cuDiv(float src1, cuMatrix src2, cuMatrix dst){
	assert(src2.sizes() == dst.sizes());
	int threadnum = MAX_THREADNUM > src2.cols() ? src2.cols() : MAX_THREADNUM;
	cuDiv_kernel<<<dim3(src2.rows()), dim3(threadnum)>>>(src1,
			src2.getDev(), dst.getDev(), src2.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst)");
}
