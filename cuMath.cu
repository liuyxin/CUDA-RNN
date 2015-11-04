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
	cuMatrix res(cumat.rows(), cumat.cols());
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	ReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(),
			res.getDev(), cumat.rows(), cumat.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
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
	cuMatrix res(cumat.rows(), cumat.cols());
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	dReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(),
			res.getDev(), cumat.cols(), MAX_THREADNUM);
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

cuMatrix reduceMax(cuMatrix src) {
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	reduce_max_kernel<<<dim3(1), dim3(threadnum)>>>(src.getDev(), res.getDev(),
			src.rows(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
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

cuMatrix reduceSum(cuMatrix src) {
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	reduce_sum_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.rows(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
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

cuMatrix Log(cuMatrix src) {
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	log_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
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

cuMatrix Exp(cuMatrix src) {
	cuMatrix res(src.rows(), src.cols());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	exp_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(), res.getDev(),
			src.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementExp");
	return res;
}

__global__ void Pow_kernel(float* dev_x, float* dev_y, float* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], dev_y[x * cols + y]);
		y += maxt;
	}
}

__global__ void Pow_kernel(float* dev_x, float y_, float* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], y_);
		y += maxt;
	}
}

cuMatrix Pow(cuMatrix x,cuMatrix y) {
	if (!(x.rows() == y.rows())) {
		printf(
				"cuMatrix Pow(cuMatrix x,cuMatrix y) error: rows!\n");
		exit(0);
	}
	if (!(x.cols() == y.cols())) {
		printf(
				"cuMatrix Pow(cuMatrix x,cuMatrix y) error: cols!\n");
		exit(0);
	}
	cuMatrix res(x.rows(), x.cols());
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	Pow_kernel<<<dim3(x.rows()), dim3(threadnum)>>>(x.getDev(),
			y.getDev(), res.getDev(), x.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("cuMatrix Pow(cuMatrix x,cuMatrix y)");
	return res;
}

cuMatrix Pow(cuMatrix x,float y) {
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	cuMatrix res(x.rows(), x.cols());
	Pow_kernel<<<dim3(x.rows()), dim3(threadnum)>>>(x.getDev(), y,
			res.getDev(), x.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementPow matrix float matrix ");
	return res;
}
