#include "cuMatrix.h"

static int MAX_THREADNUM = Devices::instance()->max_ThreadsPerBlock();

cublasHandle_t& getHandle() {
	static cublasHandle_t handle = NULL;
	if (handle == NULL) {
		cublasStatus_t stat;
		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf("init: CUBLAS initialization failed\n");
			exit(0);
		}
	}
	return handle;
}

__global__ void add_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] + dev_y[x * cols + y];
		y += maxt;
	}
}
__global__ void add_kernel(float* dev_x, float y_, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] + y_;
		y += maxt;
	}
}

cuMatrix cuMatrix::operator +(cuMatrix cumat) {
	if (!size) {
		if (cumat.data->getDev() == NULL) {
			printf("cuMatrix error : both matrix are empty.\n");
			exit(0);
		}
		cuMatrix res = cumat;
		return res;
	}
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	add_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator +(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	add_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix + float");
	return res;
}

__global__ void dec_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] - dev_y[x * cols + y];
		y += maxt;
	}
}
__global__ void dec_kernel(float* dev_x, float y_, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] - y_;
		y += maxt;
	}
}

cuMatrix cuMatrix::operator -(cuMatrix cumat) {
	if (!size) {
		if (cumat.data->getDev() == NULL) {
			printf("cuMatrix error : both matrix are empty.\n");
			exit(0);
		}
		cuMatrix res = cumat * -1.0f;

		return res;
	}
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dec_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix - cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator -(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	dec_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix - float");
	return res;
}

__global__ void mul_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] * dev_y[x * cols + y];
		y += maxt;
	}
}
__global__ void mul_kernel(float* dev_x, float y_, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] * y_;
		y += maxt;
	}
}

cuMatrix cuMatrix::Mul(cuMatrix cumat) {
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix * cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator *(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix * float");
	return res;
}
//res = this * cumat

//res = this * cumat
cuMatrix cuMatrix::operator *(cuMatrix cumat) {
	assert(cols() == cumat.rows());
	cuMatrix res(rows(), cumat.cols());
	float alpha = 1.0;
	float beta = 0.0;
	cublasStatus_t stat;
	stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, cumat.cols(),
			rows(), cumat.rows(), &alpha, cumat.getDev(), cumat.cols(),
			getDev(), cols(), &beta, res.getDev(), res.cols());
	cudaStreamSynchronize(0);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("cuMatrix::Mul() error\n");
		exit(0);
	}
	return res;
}

__global__ void div_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_y[x * cols + y] != 0) {
			dev_z[x * cols + y] = dev_x[x * cols + y] / dev_y[x * cols + y];
		}
		y += maxt;
	}
}
__global__ void div_kernel(float* dev_x, float y_, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (y_ != 0) {
			dev_z[x * cols + y] = dev_x[x * cols + y] / y_;
		}
		y += maxt;
	}
}

cuMatrix cuMatrix::operator /(cuMatrix cumat) {
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	div_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix / cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator /(float i) {
	assert(data->getDev() != NULL);
	assert(i != 0);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res(rows(), cols());
	div_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix / float");
	return res;
}

__global__ void t_kernel(float* dev_src, float* dev_res, int res_r, int res_c,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < res_c) {
		dev_res[x * res_c + y] = dev_src[y * res_r + x];
		y += maxt;
	}
}

cuMatrix cuMatrix::t() {
	assert(cols() != 0 && rows() != 0);
	cuMatrix res(cols(), rows());
	int threadnum = MAX_THREADNUM > res.cols() ? res.cols() : MAX_THREADNUM;
	t_kernel<<<dim3(res.rows()), dim3(threadnum)>>>(data->getDev(),
			res.data->getDev(), res.rows(), res.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("pre-element add cuMatrix / float");
	return res;
}

__global__ void Div_kernel(float x_, float* dev_y, float* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_y[x * cols + y] != 0) {
			dev_z[x * cols + y] = x_ / dev_y[x * cols + y];
		}
		y += maxt;
	}
}
cuMatrix operator /(float x, cuMatrix cumat) {
	cuMatrix res(cumat.rows(), cumat.cols());
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	Div_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(x, cumat.getDev(),
			res.getDev(), cumat.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementDiv double matrix matrix ");
	return res;
}

//__global__ void ReLU_kernel(float* src, float* dst, int rows, int cols,
//		int maxt) {
//	int x = blockIdx.x;
//	int y = threadIdx.x;
//	while (y < cols) {
//		assert(x * cols + y < rows * cols);
//		if (src[x * cols + y] <= 0) {
//			dst[x * cols + y] = 0;
//		} else {
//			dst[x * cols + y] = src[x * cols + y];
//		}
//		y += maxt;
//	}
//}
//
//cuMatrix ReLU(cuMatrix& cumat) {
//	cuMatrix res(cumat.rows(), cumat.cols());
//	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
//	ReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(),
//			res.getDev(), cumat.rows(), cumat.cols(), MAX_THREADNUM);
//	checkCudaErrors(cudaDeviceSynchronize());
//	getLastCudaError("ReLU ");
//	return res;
//}
//
//__global__ void dReLU_kernel(float* src, float* dst, int cols, int maxt) {
//	int x = blockIdx.x;
//	int y = threadIdx.x;
//	while (y < cols) {
//		if (src[x * cols + y] <= 0) {
//			dst[x * cols + y] = 0;
//		} else {
//			dst[x * cols + y] = 1;
//		}
//		y += maxt;
//	}
//}
//
//cuMatrix dReLU(cuMatrix cumat) {
//	cuMatrix res(cumat.rows(), cumat.cols());
//	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
//	dReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(),
//			res.getDev(), cumat.cols(), MAX_THREADNUM);
//	getLastCudaError("dReLU ");
//	return res;
//}
//
//__global__ void reduce_max_kernel(float* dev_x, float* dev_y, int rows,
//		int cols, int maxt) {
//	int tid = threadIdx.x;
//	while (tid < cols) {
//		float max = (float) LONG_MIN;
//		for (int i = 0; i < rows; i++) {
//			max = max > dev_x[i * cols + tid] ? max : dev_x[i * cols + tid];
//		}
//		for (int i = 0; i < rows; i++) {
//			dev_y[i * cols + tid] = max;
//		}
//		tid += maxt;
//	}
//}
//
//cuMatrix reduceMax(cuMatrix src) {
//	cuMatrix res(src.rows(), src.cols());
//	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
//	reduce_max_kernel<<<dim3(1), dim3(threadnum)>>>(src.getDev(),
//			res.getDev(), src.rows(), src.cols(), MAX_THREADNUM);
//	checkCudaErrors(cudaDeviceSynchronize());
//	getLastCudaError("reduce_max");
//	return res;
//}
////  share memory?
//__global__ void reduce_sum_kernel(float* dev_x, float* dev_y, int rows,
//		int cols, int maxt) {
//	int tidx = blockIdx.x;
//	int tidy = threadIdx.x;
//	float sum = 0;
//	while (tidy < cols) {
//		for (int i = 0; i < rows; i++) {
//			sum += dev_x[i * cols + tidy];
//		}
//		dev_y[tidx * cols + tidy] = sum;
//		tidy += maxt;
//	}
//}
//
//cuMatrix reduceSum(cuMatrix src) {
//	cuMatrix res(src.rows(), src.cols());
//	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
//	reduce_sum_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
//			res.getDev(), src.rows(), src.cols(), MAX_THREADNUM);
//	checkCudaErrors(cudaDeviceSynchronize());
//	getLastCudaError("reduce_sum");
//	return res;
//}
