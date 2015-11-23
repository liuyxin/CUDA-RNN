#include "cuMatrix.h"

static int MAX_THREADNUM = Devices::instance()->max_ThreadsPerBlock();
//map<int, shared_ptr<MatData> > cuMatrix::TmpMemory;

//cublasHandle_t& getHandle() {
//	static cublasHandle_t handle = NULL;
//	if (handle == NULL) {
//		cublasStatus_t stat;
//		stat = cublasCreate(&handle);
//		if (stat != CUBLAS_STATUS_SUCCESS) {
//			printf("init: CUBLAS initialization failed\n");
//			exit(0);
//		}
//	}
//	return handle;
//}
map<int, shared_ptr<MatData>> cuMatrix::tmpMemory;
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
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}

	add_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator +(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}
	add_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + float");
	return res;
}

void cuMatrix::operator +=(cuMatrix cumat) {
	if (!size) {
		printf("cuMatrix error:: +=\n");
		exit(0);
	}
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	add_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
}

void cuMatrix::operator +=(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	add_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + float");
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
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}
	dec_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix - cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator -(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}
	dec_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix - float");
	return res;
}

void cuMatrix::operator -=(cuMatrix cumat) {
	if (!size) {
		printf("cuMatrix:: -= error\n");
	}
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	dec_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix - cuMatrix");
}

void cuMatrix::operator -=(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	dec_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix - float");
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
	assert(cumat.sizes() == sizes());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix * cuMatrix");
	return res;
}

void cuMatrix::Mul2(cuMatrix cumat,cuMatrix& dst){
	assert(cumat.sizes() == sizes());
	assert(cumat.sizes() == dst.sizes());
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
				cumat.data->getDev(), dst.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix * cuMatrix");
}

void cuMatrix::Mul2(float i ,cuMatrix& res){
	assert(sizes() == res.sizes());
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
				res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix::Mul2(float i ,cuMatrix& res)");
}

cuMatrix cuMatrix::operator *(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix * float");
	return res;
}
//res = this * cumat
cuMatrix cuMatrix::operator *(cuMatrix cumat) {
	assert(cols() == cumat.rows());
//	cuMatrix res(rows(), cumat.cols());
	cuMatrix res;
	int tmpSize = rows() * cumat.cols() * sizeof(float);
	if (tmpMemory.find(tmpSize) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[tmpSize], rows(), cumat.cols());
	} else {
		res = cuMatrix(rows(), cumat.cols());
		tmpMemory[tmpSize] = res.data;
	}
	float alpha = 1.0;
	float beta = 0.0;
	cublasHandle_t handle = NULL;
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("init: CUBLAS initialization failed\n");
		exit(0);
	}

	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cumat.cols(),
			rows(), cumat.rows(), &alpha, cumat.getDev(), cumat.cols(),
			getDev(), cols(), &beta, res.getDev(), res.cols());
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
	return res;
}

void cuMatrix::operator *=(float i) {
	assert(data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	mul_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix * float");
}

__global__ void div_kernel(float* dev_x, float* dev_y, float* dev_z, int cols,
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
__global__ void div_kernel(float* dev_x, float y_, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (y_ != 0) {
			dev_z[x * cols + y] = dev_x[x * cols + y] / y_;
		}
		else{
			dev_z[x * cols + y] = 0;
		}
		y += maxt;
	}
}

cuMatrix cuMatrix::operator /(cuMatrix cumat) {
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}
	div_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix / cuMatrix");
	return res;
}

cuMatrix cuMatrix::operator /(float i) {
	assert(data->getDev() != NULL);
	assert(i != 0);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], rows(), cols());
	} else {
		res = cuMatrix(rows(), cols());
		tmpMemory[size] = res.data;
	}
	div_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			res.data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix / float");
	return res;
}

void cuMatrix::operator /=(cuMatrix cumat) {
	assert(cumat.rows() == rows() && cumat.cols() == cols());
	assert(data->getDev() != NULL && cumat.data->getDev() != NULL);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	div_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(),
			cumat.data->getDev(), data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix / cuMatrix");
}

void cuMatrix::operator /=(float i) {
	assert(data->getDev() != NULL);
	assert(i != 0);
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	div_kernel<<<dim3(rows()), dim3(threadnum)>>>(data->getDev(), i,
			data->getDev(), cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix / float");
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
//	cuMatrix res(cols(), rows());
	cuMatrix res;
	if (tmpMemory.find(size) != tmpMemory.end()) {
		res = cuMatrix(tmpMemory[size], cols(), rows());
	} else {
		res = cuMatrix(cols(), rows());
		tmpMemory[size] = res.data;
	}
	int threadnum = MAX_THREADNUM > res.cols() ? res.cols() : MAX_THREADNUM;
	t_kernel<<<dim3(res.rows()), dim3(threadnum)>>>(data->getDev(),
			res.data->getDev(), res.rows(), res.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix / float");
	return res;
}

__global__ void Div_kernel(float x_, float* dev_y, float* dev_z, int cols,
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
cuMatrix operator /(float x, cuMatrix cumat) {
	cuMatrix res;
	if (cuMatrix::tmpMemory.find(cumat.sizes())
			!= cuMatrix::tmpMemory.end()) {
		res = cuMatrix(cuMatrix::tmpMemory[cumat.sizes()], cumat.rows(),
				cumat.cols());
	} else {
		res = cuMatrix(cumat.rows(), cumat.cols());
		cuMatrix::tmpMemory[cumat.sizes()] = res.data;
	}
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	Div_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(x, cumat.getDev(),
			res.getDev(), cumat.cols(), MAX_THREADNUM);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementDiv double matrix matrix ");
	return res;
}

__global__ void ReLU2_kernel(float* src, float* dst, int rows, int cols,
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

void cuMatrix::ReLU2(cuMatrix& cumat) {
	assert(sizes() == cumat.sizes());
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	ReLU2_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(getDev(),
			cumat.getDev(), cumat.rows(), cumat.cols(), threadnum);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ReLU2");
}

__global__ void square_kernel(float* dev_x, float y_, float* dev_z, int cols,
		int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], y_);
		y += maxt;
	}
}

void cuMatrix::Square2(cuMatrix& cumat){
	assert(sizes() == cumat.sizes());
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	square_kernel<<<dim3(rows()), dim3(threadnum)>>>( getDev(), 2.0f,
			cumat.getDev(), cols(), threadnum);
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementPow matrix float matrix ");

}


