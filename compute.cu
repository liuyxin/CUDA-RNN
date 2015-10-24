#include "compute.h"
static int MAX_THREADNUM = Devices::instance()->max_ThreadsPerBlock();

__global__ void ElementAdd_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] + dev_y[x * cols + y];
		y += maxt;
	}
}
__global__ void ElementAdd_kernel(double* dev_x, double y_, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] + y_;
		y += maxt;
	}
}

__global__ void ElementDec_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] - dev_y[x * cols + y];
		y += maxt;
	}
}
__global__ void ElementDec_kernel(double* dev_x, double y_, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] - y_;
		y += maxt;
	}
}

__global__ void ElementMul_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] * dev_y[x * cols + y];
		y += maxt;
	}
}

__global__ void ElementMul_kernel(double* dev_x, double y_, double* dev_z,
		int rows, int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
//		if((x*cols +y) > rows *cols)
//			printf("hello");
		dev_z[x * cols + y] = dev_x[x * cols + y] * y_;
		y += maxt;
	}
}

__global__ void ElementDiv_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_y[x * cols + y] != 0) {
			dev_z[x * cols + y] = dev_x[x * cols + y] / dev_y[x * cols + y];
		}
		y += maxt;
	}
}

__global__ void ElementDiv_kernel(double* dev_x, double y_, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (y_ != 0) {
			dev_z[x * cols + y] = dev_x[x * cols + y] / y_;
		}
		y += maxt;
	}
}

__global__ void ElementDiv_kernel(double x_, double* dev_y, double* dev_z,
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

__global__ void ElementPow_kernel(double* dev_x, double* dev_y, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], dev_y[x * cols + y]);
		y += maxt;
	}
}

__global__ void ElementPow_kernel(double* dev_x, double y_, double* dev_z,
		int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], y_);
		y += maxt;
	}
}
__global__ void drop_kernel(double* dev_x, int cols,int tmax) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_x[x * cols + y] > 10) {
			dev_x[x * cols + y] = dev_x[x * cols + y] - (int) dev_x[x * cols + y];
		}
		y += tmax;
	}

}

void ElementAdd(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z) {
	if (!((x->rows == y->rows) && (y->rows == z->rows))) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!((x->cols == y->cols) && (y->cols == z->cols))) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementAdd_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(),
			y->getDev(), z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementAdd matrix matrix matrix ");
}
void ElementAdd(cuMatrix<double>* x, double y, cuMatrix<double>* z) {
	if (!(x->rows == z->rows)) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!(x->cols == z->cols)) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementAdd_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(), y,
			z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementAdd matrix double matrix ");
}

void ElementDec(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z) {
	if (!((x->rows == y->rows) && (y->rows == z->rows))) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!((x->cols == y->cols) && (y->cols == z->cols))) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementDec_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(),
			y->getDev(), z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementDec matrix matrix matrix ");
}
void ElementDec(cuMatrix<double>* x, double y, cuMatrix<double>* z) {
	if (!(x->rows == z->rows)) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!(x->cols == z->cols)) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementDec_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(), y,
			z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementDec matrix double matrix ");
}

void ElementMul(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z) {
	if (!((x->rows == y->rows) && (y->rows == z->rows))) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!((x->cols == y->cols) && (y->cols == z->cols))) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementMul_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(),
			y->getDev(), z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementMul matrix matrix matrix ");
}

void ElementMul(cuMatrix<double>* x, double y, cuMatrix<double>* z) {
	if (!(x->rows == z->rows)) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!(x->cols == z->cols)) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementMul_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(), y,
			z->getDev(), x->rows, x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementMul matrix double matrix ");
}

void ElementDiv(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z) {
	if (!((x->rows == y->rows) && (y->rows == z->rows))) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!((x->cols == y->cols) && (y->cols == z->cols))) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementDiv_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(),
			y->getDev(), z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementDiv matrix matrix matrix ");
}

void ElementDiv(cuMatrix<double>* x, double y, cuMatrix<double>* z) {
	if (!(x->rows == z->rows)) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!(x->cols == z->cols)) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementDiv_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(), y,
			z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementDiv matrix double matrix ");
}
void ElementDiv(double x, cuMatrix<double>* y, cuMatrix<double>* z) {
	if (!(y->rows == z->rows)) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!(y->cols == z->cols)) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > y->cols ? y->cols : MAX_THREADNUM;
	ElementDiv_kernel<<<dim3(y->rows), dim3(threadnum)>>>(x, y->getDev(),
			z->getDev(), y->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementDiv double matrix matrix ");

}

void ElementPow(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z) {
	if (!((x->rows == y->rows) && (y->rows == z->rows))) {
		printf(
				"void ElementPow(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z)error: rows!\n");
		exit(0);
	}
	if (!((x->cols == y->cols) && (y->cols == z->cols))) {
		printf(
				"void ElementPow(cuMatrix<double>* x, cuMatrix<double>* y, cuMatrix<double>* z) error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementPow_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(),
			y->getDev(), z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementPow matrix matrix matrix ");
}

void ElementPow(cuMatrix<double>* x, double y, cuMatrix<double>* z) {
	if (!(x->rows == z->rows)) {
		printf(
				"ElementPow(cuMatrix<double>* x, double y, cuMatrix<double>* z) error: rows!\n");
		exit(0);
	}
	if (!(x->cols == z->cols)) {
		printf(
				"ElementPow(cuMatrix<double>* x, double y, cuMatrix<double>* z) error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	ElementPow_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(), y,
			z->getDev(), x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementPow matrix double matrix ");
}

__global__ void ReLU_kernel(double* dev_a, int rows, int cols, int maxt) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		assert(x * cols + y < rows * cols);
		if (dev_a[x * cols + y] <= 0) {
			dev_a[x * cols + y] = 0;
		}
		y += maxt;
	}
}

void ReLU(cuMatrix<double>* cm) {
	int threadnum = MAX_THREADNUM > cm->cols ? cm->cols : MAX_THREADNUM;
	ReLU_kernel<<<dim3(cm->rows), dim3(threadnum)>>>(cm->getDev(), cm->rows,
			cm->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ReLU ");
}

__global__ void dReLU_kernel(double* src, double* dst, int cols, int maxt) {
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

cuMatrix<double>* dReLU(cuMatrix<double>* cm) {
	cuMatrix<double>* res = new cuMatrix<double>(cm->getHost(), cm->rows,
			cm->cols, 1);
	int threadnum = MAX_THREADNUM > cm->cols ? cm->cols : MAX_THREADNUM;
	dReLU_kernel<<<dim3(cm->rows), dim3(threadnum)>>>(cm->getDev(),
			res->getDev(), cm->cols, MAX_THREADNUM);
	getLastCudaError("dReLU ");
	return res;
}

__global__ void reduce_max_kernel(double* dev_x, double* dev_y, int rows,
		int cols, int maxt) {
	int tid = threadIdx.x;
	while (tid < cols) {
		double max = (double) LONG_MIN;
		for (int i = 0; i < rows; i++) {
			max = max > dev_x[i * cols + tid] ? max : dev_x[i * cols + tid];
		}
		for (int i = 0; i < rows; i++) {
			dev_y[i * cols + tid] = max;
		}
		tid += maxt;
	}
}

void reduce_max(cuMatrix<double>* src, cuMatrix<double>* dst) {
	if (!(src->rows == dst->rows)) {
		printf("reduce_max error: rows!\n");
		exit(0);
	}
	if (!(src->cols == dst->cols)) {
		printf("reduce_max error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > src->cols ? src->cols : MAX_THREADNUM;
	reduce_max_kernel<<<dim3(1), dim3(threadnum)>>>(src->getDev(),
			dst->getDev(), src->rows, src->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("reduce_max ");
}
//  share memory?
__global__ void reduce_sum_kernel(double* dev_x, double* dev_y, int rows,
		int cols, int maxt) {
	int tidx = blockIdx.x;
	int tidy = threadIdx.x;
	double sum = 0;
	while (tidy < cols) {
		for (int i = 0; i < rows; i++) {
			sum += dev_x[i * cols + tidy];
		}
		dev_y[tidx * cols + tidy] = sum;
		tidy += maxt;
	}
}

void reduce_sum(cuMatrix<double>* src, cuMatrix<double>* dst) {
	if (!(src->rows == dst->rows)) {
		printf("reduce_sum error: rows!\n");
		exit(0);
	}
	if (!(src->cols == dst->cols)) {
		printf("reduce_sum error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > src->cols ? src->cols : MAX_THREADNUM;
	reduce_sum_kernel<<<dim3(src->rows), dim3(threadnum)>>>(src->getDev(),
			dst->getDev(), src->rows, src->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("reduce_sum");
}

__global__ void exp_mat_kernel(double* dev_x, double* dev_y, int cols,
		int maxt) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols) {
		dev_y[bid * cols + tid] = exp(dev_x[bid * cols + tid]);
		tid += maxt;
	}
}

void ElementExp(cuMatrix<double>* src, cuMatrix<double>* dst) {
	if (!(src->rows == dst->rows)) {
		printf("ElementExp error: rows!\n");
		exit(0);
	}
	if (!(src->cols == dst->cols)) {
		printf("ElementExp error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > src->cols ? src->cols : MAX_THREADNUM;
	exp_mat_kernel<<<dim3(src->rows), dim3(threadnum)>>>(src->getDev(),
			dst->getDev(), src->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementExp");
}

__global__ void maxtri_sum_kernel(double* src, double* res, int threadnum,
		int cols) {
	extern __shared__ double cache[];
	int tid = threadIdx.x;
	int bid = blockIdx.x;
//	int m = blockDim.x;
	int m = threadnum;
	cache[tid] = src[bid * cols + tid];
	while (tid + m < cols) {
		cache[tid] += src[bid * cols + tid + m];
		tid += m;
	}
	tid = threadIdx.x;
	__syncthreads();

	while (m >>= 1, m) {
		if (tid < m) {
			cache[tid] = cache[tid] + cache[tid + m];
		}
		if ((tid == 0) && (m != 1) && (m & 1)) {
			cache[tid] += cache[m - 1];
		}
		__syncthreads();
	}
	if (tid == 0)
		res[bid] = cache[0];
}

double matrix_sum(cuMatrix<double>* src) {
	double* res = (double*) malloc(sizeof(double) * src->rows);
	double *res_dev;
	int threadnum = MAX_THREADNUM > src->cols ? src->cols : MAX_THREADNUM;
	assert(!(threadnum & 1));
	checkCudaErrors(cudaMalloc((void** ) &res_dev, sizeof(double) * src->rows));
	if (Devices::instance()->get_prop().sharedMemPerBlock
			< sizeof(double) * threadnum) {
		threadnum = Devices::instance()->get_prop().sharedMemPerBlock
				/ sizeof(double);
	}
	maxtri_sum_kernel<<<dim3(src->rows), dim3(threadnum),
			sizeof(double) * threadnum>>>(src->getDev(), res_dev, threadnum,
			src->cols);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("matrix_sum");
	checkCudaErrors(
			cudaMemcpy(res, res_dev, sizeof(double) * src->rows,
					cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(res_dev));
	double _res = 0.0;
	for (int i = 0; i < src->rows; i++)
		_res += res[i];
	free(res);
	return _res;
}

__global__ void log_mat_kernel(double* dev_x, double* dev_y, int cols,
		int maxt) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols) {
		dev_y[bid * cols + tid] = log(dev_x[bid * cols + tid]);
		tid += maxt;
	}
}

void ElementLog(cuMatrix<double>* src, cuMatrix<double>* dst) {
	if (!(src->rows == dst->rows)) {
		printf("ElementAdd error: rows!\n");
		exit(0);
	}
	if (!(src->cols == dst->cols)) {
		printf("ElementAdd error: cols!\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > src->cols ? src->cols : MAX_THREADNUM;
	log_mat_kernel<<<dim3(src->rows), dim3(threadnum)>>>(src->getDev(),
			dst->getDev(), src->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementLog");
}

void drop(cuMatrix<double>* x) {
	int threadnum = MAX_THREADNUM > x->cols ? x->cols : MAX_THREADNUM;
	drop_kernel<<<dim3(x->rows), dim3(threadnum)>>>(x->getDev(),
			 x->cols, MAX_THREADNUM);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("ElementPow matrix matrix matrix ");
}

