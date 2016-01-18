#include "cuMath.h"
const int MAX_THREADNUM = Devices::instance()->maxThreadNum();
const int* blockdim = Devices::instance()->blockDim();
__global__ void ReLU_kernel(float* src, float* dst, int rows, int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		assert(x * cols + y < rows * cols);
		if (src[x * cols + y] <= 0) {
			dst[x * cols + y] = 0;
		} else {
			dst[x * cols + y] = src[x * cols + y];
		}
		y += blockDim.x;
	}
}

cuMatrix ReLU(cuMatrix& cumat) {
	//	cuMatrix res(cumat.rows(), cumat.cols());
	cuMatrix res;
	int size = cumat.sizes();
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, cumat.rows(), cumat.cols());
	} else {
		res = cuMatrix(cumat.rows(), cumat.cols());
		mem.set(res.data);
	}	
	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	ReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(),
			res.getDev(), cumat.rows(), cumat.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ReLU ");
	return res;
}

__global__ void dReLU_kernel(float* src, float* dst, int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (src[x * cols + y] <= 0) {
			dst[x * cols + y] = 0;
		} else {
			dst[x * cols + y] = 1;
		}
		y += blockDim.x;
	}
}

cuMatrix dReLU(cuMatrix& cumat) {
	cuMatrix res;
	int size = cumat.sizes();

	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, cumat.rows(), cumat.cols());
	} else {
		res = cuMatrix(cumat.rows(), cumat.cols());
		mem.set(res.data);
	}	

	int threadnum = MAX_THREADNUM > cumat.cols() ? cumat.cols() : MAX_THREADNUM;
	dReLU_kernel<<<dim3(cumat.rows()), dim3(threadnum)>>>(cumat.getDev(), res.getDev(), cumat.cols());
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
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		mem.set(res.data);
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
		int cols) {
	int tidx = blockIdx.x;
	int tidy = threadIdx.x;
	float sum = 0;
	while (tidy < cols) {
		for (int i = 0; i < rows; i++) {
			sum += dev_x[i * cols + tidy];
		}
		dev_y[tidx * cols + tidy] = sum;
		tidy += blockDim.x;
	}
}

cuMatrix reduceSum(cuMatrix& src) {
	cuMatrix res;
	int size = src.sizes();
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		mem.set(res.data);
	}	
	
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	reduce_sum_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.rows(), src.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("reduce_sum");
	return res;
}

__global__ void log_kernel(float* dev_x, float* dev_y, int cols) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols) {
		dev_y[bid * cols + tid] = log(dev_x[bid * cols + tid]);
		tid += blockDim.x;
	}
}

cuMatrix Log(cuMatrix& src) {
	cuMatrix res;
	int size = src.sizes();
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		mem.set(res.data);
	}	
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	log_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementLog");
	return res;
}

cuMatrix4d Log(cuMatrix4d& src) {
	cuMatrix4d res;
	int size = src.sizes();
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix4d(tmpPtr, src.rows(), src.cols(), src.channals(), src.ts());
	} else {
		res = cuMatrix4d(src.rows(), src.cols(), src.channals(), src.ts());
		mem.set(res.data);
	}	
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	log_kernel<<<dim3(src.rows()*src.channals()*src.ts()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("4dElementLog");
	return res;
}
__global__ void exp_kernel(float* dev_x, float* dev_y, int cols) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	while (tid < cols) {
		dev_y[bid * cols + tid] = exp(dev_x[bid * cols + tid]);
		tid += blockDim.x;
	}
}

cuMatrix Exp(cuMatrix& src) {
	//	cuMatrix res(src.rows(), src.cols());
	cuMatrix res;
	int size = src.sizes();
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, src.rows(), src.cols());
	} else {
		res = cuMatrix(src.rows(), src.cols());
		mem.set(res.data);
	}	
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	exp_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementExp");
	return res;
}

void Exp(cuMatrix& src, cuMatrix& res) {
	assert(src.sizes() == res.sizes());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	exp_kernel<<<dim3(src.rows()), dim3(threadnum)>>>(src.getDev(),
			res.getDev(), src.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementExp");
}

__global__ void Pow_kernel(float* dev_x, float* dev_y, float* dev_z, int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], dev_y[x * cols + y]);
		y += blockDim.x;
	}
}

__global__ void Pow_kernel(float* dev_x, float y_, float* dev_z, int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = pow(dev_x[x * cols + y], y_);
		y += blockDim.x;
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
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, x.rows(), x.cols());
	} else {
		res = cuMatrix(x.rows(), x.cols());
		mem.set(res.data);
	}	
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	Pow_kernel<<<dim3(x.rows()), dim3(threadnum)>>>(x.getDev(), y.getDev(),
			res.getDev(), x.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix Pow(cuMatrix x,cuMatrix y)");
	return res;
}

cuMatrix Pow(cuMatrix x, float y) {
	int size = x.sizes();
	cuMatrix res;
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix(tmpPtr, x.rows(), x.cols());
	} else {
		res = cuMatrix(x.rows(), x.cols());
		mem.set(res.data);
	}	
	int threadnum = MAX_THREADNUM > x.cols() ? x.cols() : MAX_THREADNUM;
	Pow_kernel<<<dim3(x.rows()), dim3(threadnum)>>>(x.getDev(), y, res.getDev(),
			x.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("ElementPow matrix float matrix ");
	return res;
}

void cuMultiplication(cuMatrix src1, cuMatrix src2, cuMatrix dst) {
	assert(src1.cols() == src2.rows());
	assert(src1.rows() == dst.rows() && src2.cols() == dst.cols());
	float alpha = 1.0;
	float beta = 0.0;
	cublasStatus_t stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, src2.cols(),
			src1.rows(), src2.rows(), &alpha, src2.getDev(), src2.cols(),
			src1.getDev(), src1.cols(), &beta, dst.getDev(), dst.cols());
	cudaStreamSynchronize(0);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("cuMatrix::Mul() error\n");
		exit(0);
	}
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("init: CUBLAS destory failed\n");
		exit(0);
	}
}

__global__ void cuPlus_kernel(float* dev_x, float* dev_y, float* dev_z,	int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] + dev_y[x * cols + y];
		y += blockDim.x;
	}
}

//dst = src1 + src2
void cuPlus(cuMatrix src1, cuMatrix src2, cuMatrix dst) {
	assert(src1.sizes() == src2.sizes() && dst.sizes() == src1.sizes());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	cuPlus_kernel<<<dim3(src1.rows()), dim3(threadnum)>>>(src1.getDev(),
			src2.data->getDev(), dst.data->getDev(), src1.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
}

__global__ void cuDec_kernel(float* dev_x, float* dev_y, float* dev_z, int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		dev_z[x * cols + y] = dev_x[x * cols + y] - dev_y[x * cols + y];
		y += blockDim.x;
	}
}

//dst = src1 - src2
void cuDec(cuMatrix src1, cuMatrix src2, cuMatrix dst) {
	assert(src1.sizes() == src2.sizes() && dst.sizes() == src1.sizes());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	cuDec_kernel<<<dim3(src1.rows()), dim3(threadnum)>>>(src1.getDev(),
			src2.data->getDev(), dst.data->getDev(), src1.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix + cuMatrix");
}

void cuDec(cuMatrix4d src1,cuMatrix4d src2, cuMatrix4d dst){
	assert(src1.sizes() == src2.sizes() && dst.sizes() == src1.sizes());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	cuDec_kernel<<<dim3(src1.rows()*src1.channals()*src1.ts()), dim3(threadnum)>>>(src1.getDev(),
			src2.data->getDev(), dst.data->getDev(), src1.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("void cuDec(cuMatrix4d src1,cuMatrix4d src2, cuMatrix4d dst)");
}
__global__ void cuDiv_kernel(float* dev_x, float* dev_y, float* dev_z, int cols ) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_y[x * cols + y] != 0) {
			dev_z[x * cols + y] = dev_x[x * cols + y] / dev_y[x * cols + y];
		} else {
			dev_z[x * cols + y] = 0;
		}
		y += blockDim.x;
	}
}
void cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst) {
	assert(src1.sizes() == src2.sizes());
	assert(src1.sizes() == dst.sizes());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	cuDiv_kernel<<<dim3(src1.rows()), dim3(threadnum)>>>(src1.getDev(),
			src2.getDev(), dst.getDev(), src1.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst)");
}

__global__ void cuDiv_kernel(float x_, float* dev_y, float* dev_z, int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev_y[x * cols + y] != 0) {
			dev_z[x * cols + y] = x_ / dev_y[x * cols + y];
		} else {
			dev_z[x * cols + y] = 0;
		}
		y += blockDim.x;
	}
}
void cuDiv(float src1, cuMatrix src2, cuMatrix dst) {
	assert(src2.sizes() == dst.sizes());
	int threadnum = MAX_THREADNUM > src2.cols() ? src2.cols() : MAX_THREADNUM;
	cuDiv_kernel<<<dim3(src2.rows()), dim3(threadnum)>>>(src1, src2.getDev(),
			dst.getDev(), src2.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuDiv(cuMatrix src1, cuMatrix src2, cuMatrix dst)");
}
curandGenerator_t getGen() {
	static curandGenerator_t gen = NULL;
	if (gen == NULL) {
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	}
	return gen;
}

__global__ void creatBnl_kernel(float* dev, float threshold,int cols) {

	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		if (dev[x * cols + y] < threshold) {
			dev[x * cols + y] = 1.0f;
		} else {
			dev[x * cols + y] = 0.0f;
		}
		y += blockDim.x;
	}
}
void creatBnl(cuMatrix4d& bnl, float threshold) {
	curandGenerator_t gen = getGen();
	srand((unsigned) time(NULL));
	long seed = rand();
	curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandGenerateUniform(gen, bnl.getDev(), bnl.rows() * bnl.cols() * bnl.channals() * bnl.ts());
	int threadnum = MAX_THREADNUM > bnl.cols() ? bnl.cols() : MAX_THREADNUM;
	creatBnl_kernel<<<dim3(bnl.rows() * bnl.ts() * bnl.channals()),dim3(threadnum)>>>(bnl.getDev(),
			threshold, bnl.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("creatBnl(cuMatrix& bnl, float threshold)");
}

__global__ void addKernel4(float* src1,float* src2, float* dst,int col){
	int tid = threadIdx.x;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int bidz = blockIdx.z;
	while (tid < col) {
		dst[tid + bidx*col + bidy*col*gridDim.x + bidz*col*gridDim.x*gridDim.y] = 
			src1[tid +bidx*col + bidy*col*gridDim.x + bidz*col*gridDim.x*gridDim.y] + 
			src2[tid +bidx*col + bidy*col*gridDim.x + bidz*col*gridDim.x*gridDim.y];
		tid += blockDim.x;
	}
}

void cuMatrix4d_Add(cuMatrix4d& src1,cuMatrix4d& src2, cuMatrix4d& dst)
{
	assert(src1.len() == src2.len() && src1.len() == dst.len());
	assert(src1.ts() == src2.ts() && src1.ts() == dst.ts());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	addKernel4<<<dim3(src1.rows(),src1.channals(),src1.ts()),dim3(threadnum)>>>(src1.getDev(),src2.getDev(),dst.getDev(),src1.cols());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d_add");
}

__global__ void eleMulKernel4(float* src1,float* src2, float* dst,int col){
	int tid = threadIdx.x;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int bidz = blockIdx.z;
	while (tid < col) {
		dst[tid + bidx*col + bidy*col*gridDim.x + bidz*col*gridDim.x*gridDim.y] = 
			src1[tid +bidx*col + bidy*col*gridDim.x + bidz*col*gridDim.x*gridDim.y] * 
			src2[tid +bidx*col + bidy*col*gridDim.x + bidz*col*gridDim.x*gridDim.y];
		tid += blockDim.x;
	}
}

void cuMatrix4d_eleMul(cuMatrix4d& src1,cuMatrix4d& src2, cuMatrix4d& dst)
{
	assert(src1.len() == src2.len() && src1.len() == dst.len());
	assert(src1.ts() == src2.ts() && src1.ts() == dst.ts());
	int threadnum = MAX_THREADNUM > src1.cols() ? src1.cols() : MAX_THREADNUM;
	eleMulKernel4<<<dim3(src1.rows(),src1.channals(),src1.ts()),dim3(threadnum)>>>(src1.getDev(),src2.getDev(),dst.getDev(),src1.cols());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d_eleMul");
}

void cuMatrix4d_matMul(cuMatrix4d& src1,cuMatrix4d src2, cuMatrix4d& dst)
{
	assert(src1.cols() == src2.rows());
	assert(src1.rows() == dst.rows());
	assert(src2.cols() == dst.cols());
	assert(src1.ts() == src2.ts() && src1.ts() == dst.ts());
	float alpha = 1.0;
	float beta = 0.0;
	unsigned size = dst.sizes() * dst.ts() * dst.channals();	
	if(Devices::instance()->availableMemory < size * 1.3){
		for(int i = 0 ; i < src1.ts() ; i ++){
			for(int j = 0 ; j < src1.channals() ; j ++){
				cublasStatus_t stat;
				float* s1 = src1.data->getDev() + i*src1.area3D() + j*src1.area2D();
				float* s2 = src2.data->getDev() + i*src2.area3D() + j*src2.area2D();
				float* d =  dst.data->getDev() + i*dst.area3D() + j*dst.area2D();
				stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, src2.cols(),
						src1.rows(), src2.rows(), &alpha, s2, src2.cols(),
						s1, src1.cols(), &beta, 
						d, dst.cols());
				if (stat != CUBLAS_STATUS_SUCCESS) {
					printf("cuMatrix4d_matMul(cuMatrix4d& src1, cuMatrix4d& src2, cuMatrix4d& dst) error\n");
					exit(0);
				}
			}
		}
		getLastCudaError("cuMatrix4d_matMul");
	}
	else{
		cuMatrix tmpRes;
		tmpMemory mem(size);
		shared_ptr<MatData> tmpPtr = mem.getMem();
		if (tmpPtr != NULL) {
			tmpRes = cuMatrix(tmpPtr, dst.channals() * dst.ts() * dst.rows(), dst.channals() * dst.ts() * dst.cols() );
		} else {
			tmpRes = cuMatrix(dst.channals() * dst.ts() * dst.rows(), dst.channals() * dst.ts() * dst.cols());
			mem.set(tmpRes.data);
		}	
		cuMatrix tmpSrc2;	
		tmpMemory mem2(size);
		shared_ptr<MatData> tmpPtr2 = mem2.getMem();
		if (tmpPtr2 != NULL) {
			tmpSrc2 = cuMatrix(tmpPtr2,src2.rows(),src2.channals() * src2.ts() * src2.cols());
		} else {
			tmpSrc2 = cuMatrix(src2.rows(),src2.channals() * src2.ts() * src2.cols());
			mem2.set(tmpSrc2.data);
		}	
		cuMatrix4dRightTrans(src2,tmpSrc2);

		cublasStatus_t stat;
		float* s1 = src1.getDev();
		float* s2 = tmpSrc2.getDev();
		float* d =  tmpRes.getDev();
		stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, tmpSrc2.cols(),
				src1.rows() * src1.channals() * src1.ts(), tmpSrc2.rows(), &alpha, s2, tmpSrc2.cols(),
				s1, src1.cols(), &beta, 
				d, tmpRes.cols());
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf("cuMatrix::Mul() error\n");
			exit(0);
		}
		extractMatrix(tmpRes,dst);	
	}	
}


__global__ void extMatrixKernel(float* src, float* dst, int area2D, int col){
	int x = blockIdx.x;
	int y = threadIdx.x;
	int z = blockIdx.y;
	int tmp1 = area2D * gridDim.y;
	int tmp2 = col * gridDim.y;
	while(y < col){
		dst[area2D*z + x*col + y] = src[(tmp1+col)*z + x*tmp2 + y];	
		y += blockDim.x;
	}
}

void extractMatrix(cuMatrix& src,cuMatrix4d& dst){
	assert(src.rows() == dst.rows()*dst.channals()*dst.ts() && src.cols() == dst.cols() * dst.channals() * dst.ts());
	int threadnum = MAX_THREADNUM > dst.cols() ? dst.cols() : MAX_THREADNUM;
	extMatrixKernel<<<dim3(dst.rows(),dst.channals()*dst.ts()),dim3(threadnum)>>>(src.getDev(), dst.getDev(), dst.area2D(), dst.cols());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("extractMatrix");
}

void cuMatrix4d_matMul(cuMatrix src1, cuMatrix4d& src2, cuMatrix4d& dst){
	assert(src1.cols() == src2.rows());
	assert(src1.rows() == dst.rows() && dst.cols() ==src2.cols());	
	assert(src2.ts() == dst.ts() && src2.channals() == dst.channals());
	unsigned size = dst.sizes() ;	
	float alpha = 1.0;
	float beta = 0.0;
	if(Devices::instance()->availableMemory < size * 1.3){
		for(int i = 0 ; i < src2.ts() ; i ++){
			for(int j = 0 ; j < src2.channals() ; j ++){
				cublasStatus_t stat;
				float* s1 = src1.data->getDev();
				float* s2 = src2.data->getDev() + i*src2.area3D() + j*src2.area2D();
				float* d =  dst.data->getDev() + i*dst.area3D() + j*dst.area2D();
				stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, src2.cols(),
						src1.rows(), src2.rows(), &alpha, s2, src2.cols(),
						s1, src1.cols(), &beta, 
						d, dst.cols());
				if (stat != CUBLAS_STATUS_SUCCESS) {
					printf("cuMatrix4d_matMul(cuMatrix& src1, cuMatrix4d& src2, cuMatrix4d& dst) error\n");
					exit(0);
				}
			}
		}
		getLastCudaError("cuMatrix4d_matMul(cuMatrix& src1, cuMatrix4d& src2, cuMatrix4d& dst)");
	}else{
	
		cuMatrix tmpRes;
		tmpMemory mem(size);
		shared_ptr<MatData> tmpPtr = mem.getMem();
		if (tmpPtr != NULL) {
			tmpRes = cuMatrix(tmpPtr,  dst.rows(),dst.channals() * dst.ts() * dst.cols() );
		} else {
			tmpRes = cuMatrix( dst.rows(),dst.channals() * dst.ts() * dst.cols());
			mem.set(tmpRes.data);
		}	
		cuMatrix tmpSrc2;	
		tmpMemory mem2(src2.sizes());
		shared_ptr<MatData> tmpPtr2 = mem2.getMem();
		if (tmpPtr2 != NULL) {
			tmpSrc2 = cuMatrix(tmpPtr2,src2.rows(),src2.channals() * src2.ts() * src2.cols());
		} else {
			tmpSrc2 = cuMatrix(src2.rows(),src2.channals() * src2.ts() * src2.cols());
			mem2.set(tmpSrc2.data);
		}	
		cuMatrix4dRightTrans(src2,tmpSrc2);

		cublasStatus_t stat;
		float* s1 = src1.getDev();
		float* s2 = tmpSrc2.getDev();
		float* d =  tmpRes.getDev();
		stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, tmpSrc2.cols(),
				src1.rows() , tmpSrc2.rows(), &alpha, s2, tmpSrc2.cols(),
				s1, src1.cols(), &beta, 
				d, tmpRes.cols());
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf("cuMatrix::Mul() error\n");
			exit(0);
		}
		cuMatrix4dRightInverseTrans(tmpRes,dst);
	}

}

//blockIdx.x .y .z = src.rows(),src.channals(), src.ts()
//threadIdx.x = src.cols().
__global__ void RTkernel(float *src, float *dst, int a3, int a2 , int col){
	int x = blockIdx.x;
	int y = threadIdx.x;
	int ch = blockIdx.y;
	int ts = blockIdx.z;
	while(y < col){
		int offset = ts*a3 + ch*a2 + y * gridDim.x + x;
		int i = offset / gridDim.x;
		int j = offset % gridDim.x;	
		dst[j * gridDim.z * gridDim.y * col + i] = src[ts*a3 + ch*a2 + x*col + y];
		y += blockDim.x;
	}
}

void cuMatrix4dRightTrans(cuMatrix4d& src,cuMatrix& dst){
	assert(src.sizes() == dst.sizes());
	int threadnum = MAX_THREADNUM > src.cols() ? src.cols() : MAX_THREADNUM;
	RTkernel<<<dim3(src.rows(),src.channals(),src.ts()),dim3(threadnum)>>>(src.getDev(), dst.getDev(), src.area3D(), src.area2D(), src.cols());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d_RT");
}

__global__ void RTinverseKernel(float* src, float* dst,int a2, int col){
	int x = blockIdx.x;
	int y = threadIdx.x;
	int z = blockIdx.y;
	while(y < col){
		int offset = (z*col + y)*gridDim.x + x;
		int k = offset/a2;
		int tmp = offset%a2;
		int i = tmp/gridDim.x;
		int j = tmp%gridDim.x;
		dst[k*a2 + j * col + i] = src[x * col * gridDim.y + z * col + y];
		y+=blockDim.x;
	}
}

void cuMatrix4dRightInverseTrans(cuMatrix&src,cuMatrix4d& dst){
	assert(src.rows() == dst.rows() && src.cols() == dst.cols()*dst.channals()*dst.ts());	
	int threadnum = MAX_THREADNUM > dst.cols() ? dst.cols() : MAX_THREADNUM;
	RTinverseKernel<<<dim3(dst.rows(),dst.channals()*dst.ts()),dim3(threadnum)>>>(src.getDev(),dst.getDev(),dst.area2D(),dst.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d_RTinverse");
}


__global__ void squareKernel(float* src, float* dst, int cols, int a2){
	int x = blockIdx.x;
	int y = threadIdx.x;
	int z = blockIdx.y;
	while(y < cols){
		dst[a2 * z +  x * cols + y] = pow(src[a2 * z + x * cols + y], 2.0f);
		y += blockDim.x;
	}
}

void square(cuMatrix4d &src,cuMatrix4d &dst){
	assert(src.len() == dst.len());
	int threadnum = MAX_THREADNUM > dst.cols() ? dst.cols() : MAX_THREADNUM;
	squareKernel<<<dim3(src.rows(),src.channals()*src.ts()),dim3(threadnum)>>>(src.getDev(),dst.getDev(),src.cols(),src.area2D());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("square");
}

