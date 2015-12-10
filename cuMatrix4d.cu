#include "cuMatrix4d.h"
const int maxThreadNum = Devices::instance()->maxThreadNum();
const int* blockdim = Devices::instance()->blockDim();
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
	int threadnum = maxThreadNum > src1.cols() ? src1.cols() : maxThreadNum;
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
	int threadnum = maxThreadNum > src1.cols() ? src1.cols() : maxThreadNum;
	eleMulKernel4<<<dim3(src1.rows(),src1.channals(),src1.ts()),dim3(threadnum)>>>(src1.getDev(),src2.getDev(),dst.getDev(),src1.cols());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d_eleMul");
}

void cuMatrix4d_matMul(cuMatrix4d& src1,cuMatrix4d& src2, cuMatrix4d& dst)
{
	assert(src1.cols() == src2.rows());
	assert(src1.rows() == dst.rows());
	assert(src2.cols() == dst.cols());
	assert(src1.ts() == src2.ts() && src1.ts() == dst.ts());
	float alpha = 1.0;
	float beta = 0.0;
	for(int i = 0 ; i < src1.ts() ; i ++){
		for(int j = 0 ; j < src1.channals() ; j ++){
			cublasStatus_t stat;
//			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, src2.cols(),
//					src1.rows(), src2.rows(), &alpha, src2.data->getDev()[i*src2.area3D() + j*src2.area2D()], src2.cols(),
//					src1.data->getDev()[i*src1.area3D() + j*src1.area2D()], src1.cols(), &beta, 
//					dst.data->getDev()[i*dst.area3D() + j*dst.area2D()], dst.cols());
			float* s1 = src1.data->getDev() + i*src1.area3D() + j*src1.area2D();
			float* s2 = src2.data->getDev() + i*src2.area3D() + j*src2.area2D();
			float* d =  dst.data->getDev() + i*dst.area3D() + j*dst.area2D();
			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, src2.cols(),
					src1.rows(), src2.rows(), &alpha, s2, src2.cols(),
					s1, src1.cols(), &beta, 
					d, dst.cols());
			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf("cuMatrix::Mul() error\n");
				exit(0);
			}
		}
	}
	getLastCudaError("cuMatrix4d_matMul");
}


