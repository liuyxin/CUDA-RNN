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
	unsigned size = dst.sizes() * dst.ts() * dst.ts();	
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
					printf("cuMatrix::Mul() error\n");
					exit(0);
				}
			}
		}
		getLastCudaError("cuMatrix4d_matMul");
	}
	else{
		cuMatrix tmpRes;	
		if (cuMatrix::tmpMemory.find(size) != cuMatrix::tmpMemory.end()) {
			tmpRes = cuMatrix(cuMatrix::tmpMemory[size], dst.channals() * dst.ts() * dst.rows(),dst.channals() * dst.ts() * dst.cols());
		} else{ 
			tmpRes = cuMatrix(dst.channals() * dst.ts() * dst.rows(),dst.channals() * dst.ts() * dst.cols());
			cuMatrix::tmpMemory[size] = tmpRes.data;
		}
		cuMatrix tmpSrc2;	
		if (cuMatrix::tmpMemory.find(src2.sizes()) != cuMatrix::tmpMemory.end()) {
			tmpSrc2 = cuMatrix(cuMatrix::tmpMemory[size], src2.rows(),src2.channals() * src2.ts() * src2.cols());
		} else{ 
			tmpSrc2 = cuMatrix(src2.rows(),src2.channals() * src2.ts() * src2.cols());
			cuMatrix::tmpMemory[src2.sizes()] = tmpSrc2.data;
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
	int threadnum = maxThreadNum > src.cols() ? src.cols() : maxThreadNum;
	RTkernel<<<dim3(src.rows(),src.channals(),src.ts()),dim3(threadnum)>>>(src.getDev(), dst.getDev(), src.area3D(), src.area2D(), src.cols());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d_RT");
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
	int threadnum = maxThreadNum > dst.cols() ? dst.cols() : maxThreadNum;
	extMatrixKernel<<<dim3(dst.rows(),dst.channals()*dst.ts()),dim3(threadnum)>>>(src.getDev(), dst.getDev(), dst.area2D(), dst.cols());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("extractMatrix");
}

