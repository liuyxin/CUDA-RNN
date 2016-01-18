#include "cuMatrix4d.h"
static int MAX_THREADNUM = Devices::instance()->maxThreadNum();
static __device__ unsigned int __count = 0;
static __shared__ bool isLastBlockDone;
__global__ void getSumKernel(float* src, float* c, int col){
	extern __shared__ float sm[]; 
	int x = blockIdx.x;
	int y = threadIdx.x;
	int t = y;
	sm[y] = src[x * col  + y];
	t += blockDim.x;
//	__syncthreads();
	while(t < col){
		sm[y] += src[x*col  + t];
		t += blockDim.x;
	}
	__syncthreads();
	t = blockDim.x;
	while(t != 1){
		int skip = (t + 1) >> 1;
		if(y < (t >> 1)){
			sm[y] += sm[y + skip];		
		}
		t = (t+1)>>1;
		__syncthreads();
	}
	if(y == 0){
		c[x] = sm[0];
		__threadfence();
			unsigned int value = atomicInc(&__count , gridDim.x);//count > gridDim.x? 0 : count++;
			isLastBlockDone = (value == (gridDim.x-1));
	}
	__syncthreads();
	if(isLastBlockDone){
		int len = gridDim.x;
		t = y;
		sm[y] = c[t];
		t += blockDim.x;
		while(t < len){
			sm[y] += c[t];
			t += blockDim.x;
		}
		__syncthreads();
		t = blockDim.x;
		while(t != 1){
			int skip = (t + 1) >> 1;
			if(y < (t >> 1)){
				sm[y] += sm[y + skip];		
			}
			t = (t+1)>>1;
			__syncthreads();
		}
		if(y == 0){
			c[0] = sm[0];
		}
		__count = 0 ;
	}
	__syncthreads();
}

float& cuMatrix4d::getSum(){
	int tmpSize = rows() * channals() * ts() * sizeof(float); 	
	tmpMemory tmpm(tmpSize);
	shared_ptr<MatData> mem = tmpm.getMem();
	if(mem == NULL){
		mem = make_shared < MatData >(1,rows() * channals() * ts());
		tmpm.set(mem);
	}
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	int smlen = threadnum;
	getSumKernel<<<dim3(rows()*channals()*ts()),dim3(threadnum), smlen * sizeof(float)>>>(getDev(),mem->getDev(),cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d::getSum()");
	cudaMemcpyAsync(&sum,mem->getDev(),sizeof(float),cudaMemcpyDeviceToHost);
	return sum;
}

__global__ void mulKernel(float* dev_x, float* dev_y, float* dev_z, int a2, int cols) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	int z = blockIdx.y;
	while (y < cols) {
		dev_z[z*a2  + x * cols + y] = dev_x[z*a2  + x * cols + y] * dev_y[z*a2  + x * cols + y];
		y += blockDim.x;
	}
}

cuMatrix4d cuMatrix4d::Mul(cuMatrix4d m) {
	assert(m.sizes() == sizes());
	int threadnum = MAX_THREADNUM > cols() ? cols() : MAX_THREADNUM;
	cuMatrix4d res;
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix4d(tmpPtr, rows(), cols(), channals(), ts());
	} else {
		res = cuMatrix4d(rows(), cols(), channals(), ts());
		mem.set(res.data);
	}	

	mulKernel<<<dim3(rows(),ts()*channals()), dim3(threadnum)>>>(data->getDev(),
			m.data->getDev(), res.data->getDev(),m.area2D() ,cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("cuMatrix4d::Mul(cuMatrix4d)");
	return res;
}
__global__ void t_kernel(float* dev_src, float* dev_res, int res_r, int res_c, int a2){
	int x = blockIdx.x;
	int y = threadIdx.x;
	int z = blockIdx.y;
	while (y < res_c) {
		dev_res[z * a2 + x * res_c + y] = dev_src[z * a2 +y * res_r + x];
		y += blockDim.x;
	}
}

cuMatrix4d cuMatrix4d::t() {
	assert(cols() != 0 && rows() != 0);
	cuMatrix4d res;
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		res = cuMatrix4d(tmpPtr, cols(), rows(), channals(), ts());
	} else {
		res = cuMatrix4d(cols(), rows(), channals(), ts());
		mem.set(res.data);
	}	
	int threadnum = MAX_THREADNUM > res.cols() ? res.cols() : MAX_THREADNUM;
	t_kernel<<<dim3(res.rows(),res.channals()*res.ts()), dim3(threadnum)>>>(data->getDev(),
			res.data->getDev(), res.rows(), res.cols(), area2D());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("pre-element add cuMatrix / float");
	return res;
}
