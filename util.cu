#include "util.h"
const int MAX_THREADNUM = Devices::instance()->maxThreadNum();
__device__ float max_(float a, float b){
	return a>b?a:b;
}
__global__ void n2aKernel(float* non,float* act, float* bnl,int col){	
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < col) {
		if (src[x * col + y] <= 0) {
			dst[x * col + y] = 0;
		} else {
			dst[x * col + y] = src[x * col + y] * bnl[x * col + y];
		}
		y += blockDim.x;
	}
}

void non2acti(cuMatrix4d& non, cuMatrix4d& acti,cuMatrix4d& bnl,int t){
	assert(non.len() == acti.len() && non.len() == bnl.len());
	int threadnum = MAX_THREADNUM > non.cols() ? non.cols() : MAX_THREADNUM;
	n2aKernel<<<dim3(non.rows()), dim3(threadnum)>>>(non.getDev()+t*area2D,
			acti.getDev()+t*area2D, bnl.getDev()+t*area2D, non.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("non2acti");
}

__global__ void anaKernel(float* a, float* n,float* p, float* bnl,int a2,int col){
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < col) {
		n[x*col + y] += p[x*col+y];
		if (n[x * col + y] <= 0) {
			a[x * col + y] = 0;
		} else {
			a[x * col + y] = n[x * col + y] * bnl[x * col + y];
		}
		y += blockDim.x;
	}
}
// acti[t-1] -> nonlin[t] -> acti[t] or acti[t+1]->nonlin[t]->acti[t]
void acti2non2acti(cuMatrix4d& acti, cuMatrix4d& non,cuMatrix4d& bnl ,cuMatrix& w,int t,bool f){
	cuMatrix tmpRes;
	int tmpSize = w.rows() * acti.cols() * sizeof(float);
	if (cuMatrix::tmpMemory.find(tmpSize) != tmpMemory.end()) {
		tmpRes = cuMatrix(cuMatrix::tmpMemory[tmpSize], w.rows(), acti.cols());
	} else {
		tmpRes = cuMatrix(w.rows(), acti.cols());
		cuMatrix::tmpMemory[tmpSize] = tmpRes.data;
	}
	float alpha = 1.0;
	float beta = 0.0;
	cublasStatus_t stat;
	if(f == TIMEFORWARD){
		stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti.cols(),
				w.rows(), acti.rows(), &alpha, acti.getDev()+(t-1)*acti.area2D(), acti.cols(),
				w.getDev(), w.cols(), &beta, tmpRes.getDev(), tmpRes.cols());
	}else {
		stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti.cols(),
				w.rows(), acti.rows(), &alpha, acti.getDev()+(t+1)*acti.area2D(), acti.cols(),
				w.getDev(), w.cols(), &beta, tmpRes.getDev(), tmpRes.cols());
	}
	cudaStreamSynchronize(0);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("acti2non2acti(cuMatrix4d& acti, cuMatrix4d& non,cuMatrix4d& bnl ,cuMatrix& w,int t) error\n");
		exit(0);
	}
	int threadnum = MAX_THREADNUM > non.cols() ? non.cols() : MAX_THREADNUM;
	anaKernel<<<dim3(non.rows()),dim3(threadnum)>>>(acti.getDev()+t*acti.area2D(),non.getDev()+t*non.area2D(),tmpRes.getDev(),bnl.getDev()+t*bnl.area2D(),bnl.area2D(),bnl.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("acti2non2acti");
}


void hiddenForward(cuMatrix4d& nonlin, cuMatrix4d& acti,cuMatrix& weight,cuMatrix4d& bnl, bool f){
	if(f == TIMEFORWARD){
		non2acti(nonlin,acti,bnl,0);			
		for(int t = 1 ; t < nonlin.ts() ; t++){
			acti2non2acti(acti,non,bnl,weight,t,TIMEFORWARD);
		}
	}
	else{
		non2acti(nonlin,acti,bnl,Config::instance()->get_ngram()-1);
		for(int t = Config::instance()->get_ngram()-2 ; t >= 0 ; t--){
			acti2non2acti(acti,non,bnl,weight,t,TIMEBACKWARD);
		}
	}
}

//block.x: p.cols(); block.y:p.channals()*p.ts()
//thread:p.rows()
__global__ void smrKernel(float* p,int a2){
	__shared__ float cache[];
	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.z;
	float *max_ = cache;
	float *sum_ = cache + blockDim.x;
	max_[x] = p[a2*z + x*gridDim.x + y];	
	__syncthreads();
	int len = blockDim.x;
	while(len != 1){
        	int skip = (len + 1) >> 1;
		if(x < (len >> 1)){
			max_[x] = max_(cache[x],cache[x+skip]);			
		}
		len = (len + 1) >> 1;	
		__syncthreads();
	}
	p[a2*z + x*gridDim.x + y] -= max_[0];
	p[a2*z + x*gridDim.x + y] = exp(p[a2*z + x*gridDim.x + y]);	
	sum_[x] = p[a2*z + x*gridDim.x + y];
	__syncthreads();
	len = blockDim.x;
	while(len != 1){
		int skip = (len + 1) >> 1;
		if(x < (len >> 1)){
			sum_[x] += sum_[x + skip];
		}
		len = (len + 1) >> 1;
		__syncthreads();
	}
	p[a2*z + x*gridDim.x + y] /= sum_[0];
}

void smrForward(cuMatrix& wr,cuMatrix4d& ar,cuMatrix& wl, cuMatrix4& al,cuMatrix4d &p){
	cuMatrix4d& tmpp;
	if (cuMatrix::tmpMemory.find(p.sizes()) != cuMatrix::tmpMemory.end()) {
		tmpp = cuMatrix4d(cuMatrix::tmpMemory[p.sizes()], p.rows(), p.cols(), p.channals(), p.ts());
	} else {
		tmpp = cuMatrix4d(p.rows(), p.cols(), p.channals(), p.ts());
		cuMatrix::tmpMemory[p.sizes()] = tmpp.data;
	}
	cuMatrix4d_matMul(wl,al,p);
	cuMatrix4d_matMul(wr,ar,tmpp);
	cuMatrix4d_Add(p,tmpp,p);
	assert(2 * p.rows() *sizeof(float) <= Devices::instance()->get_sharedmemorysize());	
	smrKernel<<<dim3(p.cols,p.channals() * p.ts(), dim3(p.rows()), 2 * p.rows() * sizeof(float))>>>(p.getDev(), p.area2D());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrForward");
}
__global__ void getSmrgrad(float* src,  float* dst, int cols, int ts, int a2){
	int x = blockIdx.x;
	int y = threadIdx.x;
	float* s = src;
	float* d = dst;
	while(y < cols){
		d[x * cols + y] = 0.0f - s[x * cols() + y];
		d = d + a2;
		s = s + a2;
		for(int i = 1 ; i < ts ; i ++){
			d[x * cols + y] -= s[x * cols() + y];
			d = d + a2;
			s = s + a2;		
		}
	}
}
__global__ void getSmrd2(float* src,  float* dst, int cols, int ts,int a2) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	float* s = src;
	float* d = dst;
	while(y < cols){
		d[x * cols + y] = s[x * cols() + y];
		d = d + a2;
		s = s + a2;		
		for(int i = 1 ; i < ts ; i ++){
			d[x * cols + y] += s[x * cols() + y];
			d = d + a2;
			s = s + a2;		
		}
	}
}
void smrBP(SoftMax& smr, cuMatrix4d& acti_l,cuMatrix4d& acti_r,cuMatrix4d& acti_l2,cuMatrix4d& acti_r2,cuMatrix4d& dis, cuMatrix4d& dis2, int nSamples){
	cuMatrix4d tmpRes;
	if (cuMatrix::tmpMemory.find(p.sizes()) != cuMatrix::tmpMemory.end()) {
		tmpRes = cuMatrix4d(cuMatrix::tmpMemory[p.sizes()], p.rows(), p.cols(), p.channals(), p.ts());
	} else {
		tmppRes = cuMatrix4d(p.rows(), p.cols(), p.channals(), p.ts());
		cuMatrix::tmpMemory[p.sizes()] = tmpp.data;
	}
	int threadnum = MAX_THREADNUM > tmpRes.cols() ? tmpRes.cols() : MAX_THREADNUM;
	cuMatrix4d_matMul(dis, acti_l, tmpRes);
	getSmrgrad<<<dim3(tmpRes.rows()),dim3(threadnum)>>>(tmpRes.getDev(),smr.W_lgrad.getDev(),tmpRes.cols(),tmpRes.ts(),tmpRes.area2D());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrBP: getSmrlgrad");

	cuMatrix4d_matMul(dis, acti_r, tmpRes);
	getSmrgrad<<<dim3(tmpRes.rows()),dim3(threadnum)>>>(tmpRes.getDev(),smr.W_rgrad.getDev(),tmpRes.cols(),tmpRes.ts(),tmpRes.area2D());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrBP: getSmrrgrad");

	cuMatrix4d_matMul(dis2, acti_l2, tmpRes);
	getSmrgetd2<<<dim3(tmpRes.rows()),dim3(threadnum)>>>(tmpRes.getDev(),smr.W_ld2.getDev(),tmpRes.cols(),tmpRes.ts(),tmpRes.area2D());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrBP: getSmrld2");

	cuMatrix4d_matMul(dis2, acti_r2, tmpRes);
	getSmrgetd2<<<dim3(tmpRes.rows()),dim3(threadnum)>>>(tmpRes.getDev(),smr.W_rd2.getDev(),tmpRes.cols(),tmpRes.ts(),tmpRes.area2D());	
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrBP: getSmrrd2");
}

//dReLU & bnl
__global__ void bpttKernel(float* p1 , float* p2 , float* non, float* bnl ,int cols){
	int x = blockIdx.x;
	int y = threadIdx.x;
	while(y < cols){
		if(bnl[x * cols + y] == 0 || non[x * cols + y] <= 0){
			p1[x * cols + y] = 0.0f;
		} else{
			p1[x * cols + y] += p2[x * cols + y];
		}	
		y += blockDim.x;
	}
}
__global__ void bpttInit(float* p1, float* non, float* bnl ,int cols){
	int x = blockIdx.x;
	int y = threadIdx.x;
	while(y < cols){
		if(bnl[x * cols + y] == 0 || non[x * cols + y] <= 0){
			p1[x * cols + y] = 0.0f;
		}
		y += blockDim.x;
	}
}
//w is hidden.W_l.t()
void hiddenBPTT(cuMatrix4d& delta, cuMatrix w, cuMatrix4d& non, cuMatrix4d& bnl, bool f){
	cuMatrix tmpRes;
	int tmpSize = w.rows() * delta.cols() * sizeof(float);
	int ts = delta.ts();
	if (cuMatrix::tmpMemory.find(tmpSize) != cuMatrix::tmpMemory.end()) {
		tmpRes = cuMatrix(cuMatrix::tmpMemory[tmpSize], w.rows(), delta.cols());
	} else {
		tmpRes = cuMatrix(w.rows(), delta.cols());
		cuMatrix::tmpMemory[tmpSize] = tmpRes.data;
	}
	float alpha = 1.0;
	float beta = 0.0;
	cublasStatus_t stat;
	int threadnum = MAX_THREADNUM > delta.cols() ? delta.cols() : MAX_THREADNUM;
	if(f == TIMEFORWARD){
		int off = (delta.ts() - 1) * delta.area2D();
		bpttInit<<<dim3(delta.rows()),dim3(threadnum)>>>(delta.getDev() + off, non.getDev() + off, bnl.getDev() + off, delta.cols());
		for(int t = ts - 2 ; t >= 0 ; t -- ){
			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, delta.cols(),
					w.rows(), delta.rows(), &alpha, delta.getDev()+(t+1)*delta.area2D(), delta.cols(),
					w.getDev(), w.cols(), &beta, tmpRes.getDev(), tmpRes.cols());
			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf("hiddenBPTT(cuMatrix4d& delta, cuMatrix& w, cuMatrix4d& non, cuMatrix4d& bnl, bool f)\n");
				exit(0);
			}
			off = t * delta.area2D();
			bpttKernel<<<dim3(delta.rows()),dim3(threadnum)>>>(delta.getDev() + off , tmpRes.getDev() , non.getDev() + off , bnl.getDev() + off, delta.cols());
			checkCudaErrors(cudaStreamSynchronize(0));
			getLastCudaError("bpttKernel timeforward");
		}
	}else{
		bpttInit<<<dim3(delta.rows()),dim3(threadnum)>>>(delta.getDev(), non.getDev(), bnl.getDev(), delta.cols());
		for(int t = 1 ; t < ts ; t ++){
			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, delta.cols(),
					w.rows(), delta.rows(), &alpha, delta.getDev()+(t-1)*acti.area2D(), delta.cols(),
					w.getDev(), w.cols(), &beta, tmpRes.getDev(), tmpRes.cols());
			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf("hiddenBPTT(cuMatrix4d& delta, cuMatrix& w, cuMatrix4d& non, cuMatrix4d& bnl, bool f)\n");
				exit(0);
			}
			int off = t * delta.area2D();
			bpttKernel<<<dim3(delta.rows()),dim3(threadnum)>>>(delta.getDev() + off , tmpRes.getDev() , non.getDev() + off , bnl.getDev() + off, delta.cols());
			checkCudaErrors(cudaStreamSynchronize(0));
			getLastCudaError("bpttKernel timebackward");
		}
	}
}
