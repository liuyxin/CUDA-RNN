#include "util.h"

const int MAX_THREADNUM = Devices::instance()->maxThreadNum();
__device__ float __max(float a, float b)
{
	return a>b?a:b;
}

__global__ void n2aKernel(float* n,float* a, float* bnl,int col){	
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < col) {
		if (n[x * col + y] <= 0 ) {
			a[x * col + y] = 0;
		} else {
			a[x * col + y] = n[x * col + y] * bnl[x * col + y];
		}
		y += blockDim.x;
	}
}

void non2acti(cuMatrix4d& non, cuMatrix4d& acti,cuMatrix4d& bnl,int t){
	assert(non.len() == acti.len() && non.len() == bnl.len());
	int threadnum = MAX_THREADNUM > non.cols() ? non.cols() : MAX_THREADNUM;
	int area2D = acti.area2D();
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
	tmpMemory mem(tmpSize);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		tmpRes = cuMatrix(tmpPtr, w.rows(), acti.cols());
	} else {
		tmpRes = cuMatrix(w.rows(), acti.cols());
		mem.set(tmpRes.data);
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
			acti2non2acti(acti,nonlin,bnl,weight,t,TIMEFORWARD);
		}
	}
	else{
		non2acti(nonlin,acti,bnl,Config::instance()->get_ngram()-1);
		for(int t = Config::instance()->get_ngram()-2 ; t >= 0 ; t--){
			acti2non2acti(acti,nonlin,bnl,weight,t,TIMEBACKWARD);
		}
	}
}

//block.x: p.cols(); block.y:p.channals()*p.ts()
//thread:p.rows()
__global__ void smrKernel(float* p,int a2){
	extern __shared__ float cache[];
	int x = threadIdx.x;
	int y = blockIdx.x;
	int z = blockIdx.y;
	float *max_ = cache;
	float *sum_ = cache + blockDim.x;
	max_[x] = p[a2*z + x*gridDim.x + y];	
	__syncthreads();
	int len = blockDim.x;
	while(len != 1){
		int skip = (len + 1) >> 1;
		if(x < (len >> 1)){
			max_[x] = __max(cache[x],cache[x+skip]);			
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

void smrForward(cuMatrix& wr,cuMatrix4d& ar,cuMatrix& wl, cuMatrix4d& al,cuMatrix4d &p){
	cuMatrix4d tmpp;
	tmpMemory mem(p.sizes());
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		tmpp = cuMatrix4d(tmpPtr, p.rows(), p.cols(), p.channals(), p.ts());
	} else {
		tmpp = cuMatrix4d(p.rows(), p.cols(), p.channals(), p.ts());
		mem.set(tmpp.data);
	}	
	cuMatrix4d_matMul(wl,al,p);
	cuMatrix4d_matMul(wr,ar,tmpp);
	cuMatrix4d_Add(p,tmpp,p);
	assert(2 * p.rows() *sizeof(float) <= Devices::instance()->get_sharedmemorysize());	
	smrKernel<<<dim3(p.cols(), p.channals()*p.ts()), dim3(p.rows()), 2*p.rows()*sizeof(float)>>>(p.getDev(), p.area2D());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrForward");
}

__global__ void getSmrgrad(float* src, float* dst, float* w,int cols, int ts, int a2 , int nSamples, float decay) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	float* s = src;
	float* d = dst;
	while (y < cols) {
		d[x * cols + y] = 0 + s[x * cols + y];

		for (int i = 1; i < ts; i++) {
			s = s + a2;
			d[x * cols + y] += s[x * cols + y];
		}
		d[x * cols + y] /= nSamples;
		d[x * cols + y] += decay * w[x * cols + y];
		y += blockDim.x;
	}
}
__global__ void getSmrd2(float* src, float* dst, int cols, int ts, int a2, int nSamples, float decay) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < cols) {
		float* s = src;
		float* d = dst;
		d[x * cols + y] = s[x * cols + y];

		for (int i = 1; i < ts; i++) {
			s = s + a2;
			d[x * cols + y] += s[x * cols + y];
		}
		d[x * cols + y] /= nSamples;
		d[x * cols + y] += decay;
		y += blockDim.x;
	}
}

void smrBP(SoftMax& smr, cuMatrix4d& acti_l, cuMatrix4d& acti_r,
		cuMatrix4d& acti_l2, cuMatrix4d& acti_r2, cuMatrix4d& dis,
		cuMatrix4d& dis2, int nSamples) {
	cuMatrix4d tmpRes;
	int tmpSize = smr.W_l.sizes() * dis.channals() * dis.ts();
	tmpMemory mem(tmpSize);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		tmpRes = cuMatrix4d(tmpPtr, smr.W_l.rows(), smr.W_l.cols(), dis.channals(), dis.ts());
	} else {
		tmpRes = cuMatrix4d(smr.W_l.rows(), smr.W_l.cols(), dis.channals(), dis.ts());
		mem.set(tmpRes.data);
	}	

	cuMatrix4d_matMul(dis, acti_l.t(), tmpRes);

	int threadnum =
		MAX_THREADNUM > tmpRes.cols() ? tmpRes.cols() : MAX_THREADNUM;
	getSmrgrad<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(),
			smr.W_lgrad.getDev(), smr.W_l.getDev(), tmpRes.cols(), tmpRes.ts(), tmpRes.area2D(),nSamples,smr.get_WeightDecay());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrBP: getSmrlgrad");

	cuMatrix4d_matMul(dis, acti_r.t(), tmpRes);
	getSmrgrad<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(),
			smr.W_rgrad.getDev(),smr.W_r.getDev(), tmpRes.cols(), tmpRes.ts(), tmpRes.area2D(),nSamples,smr.get_WeightDecay());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrBP: getSmrrgrad");

	cuMatrix4d_matMul(dis2, acti_l2.t(), tmpRes);
	getSmrd2<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(),
			smr.W_ld2.getDev(), tmpRes.cols(), tmpRes.ts(), tmpRes.area2D(),nSamples,smr.get_WeightDecay());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("smrBP: getSmrld2");

	cuMatrix4d_matMul(dis2, acti_r2.t(), tmpRes);
	getSmrd2<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(),
			smr.W_rd2.getDev(), tmpRes.cols(), tmpRes.ts(), tmpRes.area2D(),nSamples,smr.get_WeightDecay());
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
	tmpMemory mem(tmpSize);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		tmpRes = cuMatrix(tmpPtr, w.rows(), delta.cols());
	} else {
		tmpRes = cuMatrix(w.rows(), delta.cols());
		mem.set(tmpRes.data);
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
					w.rows(), delta.rows(), &alpha, delta.getDev()+(t-1)*delta.area2D(), delta.cols(),
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

__global__ void gradKernle(float* src, float* dst, float* u,float a, int nSamples , int a2, int ts, int cols){
	int x = blockIdx.x;
	int y = threadIdx.x;
	while(y < cols){
		int off = x * cols + y;
		dst[off] = src[off];
		for(int i =  1 ; i < ts ; i ++){
			dst[off] += src[i*a2 + off];
		}
		dst[off] /= nSamples;
		dst[off] += u[off]*a;
		y += blockDim.x;
	}
}

__global__ void gradKernle(float* src, float* dst, float a, int nSamples , int a2, int ts, int cols){
	int x = blockIdx.x;
	int y = threadIdx.x;
	while(y < cols){
		int off = x * cols + y;
		dst[off] = src[off];
		for(int i =  1 ; i < ts ; i ++){
			dst[off] += src[i*a2 + off];
		}
		dst[off] /= nSamples;
		dst[off] += a;
		y += blockDim.x;
	}
}


void hiddenGetUgrad(cuMatrix4d& delta_l, cuMatrix4d& delta_r, 
		cuMatrix4d& delta_ld2, cuMatrix4d& delta_rd2,
		cuMatrix4d& acti_sum, cuMatrix4d& acti2_sum, HiddenLayer& hidden ,float WeightDecay){
	cuMatrix4d tmpRes;
	int ts = delta_l.ts();
	int nSamples = delta_l.cols();
	//	float alpha = 1.0;
	//	float beta = 0.0;
	int threadnum = MAX_THREADNUM > hidden.U_lgrad.cols() ? hidden.U_lgrad.cols() : MAX_THREADNUM;
	int tmpSize = hidden.U_lgrad.sizes() * delta_l.ts();
	tmpMemory mem(tmpSize);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		tmpRes = cuMatrix4d(tmpPtr, hidden.U_lgrad.rows(), hidden.U_lgrad.cols(), delta_l.channals(), delta_l.ts());
	} else {
		tmpRes = cuMatrix4d(hidden.U_lgrad.rows(), hidden.U_lgrad.cols(), delta_l.channals(), delta_l.ts());
		mem.set(tmpRes.data);
	}	
	assert(tmpSize != acti_sum.sizes());
	//U_lgrad
	cuMatrix4d_matMul(delta_l, acti_sum.t() , tmpRes);
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.U_lgrad.getDev(), hidden.U_l.getDev(), 
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("U_lgrad");
	//U_rgrad
	cuMatrix4d_matMul(delta_r, acti_sum.t(), tmpRes);
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.U_rgrad.getDev(), hidden.U_r.getDev(), 
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("U_rgrad");

	//U_ld2
	cuMatrix4d_matMul(delta_ld2, acti2_sum.t(), tmpRes);
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.U_ld2.getDev(),
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("U_ld2");
	//U_rd2
	cuMatrix4d_matMul(delta_rd2, acti2_sum.t(), tmpRes);
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.U_rd2.getDev(), 
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("U_rd2");
}

void hiddenGetWgrad(cuMatrix4d& delta_l, cuMatrix4d& delta_r,
		cuMatrix4d& delta_ld2, cuMatrix4d& delta_rd2,
		cuMatrix4d& acti_l, cuMatrix4d& acti_r, 
		cuMatrix4d& acti_l2, cuMatrix4d& acti_r2, HiddenLayer& hidden, float WeightDecay){
	cuMatrix4d tmpRes;
	int ts = delta_l.ts();
	float alpha = 1.0;
	float beta = 0.0;
	int threadnum = MAX_THREADNUM > hidden.U_lgrad.cols() ? hidden.U_lgrad.cols() : MAX_THREADNUM;
	int nSamples = acti_l.cols();
	int tmpSize = hidden.W_lgrad.sizes() * (delta_l.ts() - 1);
	tmpMemory mem(tmpSize);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		tmpRes = cuMatrix4d(tmpPtr, hidden.W_lgrad.rows(), hidden.W_lgrad.cols(), delta_l.channals(), delta_l.ts() - 1);
	} else {
		tmpRes = cuMatrix4d(hidden.W_lgrad.rows(), hidden.W_lgrad.cols(), delta_l.channals(), delta_l.ts() - 1);
		mem.set(tmpRes.data);
	}	
	// W_lgrad
	cuMatrix4d acti_t = acti_l.t();
	for(int i = 1 ; i < delta_l.ts() ; i ++){
		for(int j = 0 ; j < tmpRes.channals() ; j ++){
			cublasStatus_t stat;
			float* s1 = delta_l.data->getDev() + i*delta_l.area3D() + j*delta_l.area2D();
			float* s2 = acti_t.data->getDev() + (i-1)*acti_t.area3D() + j*acti_t.area2D();
			float* d =  tmpRes.data->getDev() + (i-1)*tmpRes.area3D() + j*tmpRes.area2D();
			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti_t.cols(),
					delta_l.rows(), acti_t.rows(), &alpha, s2, acti_t.cols(),
					s1, delta_l.cols(), &beta, 
					d, tmpRes.cols());
			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf("W_lgrad error\n");
				exit(0);
			}
		}
	}
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.W_lgrad.getDev(), hidden.W_l.getDev(), 
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("W_lgrad");
	// W_rgrad
	acti_t = acti_r.t();
	for(int i = 0 ; i < tmpRes.ts() ; i ++){
		for(int j = 0 ; j < tmpRes.channals() ; j ++){
			cublasStatus_t stat;
			float* s1 = delta_r.data->getDev() + i*delta_r.area3D() + j*delta_r.area2D();
			float* s2 = acti_t.data->getDev() + (i+1)*acti_t.area3D() + j*acti_t.area2D();
			float* d  = tmpRes.data->getDev() + (i)*tmpRes.area3D() + j*tmpRes.area2D();
			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti_t.cols(),
					delta_l.rows(), acti_t.rows(), &alpha, s2, acti_t.cols(),
					s1, delta_l.cols(), &beta, 
					d, tmpRes.cols());
			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf("W_lrgrad error\n");
				exit(0);
			}
		}
	}
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.W_rgrad.getDev(), hidden.W_r.getDev(),
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("W_lgrad");
	// W_ld2
	acti_t = acti_l2.t();
	for(int i = 1 ; i < delta_l.ts(); i ++){
		for(int j = 0 ; j < tmpRes.channals() ; j ++){
			cublasStatus_t stat;
			float* s1 = delta_ld2.data->getDev() + i*delta_ld2.area3D() + j*delta_ld2.area2D();
			float* s2 = acti_t.data->getDev() + (i-1)*acti_t.area3D() + j*acti_t.area2D();
			float* d  = tmpRes.data->getDev() + (i-1)*tmpRes.area3D() + j*tmpRes.area2D();
			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti_t.cols(),
					delta_l.rows(), acti_t.rows(), &alpha, s2, acti_t.cols(),
					s1, delta_l.cols(), &beta, 
					d, tmpRes.cols());
			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf("W_ld2 error\n");
				exit(0);
			}
		}
	}
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.W_ld2.getDev(), 
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("W_ld2");
	// W_rd2
	acti_t = acti_r2.t();
	for(int i = 0 ; i < tmpRes.ts() ; i ++){
		for(int j = 0 ; j < tmpRes.channals() ; j ++){
			cublasStatus_t stat;
			float* s1 = delta_rd2.data->getDev() + i*delta_rd2.area3D() + j*delta_rd2.area2D();
			float* s2 = acti_t.data->getDev() + (i+1)*acti_t.area3D() + j*acti_t.area2D();
			float* d  = tmpRes.data->getDev() + i*tmpRes.area3D() + j*tmpRes.area2D();
			stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti_t.cols(),
					delta_l.rows(), acti_t.rows(), &alpha, s2, acti_t.cols(),
					s1, delta_l.cols(), &beta, 
					d, tmpRes.cols());
			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf("W_rd2 error\n");
				exit(0);
			}
		}
	}
	gradKernle<<<dim3(tmpRes.rows()), dim3(threadnum)>>>(tmpRes.getDev(), hidden.W_rd2.getDev(), 
			WeightDecay, nSamples, tmpRes.area2D(), tmpRes.ts(), tmpRes.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("W_rd2");
}


__global__ void bpttInitKernel(float *l, float* r, int col) {
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < col) {
		float sum = l[x * col + y] + r[x * col + y];
		l[x * col + y] = sum;
		r[x * col + y] = sum;
		y += blockDim.x;
	}
}

void bpttInit(HiddenLayer& hidden, cuMatrix4d& delta_l1, cuMatrix4d& delta_r1,
		cuMatrix4d& delta_l, cuMatrix4d& delta_r,cuMatrix4d& delta_ld1, cuMatrix4d& delta_rd1,
		cuMatrix4d& delta_ld, cuMatrix4d& delta_rd) {
	cuMatrix4d_matMul(hidden.U_l.t(), delta_l1, delta_l);
	cuMatrix4d_matMul(hidden.U_r.t(), delta_r1, delta_r);
	cuMatrix4d_matMul(Pow(hidden.U_l.t(),2), delta_ld1, delta_ld);
	cuMatrix4d_matMul(Pow(hidden.U_r.t(),2), delta_rd1, delta_rd);
	int threadnum =
		MAX_THREADNUM > delta_l.cols() ? delta_l.cols() : MAX_THREADNUM;
	bpttInitKernel<<<dim3(delta_l.rows() * delta_l.ts() * delta_l.channals()),
		dim3(threadnum)>>>(delta_l.getDev(), delta_r.getDev(),
				delta_l.cols());
	bpttInitKernel<<<dim3(delta_l.rows() * delta_l.ts() * delta_l.channals()),
		dim3(threadnum)>>>(delta_ld.getDev(), delta_rd.getDev(),
				delta_l.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("bpttInit");
}
