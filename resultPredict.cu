#include "resultPredict.h"
const int MAX_THREADNUM = Devices::instance()->maxThreadNum();
__device__ float max_(float a, float b){
	return a>b?a:b;
}

void testInit(vector<HiddenLayer> &Hiddenlayers) {
	int HiddenNum = Config::instance()->HiddenConfigs.size();
	al = std::vector<cuMatrix4d>(HiddenNum + 1);
	ar = std::vector<cuMatrix4d>(HiddenNum + 1);
	as = std::vector<cuMatrix4d>(HiddenNum + 1);
	for (int i = 1; i < HiddenNum + 1; i++) {
		int r = Hiddenlayers[i - 1].U_l.rows();
		int c = i == 1 ?
			Config::instance()->get_batch_size() : al[i - 1].cols();
		al[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		ar[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		as[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
	}
}


void testNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR, bool flag) {
	int size =
		flag ? Config::instance()->trainXNum() : Config::instance()->testXNum();
	int wn = Config::instance()->get_wordNum();
	int* res = new int[size];
	int* truth = new int[size];
	int offset;
	int batch_size = Config::instance()->get_batch_size();
	int batch_amount = size / batch_size;
	for (int i = 0; i < batch_amount; i++) {
		offset = i * batch_size;
		cuMatrix4d sampleX(wn, batch_size, 1, Config::instance()->get_ngram());
		getDataMat(sampleX, offset, batch_size, wn, flag);

		predict(sampleX, Hiddenlayers, SMR, res, offset);
	}
	offset = batch_amount * batch_size;
	if (size % batch_size) {
		batch_size = size % batch_size;
		cuMatrix4d sampleX(wn, batch_size, 1, Config::instance()->get_ngram());
		for (int i = 1; i < Hiddenlayers.size() + 1; i++) {
			int r = Hiddenlayers[i - 1].U_l.rows();
			int c = i == 1 ?
				batch_size : al[i - 1].cols();
			al[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
			ar[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
			as[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		}
		getDataMat(sampleX, offset, batch_size, wn, size);
		predict(sampleX, Hiddenlayers, SMR, res, offset);
	}
	set_label(truth, size, flag);
	int error = 0;
	for (int i = 0; i < size; i++) {
		if (truth[i] != res[i]) {
			error++;
		}
	}
	float rate = (size - error) / (float) size;
	printf("total num : %d, correct : %d , correct rate: %f \n", size,
			size - error, rate);
	delete[] res;
	delete[] truth;
}

//void predict(cuMatrix4d &sampleX, vector<HiddenLayer> &Hiddenlayers,
//		SoftMax &SMR, int* output, int offset) {
//	int T = sampleX.sizes();
//	int HiddenNum = Config::instance()->HiddenConfigs.size();
//	std::vector<cuMatrix4d > acti_l(HiddenNum + 1);
//	std::vector<cuMatrix4d > acti_r(HiddenNum + 1);
//	std::vector<cuMatrix4d > acti_sum(HiddenNum + 1);
//	acti_l[0] = sampleX;
//	acti_r[0] = sampleX;
//	acti_sum[0] = sampleX;
//	for (int i = 1; i <= HiddenNum; ++i) {
//		float DropoutRate = Config::instance()->HiddenConfigs[i - 1].get_DropoutRate();
//		//time forward
//		cuMatrix4d_matMul(Hiddenlayers[i-1].U_l, acti_sum[i-1],acti_l[i]);	
//		hiddenForward_(acti_l[i],Hiddenlayers[i-1].W_l,DropoutRate,TIMEFORWARD);
//		//time backward
//		cuMatrix4d_matMul(Hiddenlayers[i-1].U_r, acti_sum[i-1],acti_r[i]);	
//		hiddenForward_(acti_r[i],Hiddenlayers[i-1].W_r,DropoutRate,TIMEBACKWARD);
//		for (int i = 1; i < acti_r.size(); i++) {
//			cuMatrix4d_Add(acti_r[i], acti_l[i], acti_sum[i]);
//		}
//	}
//	cuMatrix M(SMR.W_l.rows(), acti_l[acti_l.size() - 1].cols());
//	smrForward_(M,acti_l[acti_l.size() - 1],acti_r[acti_r.size() - 1],SMR);
//	get_res_array(M, output, offset);
//}


void predict(cuMatrix4d &sampleX, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR, int* output, int offset) {
	int T = sampleX.sizes();
	int HiddenNum = Config::instance()->HiddenConfigs.size();
	al[0] = sampleX;
	ar[0] = sampleX;
	as[0] = sampleX;
	for (int i = 1; i <= HiddenNum; ++i) {
		float DropoutRate =
			Config::instance()->HiddenConfigs[i - 1].get_DropoutRate();
		//time forward
		cuMatrix4d_matMul(Hiddenlayers[i - 1].U_l, as[i - 1], al[i]);
		hiddenForward_(al[i], Hiddenlayers[i - 1].W_l, DropoutRate,
				TIMEFORWARD);
		//time backward

		cuMatrix4d_matMul(Hiddenlayers[i - 1].U_r, as[i - 1], ar[i]);
		hiddenForward_(ar[i], Hiddenlayers[i - 1].W_r, DropoutRate,
				TIMEBACKWARD);
		for (int i = 1; i < ar.size(); i++) {
			cuMatrix4d_Add(ar[i], al[i], as[i]);
		}
	}
	cuMatrix M(SMR.W_l.rows(), al[al.size() - 1].cols());
	smrForward_(M, al[al.size() - 1], ar[ar.size() - 1], SMR);
	get_res_array(M, output, offset);
}
__global__ void n2aKernel(float* src,float* dst, float bnl,int col){	
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < col) {
		if (src[x * col + y] <= 0) {
			dst[x * col + y] = 0;
		} else {
			dst[x * col + y] = src[x * col + y] * bnl;
		}
		y += blockDim.x;
	}
}

void non2acti_(cuMatrix4d& non, cuMatrix4d& acti,float bnl,int t){
	assert(non.len() == acti.len());
	int threadnum = MAX_THREADNUM > non.cols() ? non.cols() : MAX_THREADNUM;
	n2aKernel<<<dim3(non.rows()), dim3(threadnum)>>>(non.getDev()+t*non.area2D(),
			acti.getDev()+t*acti.area2D(), bnl, non.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("non2acti");
}

__global__ void anaKernel(float* a, float* n,float* p, float bnl,int a2,int col){
	int x = blockIdx.x;
	int y = threadIdx.x;
	while (y < col) {
		n[x*col + y] += p[x*col+y];
		if (n[x * col + y] <= 0) {
			a[x * col + y] = 0;
		} else {
			a[x * col + y] = n[x * col + y] * bnl;
		}
		y += blockDim.x;
	}
}
// acti[t-1] -> nonlin[t] -> acti[t] or acti[t+1]->nonlin[t]->acti[t]
void acti2non2acti_(cuMatrix4d& acti, cuMatrix4d& non,float bnl ,cuMatrix& w,int t,bool f){
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
	anaKernel<<<dim3(non.rows()),dim3(threadnum)>>>(acti.getDev()+t*acti.area2D(),non.getDev()+t*non.area2D(),tmpRes.getDev(),bnl, acti.area2D(),acti.cols());
	checkCudaErrors(cudaStreamSynchronize(0));
	getLastCudaError("acti2non2acti");
}


void hiddenForward_(cuMatrix4d& acti,cuMatrix& weight, float dr, bool f){
	if(f == TIMEFORWARD){
		non2acti_(acti,acti,dr,0);			
		for(int t = 1 ; t < acti.ts() ; t++){
			acti2non2acti_(acti,acti,dr,weight,t,TIMEFORWARD);
		}
	}
	else{
		non2acti_(acti,acti,dr,Config::instance()->get_ngram()-1);
		for(int t = Config::instance()->get_ngram()-2 ; t >= 0 ; t--){
			acti2non2acti_(acti,acti,dr,weight,t,TIMEBACKWARD);
		}
	}
}


void smrForward_(cuMatrix& M, cuMatrix4d& acti_l, cuMatrix4d& acti_r, SoftMax& SMR){
	assert(SMR.W_l.cols() == acti_l.rows() && SMR.W_r.cols() == acti_r.rows());
	float alpha = 1.0;
	float beta = 0.0;
	int off = (acti_l.ts() / 2) * acti_l.area2D();
	cublasStatus_t stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti_l.cols(),
			SMR.W_l.rows(), acti_l.rows(), &alpha, acti_l.getDev() + off, acti_l.cols(),
			SMR.W_l.getDev(), SMR.W_l.cols(), &beta, M.getDev(), M.cols());
	cudaStreamSynchronize(0);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("predict.h, smrFORward SMR.W_l * acti_l[mid] error\n");
		exit(0);
	}
	cuMatrix tmpRes;
	int size = M.sizes();
	tmpMemory mem(size);
	shared_ptr<MatData> tmpPtr = mem.getMem();
	if (tmpPtr != NULL) {
		tmpRes = cuMatrix(tmpPtr, M.rows(), M.cols());
	} else {
		tmpRes = cuMatrix(M.rows(), M.cols());
		mem.set(tmpRes.data);
	}	
	stat = cublasSgemm(getHandle(), CUBLAS_OP_N, CUBLAS_OP_N, acti_r.cols(),
			SMR.W_r.rows(), acti_r.rows(), &alpha, acti_r.getDev() + off, acti_r.cols(),
			SMR.W_r.getDev(), SMR.W_r.cols(), &beta, tmpRes.getDev(), tmpRes.cols());
	cudaStreamSynchronize(0);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("predict.h, smrFORward SMR.W_r * acti_r[mid] error\n");
		exit(0);
	}
	M += tmpRes;
}

