#include "costGradient.h"
void costParamentInit(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR) {
	int HiddenNum = Config::instance()->HiddenConfigs.size();
	int T = Config::instance()->get_ngram();
	acti_l = vector<cuMatrix4d>(HiddenNum + 1);
	acti_r = vector<cuMatrix4d>(HiddenNum + 1);
	acti_l2 = vector<cuMatrix4d>(HiddenNum + 1);
	acti_r2 = vector<cuMatrix4d>(HiddenNum + 1);
	acti_sum = vector<cuMatrix4d>(HiddenNum + 1);
	acti2_sum = vector<cuMatrix4d>(HiddenNum + 1);

	nonlin_l = vector<cuMatrix4d>(HiddenNum);
	nonlin_r = vector<cuMatrix4d>(HiddenNum);
	bernoulli_l = vector<cuMatrix4d>(HiddenNum);
	bernoulli_r = vector<cuMatrix4d>(HiddenNum);
	delta_l = vector<cuMatrix4d>(HiddenNum + 1);
	delta_ld2 = vector<cuMatrix4d>(HiddenNum + 1);
	delta_r = vector<cuMatrix4d>(HiddenNum + 1);
	delta_rd2 = vector<cuMatrix4d>(HiddenNum + 1);
	int r, c;
	r = Config::instance()->get_wordNum();
	c = Config::instance()->get_batch_size();
	acti_l2[0] = cuMatrix4d(r,c,1,Config::instance()->get_ngram());
	acti_r2[0] = cuMatrix4d(r,c,1,Config::instance()->get_ngram());
	acti2_sum[0] = cuMatrix4d(r,c,1,Config::instance()->get_ngram());
	//	acti_l[0] = cuMatrix4d(r,c,1,Config::instance()->get_ngram());
	//	acti_r[0] = cuMatrix4d(r,c,1,Config::instance()->get_ngram());
	for(int i = 1 ; i <= HiddenNum ; i ++){	
		r = Hiddenlayers[i - 1].U_l.rows();
		c = i == 1 ?
			Config::instance()->get_batch_size() :
			acti_l[i - 1].cols();
		acti_l[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		acti_r[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		acti_l2[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		acti_r2[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		acti_sum[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		acti2_sum[i] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		nonlin_l[i - 1] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		nonlin_r[i - 1] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		bernoulli_l[i - 1] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
		bernoulli_r[i - 1] = cuMatrix4d(r, c, 1, Config::instance()->get_ngram());
	}

	p = cuMatrix4d(SMR.get_NumClasses(),acti_l[acti_l.size() - 1].cols(),1,Config::instance()->get_ngram());
	groundTruth = cuMatrix4d(SMR.get_NumClasses(),acti_l[acti_l.size() - 1].cols(),1,Config::instance()->get_ngram());
	dis = cuMatrix4d(SMR.get_NumClasses(),acti_l[acti_l.size() - 1].cols(),1,Config::instance()->get_ngram());
	dis2 = cuMatrix4d(SMR.get_NumClasses(),acti_l[acti_l.size() - 1].cols(),1,Config::instance()->get_ngram());

	delta_l[delta_l.size() - 1] = cuMatrix4d(SMR.W_l.cols(),dis.cols(),1,Config::instance()->get_ngram());
	delta_ld2[delta_l.size() - 1] = cuMatrix4d(SMR.W_l.cols(),dis.cols(),1,Config::instance()->get_ngram());
	delta_r[delta_l.size() - 1] = cuMatrix4d(SMR.W_l.cols(),dis.cols(),1,Config::instance()->get_ngram());
	delta_rd2[delta_l.size() - 1] = cuMatrix4d(SMR.W_l.cols(),dis.cols(),1,Config::instance()->get_ngram());
	for (int i = delta_l.size() - 2 ; i > 0; i--) {
		delta_l[i] = cuMatrix4d(Hiddenlayers[i].U_l.cols(), delta_l[i + 1].cols(), 1, Config::instance()->get_ngram());
		delta_ld2[i] = cuMatrix4d(Hiddenlayers[i].U_l.cols(), delta_l[i + 1].cols(), 1, Config::instance()->get_ngram());
		delta_r[i] = cuMatrix4d(Hiddenlayers[i].U_r.cols(), delta_r[i + 1].cols(), 1, Config::instance()->get_ngram());
		delta_rd2[i] = cuMatrix4d(Hiddenlayers[i].U_r.cols(), delta_r[i + 1].cols(), 1, Config::instance()->get_ngram());
	}
	for(int i = 0 ; i < HiddenNum ; i ++){
		creatBnl(bernoulli_l[i],1.0f);
	}
}

void getNetworkCost(cuMatrix4d &acti_0, cuMatrix &sampleY,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR) {
	int T = acti_0.ts();
	int nSamples = acti_0.cols();
	int HiddenNum = Config::instance()->HiddenConfigs.size();
	
	acti_l[0] = acti_0;
	acti_r[0] = acti_0;
	acti_sum[0] = acti_0;
	square(acti_0,acti_l2[0]);
	square(acti_0,acti_r2[0]);
	square(acti_0,acti2_sum[0]);
	//hiddenlayer forward;
	for (int i = 1; i <= HiddenNum; i++) {
		if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()< 1.0) {
			creatBnl(bernoulli_l[i-1],Config::instance()->HiddenConfigs[i - 1].get_DropoutRate());
		}
		// time forward
		cuMatrix4d_matMul(Hiddenlayers[i-1].U_l, acti_sum[i-1],nonlin_l[i-1]);	
		hiddenForward(nonlin_l[i-1],acti_l[i],Hiddenlayers[i-1].W_l,bernoulli_l[i-1],TIMEFORWARD);
		square(acti_l[i],acti_l2[i]);
		//time backwoard
		cuMatrix4d_matMul(Hiddenlayers[i-1].U_r, acti_sum[i-1],nonlin_r[i-1]);	
		hiddenForward(nonlin_r[i-1],acti_r[i],Hiddenlayers[i-1].W_r,bernoulli_r[i-1],TIMEBACKWARD);
		square(acti_r[i],acti_r2[i]);
	
		for (int i = 1; i < acti_r.size(); i++) {
			cuMatrix4d_Add(acti_r[i], acti_l[i], acti_sum[i]);
			cuMatrix4d_Add(acti_r2[i], acti_l2[i], acti2_sum[i]);
		}
	}
	// softmax layer forward
        smrForward(SMR.W_r,acti_r[acti_r.size() - 1], SMR.W_l,acti_l[acti_l.size() - 1], p);
	set_groundtruth(groundTruth, sampleY);

	//cost function
	float j1 = 0.0f;
	float j2 = 0.0f;
	float j3 = 0.0f;
	float j4 = 0.0f;
	cuMatrix4d tmpMat = groundTruth.Mul(Log(p));
	j1 -= tmpMat.getSum();
	j1 /= nSamples;
	
	j2 = Pow(SMR.W_l, 2.0f).getSum();
	j2 += Pow(SMR.W_r, 2.0f).getSum();
	j2 = j2 * SMR.get_WeightDecay() / 2;

	for (int i = 0; i < Hiddenlayers.size(); i++) {
		j3 += Pow(Hiddenlayers[i].W_l, 2).getSum();
		j3 += Pow(Hiddenlayers[i].W_r, 2).getSum();
		j3 = j3 * Config::instance()->HiddenConfigs[i].get_WeightDecay() / 2;
	}
	for (int i = 0; i < Hiddenlayers.size(); i++) {
		j4 += Pow(Hiddenlayers[i].U_l, 2).getSum();
		j4 += Pow(Hiddenlayers[i].U_r, 2).getSum();
		j4 = j4 * Config::instance()->HiddenConfigs[i].get_WeightDecay() / 2;
	}
	SMR.cost = j1 + j2 + j3 + j4;

	printf("j1 = %f,j2 = %f,j3 = %f,j4 = %f,smr.cost = %f\n", j1, j2, j3, j4,SMR.cost);
	// SMR backward	(dis = - (groundTruth - p))
	cuDec(p, groundTruth, dis);
	square(dis,dis2);
	//Smr t-forward
	smrBP(SMR, acti_l[acti_l.size()-1], acti_r[acti_r.size()-1],acti_l2[acti_l2.size()-1] ,acti_r2[acti_r2.size()-1] , dis, dis2,nSamples);
	//BPTT for last hidden
	//get delta
	cuMatrix4d_matMul(SMR.W_l.t(),dis,delta_l[delta_l.size()-1]);
        hiddenBPTT(delta_l[delta_l.size() - 1], Hiddenlayers[Hiddenlayers.size() - 1].W_l.t(), nonlin_l[nonlin_l.size() - 1], bernoulli_l[bernoulli_l.size() - 1], TIMEFORWARD);
	cuMatrix4d_matMul(SMR.W_r.t(),dis,delta_r[delta_r.size()-1]);
        hiddenBPTT(delta_r[delta_r.size() - 1], Hiddenlayers[Hiddenlayers.size() - 1].W_r.t(), nonlin_r[nonlin_r.size() - 1], bernoulli_r[bernoulli_r.size() - 1], TIMEBACKWARD);

	cuMatrix4d_matMul(Pow(SMR.W_l.t(), 2),dis2,delta_ld2[delta_ld2.size()-1]);
        hiddenBPTT(delta_ld2[delta_ld2.size() - 1], Pow(Hiddenlayers[Hiddenlayers.size() - 1].W_l.t(),2), nonlin_l[nonlin_l.size() - 1], bernoulli_l[bernoulli_l.size() - 1], TIMEFORWARD);
	cuMatrix4d_matMul(Pow(SMR.W_r.t(),2),dis2,delta_rd2[delta_rd2.size()-1]);
        hiddenBPTT(delta_rd2[delta_rd2.size() - 1], Pow(Hiddenlayers[Hiddenlayers.size() - 1].W_r.t(),2), nonlin_r[nonlin_r.size() - 1], bernoulli_r[bernoulli_r.size() - 1], TIMEBACKWARD);
	
	for (int i = delta_l.size() - 2; i > 0; i--){
		cuMatrix4d_matMul(Hiddenlayers[i].U_l.t(), delta_l[i + 1],delta_l[i]);
		hiddenBPTT(delta_l[i], Hiddenlayers[i-1].W_l.t(),nonlin_l[i],bernoulli_l[i],TIMEFORWARD);
		cuMatrix4d_matMul(Hiddenlayers[i].U_r.t(), delta_r[i + 1],delta_r[i]);
		hiddenBPTT(delta_r[i], Hiddenlayers[i-1].W_r.t(),nonlin_r[i],bernoulli_r[i],TIMEBACKWARD);


		cuMatrix4d_matMul(Pow(Hiddenlayers[i].U_l.t(),2), delta_ld2[i + 1],delta_ld2[i]);
		hiddenBPTT(delta_ld2[i], Pow(Hiddenlayers[i-1].W_l.t(),2),nonlin_l[i],bernoulli_l[i],TIMEFORWARD);
		cuMatrix4d_matMul(Pow(Hiddenlayers[i].U_r.t(),2), delta_rd2[i + 1],delta_rd2[i]);
		hiddenBPTT(delta_rd2[i], Pow(Hiddenlayers[i-1].W_r.t(),2),nonlin_r[i],bernoulli_r[i],TIMEBACKWARD);
	}	
	//get grad
	for (int i = HiddenNum - 1; i >= 0; i--){
		hiddenGetUgrad(delta_l[i+1],delta_r[i+1],delta_ld2[i+1],delta_rd2[i+1],acti_sum[i],acti2_sum[i],Hiddenlayers[i],
				Config::instance()->HiddenConfigs[i].get_WeightDecay());
		hiddenGetWgrad(delta_l[i+1],delta_r[i+1],delta_ld2[i+1],delta_rd2[i+1],acti_l[i+1],acti_r[i+1],acti_l2[i+1],acti_r2[i+1],
			      Hiddenlayers[i], Config::instance()->HiddenConfigs[i].get_WeightDecay());
	} 
	for (int x = 0; x < T; x++) {
		cudaMemsetAsync(acti_0.getDev(), 0, acti_0.sizes(), 0);
		cudaMemsetAsync(groundTruth.getDev(), 0, groundTruth.sizes(), 0);
	}

}
