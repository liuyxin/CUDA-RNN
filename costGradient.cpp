#include "costGradient.h"
void costParamentInit(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR) {
	int HiddenNum = Config::instance()->HiddenConfigs.size();
	int T = Config::instance()->get_ngram();
	acti_l = vector<vector<cuMatrix> >(HiddenNum + 1);
	acti_r = vector<vector<cuMatrix> >(HiddenNum + 1);
	acti_l2 = vector<vector<cuMatrix> >(HiddenNum + 1);
	acti_r2 = vector<vector<cuMatrix> >(HiddenNum + 1);
	acti_sum = vector<vector<cuMatrix> >(HiddenNum + 1);
	acti2_sum = vector<vector<cuMatrix> >(HiddenNum + 1);

	nonlin_l = vector<vector<cuMatrix> >(HiddenNum);
	nonlin_r = vector<vector<cuMatrix> >(HiddenNum);
	bernoulli_l = vector<vector<cuMatrix> >(HiddenNum);
	bernoulli_r = vector<vector<cuMatrix> >(HiddenNum);
	delta_l = vector<vector<cuMatrix> >(HiddenNum + 1);
	delta_ld2 = vector<vector<cuMatrix> >(HiddenNum + 1);
	delta_r = vector<vector<cuMatrix> >(HiddenNum + 1);
	delta_rd2 = vector<vector<cuMatrix> >(HiddenNum + 1);

	int r, c;
	for (int i = 1; i <= HiddenNum; i++) {
		for (int j = 0; j < T; j++) {
			r = Hiddenlayers[i - 1].U_l.rows();
			c = i == 1 ?
					Config::instance()->get_batch_size() :
					acti_l[i - 1][0].cols();
			acti_l[i].push_back(cuMatrix(r, c));
			acti_r[i].push_back(cuMatrix(r, c));
			acti_l2[i].push_back(cuMatrix(r, c));
			acti_r2[i].push_back(cuMatrix(r, c));
			acti_sum[i].push_back(cuMatrix(r, c));
			acti2_sum[i].push_back(cuMatrix(r, c));
			nonlin_l[i - 1].push_back(cuMatrix(r, c));
			nonlin_r[i - 1].push_back(cuMatrix(r, c));
			bernoulli_l[i - 1].push_back(cuMatrix(r, c));
			bernoulli_r[i - 1].push_back(cuMatrix(r, c));
		}
	}
	r = Config::instance()->get_wordNum();
	c = Config::instance()->get_batch_size();
	for (int i = 0; i < T; i++) {
		acti_l2[0].push_back(cuMatrix(r, c));
		acti_r2[0].push_back(cuMatrix(r, c));
		acti_l[0].push_back(cuMatrix(r, c));
		acti_r[0].push_back(cuMatrix(r, c));
	}

	for (int j = 0; j < T; j++) {
		p.push_back(
				cuMatrix(SMR.get_NumClasses(),
						acti_l[acti_l.size() - 1][j].cols()));
		groundTruth.push_back(
				cuMatrix(SMR.get_NumClasses(),
						acti_l[acti_l.size() - 1][j].cols()));
		dis.push_back(
				cuMatrix(SMR.get_NumClasses(),
						acti_l[acti_l.size() - 1][j].cols()));
		dis2.push_back(
				cuMatrix(SMR.get_NumClasses(),
						acti_l[acti_l.size() - 1][j].cols()));
	}

	for (int j = 0; j < T; j++) {
		delta_l[delta_l.size() - 1].push_back(cuMatrix(SMR.W_l.cols(), dis[0].cols()));
		delta_ld2[delta_l.size() - 1].push_back(cuMatrix(SMR.W_l.cols(), dis[0].cols()));
		delta_r[delta_l.size() - 1].push_back(cuMatrix(SMR.W_r.cols(), dis[0].cols()));
		delta_rd2[delta_l.size() - 1].push_back(cuMatrix(SMR.W_r.cols(), dis[0].cols()));
	}
	for (int i = delta_l.size() - 2 ; i > 0; i--) {
		for (int j = 0; j < T; j++) {
//			hLayers[i].U_l.t() * delta_l[i + 1][j];
			delta_l[i].push_back(cuMatrix(Hiddenlayers[i].U_l.cols(), delta_l[i + 1][j].cols()));
			delta_ld2[i].push_back(cuMatrix(Hiddenlayers[i].U_l.cols(), delta_l[i + 1][j].cols()));
			delta_r[i].push_back(cuMatrix(Hiddenlayers[i].U_r.cols(), delta_r[i + 1][j].cols()));
			delta_rd2[i].push_back(cuMatrix(Hiddenlayers[i].U_r.cols(), delta_r[i + 1][j].cols()));
		}
	}
}

void getNetworkCost(cuMatrixVector &acti_0, cuMatrix &sampleY,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR) {
	int T = acti_0.size();
	int nSamples = acti_0[0]->cols();
	int HiddenNum = Config::instance()->HiddenConfigs.size();

	for (int j = 0; j < T; j++) {
		cuMatrix *ptr = acti_0[j];
		acti_l[0][j] = *ptr;
		acti_l[0][j].Square2(acti_l2[0][j]);
		acti_r[0][j] = *ptr;
		acti_r[0][j].Square2(acti_r2[0][j]);
	}

//hiddenlayer forward;
	for (int i = 1; i <= HiddenNum; i++) {
// time forward
		for (int j = 0; j < T; j++) {

			cuMultiplication(Hiddenlayers[i - 1].U_l, acti_l[i - 1][j],
					nonlin_l[i - 1][j]);
			if (j > 0) {
				nonlin_l[i - 1][j] += Hiddenlayers[i - 1].W_l
						* acti_l[i][j - 1];
			}
			if (i > 1) {
				nonlin_l[i - 1][j] += Hiddenlayers[i - 1].U_l
						* acti_r[i - 1][j];
			}
			nonlin_l[i - 1][j].ReLU2(acti_l[i][j]);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()
					< 1.0) {
				creatBnl(bernoulli_l[i - 1][j],
						Config::instance()->HiddenConfigs[i - 1].get_DropoutRate());
				bernoulli_l[i - 1][j].Mul2(acti_l[i][j], acti_l[i][j]);
			}
			acti_l[i][j].Square2(acti_l2[i][j]);
		}

//time backwoard
		for (int j = T - 1; j >= 0; j--) {
			cuMultiplication(Hiddenlayers[i - 1].U_r, acti_r[i - 1][j],
					nonlin_r[i - 1][j]);
			if (j < T - 1) {
				nonlin_r[i - 1][j] += Hiddenlayers[i - 1].W_r
						* acti_r[i][j + 1];
			}
			if (i > 1) {
				nonlin_r[i - 1][j] += Hiddenlayers[i - 1].U_r
						* acti_l[i - 1][j];
			}
			nonlin_r[i - 1][j].ReLU2(acti_r[i][j]);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()
					< 1.0) {
				creatBnl(bernoulli_r[i - 1][j],
						Config::instance()->HiddenConfigs[i - 1].get_DropoutRate());
				bernoulli_r[i - 1][j].Mul2(acti_r[i][j], acti_r[i][j]);
			}
			acti_r[i][j].Square2(acti_r2[i][j]);
		}
	}

	for (int i = 1; i < acti_r.size(); i++) {
		for (int j = 0; j < T; j++) {
			cuPlus(acti_r[i][j], acti_l[i][j], acti_sum[i][j]);
			cuPlus(acti_r2[i][j], acti_l2[i][j], acti2_sum[i][j]);
		}
	}
// softmax layer forward

	for (int i = 0; i < T; i++) {
		cuMultiplication(SMR.W_l, acti_l[acti_l.size() - 1][i], p[i]);
		p[i] += SMR.W_r * acti_r[acti_r.size() - 1][i];
		p[i] -= reduceMax(p[i]);
		Exp(p[i], p[i]);
		p[i] /= reduceSum(p[i]);
	}
	cuMatrixVector groundTruth_tmp;

	for (int i = 0; i < T; i++) {
		cuMatrix* tmp = &groundTruth[i];
		groundTruth_tmp.push_back(tmp);
	}
	groundTruth_tmp.toGpu();
	set_groundtruth(groundTruth_tmp, sampleY);

//cost function
	float j1 = 0.0f;
	float j2 = 0.0f;
	float j3 = 0.0f;
	float j4 = 0.0f;
	for (int i = 0; i < T; i++) {
		cuMatrix cumat = groundTruth[i].Mul(Log(p[i]));
		float tmpj = cumat.getSum();
		j1 -= tmpj;
	}

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
//	SMR.cost = j1 + j2 + j3 + j4;
//
//	printf("j1 = %f,j2 = %f,j3 = %f,j4 = %f,smr.cost = %f\n", j1, j2, j3, j4,
//			SMR.cost);

// SMR backward
	for (int i = 0; i < T; i++) {
		cuDec(groundTruth[i], p[i], dis[i]);
		dis[i].Square2(dis2[i]);
	}
//Smr t-forward

	cuMultiplication(dis[0], acti_l[acti_l.size() - 1][0].t(), SMR.W_lgrad);
	SMR.W_lgrad *= -1.0f;
	for (int i = 1; i < T; i++) {
		SMR.W_lgrad -= dis[i] * acti_l[acti_l.size() - 1][i].t();
	}
	SMR.W_lgrad /= nSamples;
	SMR.W_lgrad += SMR.W_l * SMR.get_WeightDecay();

	cuMultiplication(dis2[0], acti_l2[acti_l2.size() - 1][0].t(), SMR.W_ld2);
	for (int i = 1; i < T; i++) {
		SMR.W_ld2 += dis2[i] * acti_l2[acti_l2.size() - 1][i].t();
	}
	SMR.W_ld2 /= nSamples;
	SMR.W_ld2 += SMR.get_WeightDecay();

//Smr t-backward
	cuMultiplication(dis[0], acti_r[acti_r.size() - 1][0].t(), SMR.W_rgrad);
	SMR.W_rgrad *= -1.0f;
	for (int i = 1; i < T; i++) {
		SMR.W_rgrad -= dis[i] * acti_r[acti_r.size() - 1][i].t();
	}
	SMR.W_rgrad /= nSamples;
	SMR.W_rgrad += SMR.W_r * SMR.get_WeightDecay();

	cuMultiplication(dis2[0], acti_r2[acti_r2.size() - 1][0].t(), SMR.W_rd2);
	for (int i = 1; i < T; i++) {
		SMR.W_rd2 += dis2[i] * acti_r2[acti_r2.size() - 1][i].t();
	}
	SMR.W_rd2 /= nSamples;
	SMR.W_rd2 += SMR.get_WeightDecay();

//BPTT for last hidden
//time forward
	for (int i = T - 1; i >= 0; i--) {
		cuMultiplication(SMR.W_l.t(), dis[i], delta_l[delta_l.size() - 1][i]);
		delta_l[delta_l.size() - 1][i] *= -1.0f;
		cuMultiplication(Pow(SMR.W_l.t(), 2), dis2[i],
				delta_ld2[delta_ld2.size() - 1][i]);
		if (i < T - 1) {
			delta_l[delta_l.size() - 1][i] += Hiddenlayers[Hiddenlayers.size()
					- 1].W_l.t() * delta_l[delta_l.size() - 1][i + 1];
			delta_ld2[delta_ld2.size() - 1][i] += Pow(
					Hiddenlayers[Hiddenlayers.size() - 1].W_l.t(), 2)
					* delta_ld2[delta_ld2.size() - 1][i + 1];
		}
		delta_l[delta_l.size() - 1][i].Mul2(
				dReLU(nonlin_l[nonlin_l.size() - 1][i]),
				delta_l[delta_l.size() - 1][i]);
		delta_ld2[delta_ld2.size() - 1][i].Mul2(
				Pow(dReLU(nonlin_l[nonlin_l.size() - 1][i]), 2.0),
				delta_ld2[delta_ld2.size() - 1][i]);
		if (Config::instance()->HiddenConfigs[HiddenNum - 1].get_DropoutRate()
				< 1.0) {
			bernoulli_l[bernoulli_l.size() - 1][i].Mul2(
					delta_l[delta_l.size() - 1][i],
					delta_l[delta_l.size() - 1][i]);
			bernoulli_l[bernoulli_l.size() - 1][i].Mul2(
					delta_ld2[delta_ld2.size() - 1][i],
					delta_ld2[delta_ld2.size() - 1][i]);
		}
	}
//time backward
	for (int i = 0; i < T; i++) {
		cuMultiplication(SMR.W_r.t(), dis[i], delta_r[delta_r.size() - 1][i]);
		delta_r[delta_r.size() - 1][i] *= -1.0f;
		cuMultiplication(Pow(SMR.W_r.t(), 2), dis2[i],
				delta_rd2[delta_rd2.size() - 1][i]);
		if (i > 0) {
			delta_r[delta_r.size() - 1][i] += Hiddenlayers[Hiddenlayers.size()
					- 1].W_r.t() * delta_r[delta_r.size() - 1][i - 1];
			delta_rd2[delta_rd2.size() - 1][i] += Pow(
					Hiddenlayers[Hiddenlayers.size() - 1].W_r.t(), 2)
					* delta_rd2[delta_rd2.size() - 1][i - 1];
		}
		delta_r[delta_r.size() - 1][i].Mul2(
				dReLU(nonlin_r[nonlin_r.size() - 1][i]),
				delta_r[delta_r.size() - 1][i]);
		delta_rd2[delta_rd2.size() - 1][i].Mul2(
				Pow(dReLU(nonlin_r[nonlin_r.size() - 1][i]), 2.0),
				delta_rd2[delta_rd2.size() - 1][i]);
		if (Config::instance()->HiddenConfigs[HiddenNum - 1].get_DropoutRate()
				< 1.0) {
			bernoulli_r[bernoulli_r.size() - 1][i].Mul2(
					delta_r[delta_r.size() - 1][i],
					delta_r[delta_r.size() - 1][i]);
			bernoulli_r[bernoulli_r.size() - 1][i].Mul2(
					delta_rd2[delta_rd2.size() - 1][i],
					delta_rd2[delta_rd2.size() - 1][i]);
		}
	}
//*************  hidden layers **********************
	for (int i = delta_l.size() - 2; i > 0; i--) {
		//t-forward
		for (int j = T - 1; j >= 0; j--) {
			cuMultiplication(Hiddenlayers[i].U_l.t(), delta_l[i + 1][j],
					delta_l[i][j]);
			cuMultiplication(Pow(Hiddenlayers[i].U_l.t(), 2.0),
					delta_ld2[i + 1][j], delta_ld2[i][j]);

			if (j < T - 1) {
				delta_l[i][j] += Hiddenlayers[i - 1].W_l.t()
						* delta_l[i][j + 1];
				delta_ld2[i][j] += Pow(Hiddenlayers[i - 1].W_l.t(), 2.0)
						* delta_ld2[i][j + 1];
			}
			delta_l[i][j] += Hiddenlayers[i].U_r.t() * delta_r[i + 1][j];
			delta_ld2[i][j] += Pow(Hiddenlayers[i].U_r.t(), 2.0)
					* delta_rd2[i + 1][j];

			delta_l[i][j].Mul2(dReLU(nonlin_l[i - 1][j]), delta_l[i][j]);
			delta_ld2[i][j].Mul2(Pow(dReLU(nonlin_l[i - 1][j]), 2.0),
					delta_ld2[i][j]);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()
					< 1.0) {
				delta_l[i][j].Mul2(bernoulli_l[i - 1][j], delta_l[i][j]);
				delta_ld2[i][j].Mul2(bernoulli_l[i - 1][j], delta_ld2[i][j]);
			}
		}

		//t-backward
		for (int j = 0; j < T; j++) {
			cuMultiplication(Hiddenlayers[i].U_r.t(), delta_r[i + 1][j],
					delta_r[i][j]);
			cuMultiplication(Pow(Hiddenlayers[i].U_r.t(), 2.0),
					delta_rd2[i + 1][j], delta_rd2[i][j]);
			if (j > 0) {
				delta_r[i][j] += Hiddenlayers[i - 1].W_r.t()
						* delta_r[i][j - 1];
				delta_rd2[i][j] += Pow(Hiddenlayers[i - 1].W_r.t(), 2.0)
						* delta_rd2[i][j - 1];
			}
			delta_r[i][j] += Hiddenlayers[i].U_l.t() * delta_l[i + 1][j];
			delta_rd2[i][j] += Pow(Hiddenlayers[i].U_l.t(), 2.0)
					* delta_ld2[i + 1][j];

			delta_r[i][j].Mul2(dReLU(nonlin_r[i - 1][j]), delta_r[i][j]);
			delta_rd2[i][j].Mul2(Pow(dReLU(nonlin_r[i - 1][j]), 2.0),
					delta_rd2[i][j]);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()
					< 1.0) {
				delta_r[i][j].Mul2(bernoulli_r[i - 1][j], delta_r[i][j]);
				delta_rd2[i][j].Mul2(bernoulli_r[i - 1][j], delta_rd2[i][j]);
			}
		}
	}

	for (int i = HiddenNum - 1; i >= 0; i--) {
		// forward part.
		if (i == 0) {
			cuMultiplication(delta_l[i + 1][0], acti_l[i][0].t(),
					Hiddenlayers[i].U_lgrad);
			cuMultiplication(delta_ld2[i + 1][0], acti_l2[i][0].t(),
					Hiddenlayers[i].U_ld2);
			for (int j = 1; j < T; ++j) {
				Hiddenlayers[i].U_lgrad += delta_l[i + 1][j] * acti_l[i][j].t();
				Hiddenlayers[i].U_ld2 += delta_ld2[i + 1][j]
						* acti_l2[i][j].t();
			}
		} else {
			cuMultiplication(delta_l[i + 1][0], acti_sum[i][0].t(),
					Hiddenlayers[i].U_lgrad);
			cuMultiplication(delta_ld2[i + 1][0], acti2_sum[i][0].t(),
					Hiddenlayers[i].U_lgrad);
			for (int j = 1; j < T; j++) {
				Hiddenlayers[i].U_lgrad += delta_l[i + 1][j]
						* acti_sum[i][j].t();
				Hiddenlayers[i].U_ld2 += delta_ld2[i + 1][j]
						* acti2_sum[i][j].t();
			}
		}
		Hiddenlayers[i].U_lgrad /= nSamples;
		Hiddenlayers[i].U_lgrad += Hiddenlayers[i].U_l
				* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].U_ld2 /= nSamples;
		Hiddenlayers[i].U_ld2 +=
				Config::instance()->HiddenConfigs[i].get_WeightDecay();

		cuMultiplication(delta_l[i + 1][T - 1], acti_l[i + 1][T - 2].t(),
				Hiddenlayers[i].W_lgrad);
		cuMultiplication(delta_ld2[i + 1][T - 1], acti_l2[i + 1][T - 2].t(),
				Hiddenlayers[i].W_ld2);
		for (int j = T - 2; j > 0; j--) {
			Hiddenlayers[i].W_lgrad += delta_l[i + 1][j]
					* acti_l[i + 1][j - 1].t();
			Hiddenlayers[i].W_ld2 += delta_ld2[i + 1][j]
					* acti_l2[i + 1][j - 1].t();
		}
		Hiddenlayers[i].W_lgrad /= nSamples;
		Hiddenlayers[i].W_lgrad += Hiddenlayers[i].W_l
				* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].W_ld2 /= nSamples;
		Hiddenlayers[i].W_ld2 +=
				Config::instance()->HiddenConfigs[i].get_WeightDecay();

// backward part.
		if (i == 0) {
			cuMultiplication(delta_r[i + 1][0], acti_r[i][0].t(),
					Hiddenlayers[i].U_rgrad);
			cuMultiplication(delta_rd2[i + 1][0], acti_r2[i][0].t(),
					Hiddenlayers[i].U_rd2);
			for (int j = 1; j < T; ++j) {
				Hiddenlayers[i].U_rgrad += delta_r[i + 1][j] * acti_r[i][j].t();
				Hiddenlayers[i].U_rd2 += delta_rd2[i + 1][j]
						* acti_r2[i][j].t();
			}
		} else {
			cuMultiplication(delta_r[i + 1][0], acti_sum[i][0].t(),
					Hiddenlayers[i].U_rgrad);
			cuMultiplication(delta_rd2[i + 1][0], acti2_sum[i][0].t(),
					Hiddenlayers[i].U_rgrad);
			for (int j = 1; j < T; j++) {
				Hiddenlayers[i].U_rgrad += delta_r[i + 1][j]
						* acti_sum[i][j].t();
				Hiddenlayers[i].U_rd2 += delta_rd2[i + 1][j]
						* acti2_sum[i][j].t();
			}
		}
		Hiddenlayers[i].U_rgrad /= nSamples;
		Hiddenlayers[i].U_rgrad += Hiddenlayers[i].U_r
				* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].U_rd2 /= nSamples;
		Hiddenlayers[i].U_rd2 +=
				Config::instance()->HiddenConfigs[i].get_WeightDecay();

		cuMultiplication(delta_r[i + 1][0], acti_r[i + 1][1].t(),
				Hiddenlayers[i].W_rgrad);
		cuMultiplication(delta_rd2[i + 1][0], acti_r2[i + 1][1].t(),
				Hiddenlayers[i].W_rd2);
		for (int j = 1; j < T - 1; j++) {
			Hiddenlayers[i].W_rgrad += delta_r[i + 1][j]
					* acti_r[i + 1][j + 1].t();
			Hiddenlayers[i].W_rd2 += delta_rd2[i + 1][j]
					* acti_r2[i + 1][j + 1].t();
		}
		Hiddenlayers[i].W_rgrad /= nSamples;
		Hiddenlayers[i].W_rgrad += Hiddenlayers[i].W_r
				* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].W_rd2 /= nSamples;
		Hiddenlayers[i].W_rd2 +=
				Config::instance()->HiddenConfigs[i].get_WeightDecay();
	}
	for (int x = 0; x < 5; x++) {
		cudaMemsetAsync(acti_0[x]->getDev(), 0, acti_0[x]->sizes(), 0);
		cudaMemsetAsync(groundTruth[x].getDev(), 0, groundTruth[x].sizes(), 0);
	}

}
