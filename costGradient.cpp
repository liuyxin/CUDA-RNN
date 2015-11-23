#include "costGradient.h"
void getNetworkCost(cuMatrixVector &acti_0, cuMatrix &sampleY,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR) {
	int T = acti_0.size();
	int nSamples = acti_0[0]->cols();
	int HiddenNum = Config::instance()->HiddenConfigs.size();

	vector<vector<cuMatrix> > acti_l(HiddenNum + 1);
	vector<vector<cuMatrix> > acti_r(HiddenNum + 1);
	vector<vector<cuMatrix> > nonlin_l(HiddenNum);
	vector<vector<cuMatrix> > nonlin_r(HiddenNum);
//	vector<cuMatrixVector > bernoulli_l;
//	vector<cuMatrixVector > bernoulli_r;

	for (int i = 0; i < T; i++) {
		cuMatrix* ptr = acti_0[i];
		acti_l[0].push_back(*ptr);
		acti_r[0].push_back(*ptr);
	}
	//hiddenlayer forward;
	for (int i = 1; i <= HiddenNum; i++) {
		for (int j = 0; j < T; j++) {
			acti_l[i].push_back(cuMatrix(Hiddenlayers[i - 1].U_l.rows(),
					acti_l[i - 1][0].cols()));
			acti_r[i].push_back(cuMatrix(Hiddenlayers[i - 1].U_l.rows(),
					acti_l[i - 1][0].cols()));
			nonlin_l[i-1].push_back(cuMatrix(Hiddenlayers[i - 1].U_l.rows(),
					acti_l[i - 1][0].cols()));
			nonlin_r[i-1].push_back(cuMatrix(Hiddenlayers[i - 1].U_l.rows(),
					acti_l[i - 1][0].cols()));

		}
//		bernoulli_l.push_back(tmp_bl);
//		bernoulli_r.push_back(tmp_br);

// time forward
		for (int j = 0; j < T; j++) {
			cuMatrix tmpacti = Hiddenlayers[i - 1].U_l * (acti_l[i - 1][j]);
			if (j > 0)
				tmpacti = Hiddenlayers[i - 1].W_l * (acti_l[i][j - 1])
						+ tmpacti;
			if (i > 1)
				tmpacti = Hiddenlayers[i - 1].U_l * (acti_r[i - 1][j])
						+ tmpacti;
			tmpacti.copyTo(nonlin_l[i - 1][j]);
			tmpacti = ReLU(tmpacti);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()
					< 1.0) {
			} else
				tmpacti.copyTo(acti_l[i][j]);
		}
//time backwoard
		for (int j = T - 1; j >= 0; j--) {
			cuMatrix tmpacti = Hiddenlayers[i - 1].U_r * (acti_r[i - 1][j]);
			if (j < T - 1)
				tmpacti = Hiddenlayers[i - 1].W_r * (acti_r[i][j + 1])
						+ tmpacti;
			if (i > 1)
				tmpacti = Hiddenlayers[i - 1].U_r * (acti_l[i - 1][j])
						+ tmpacti;
			tmpacti.copyTo(nonlin_r[i - 1][j]);
			tmpacti = ReLU(tmpacti);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()
					< 1.0) {
			} else
				tmpacti.copyTo(acti_r[i][j]);
		}
	}
// softmax layer forward
	vector<cuMatrix> p;
	vector<cuMatrix> groundTruth;
	for (int i = 0; i < T; i++) {
		cuMatrix M = SMR.W_l * (acti_l[acti_l.size() - 1][i]);
		M = SMR.W_r * (acti_r[acti_r.size() - 1][i]) + M;
		M = M - reduceMax(M);
		M = Exp(M);
		M = M / reduceSum(M);
		p.push_back(M);
	}
	cuMatrixVector groundTruth_tmp;
	for (int i = 0; i < T; i++) {
		cuMatrix* tmp = new cuMatrix(SMR.get_NumClasses(), nSamples);
		groundTruth_tmp.push_back(tmp);
	}
	groundTruth_tmp.toGpu();
	set_groundtruth(groundTruth_tmp, sampleY);
	for (int i = 0; i < T; i++) {
		cuMatrix* ptr = groundTruth_tmp[i];
		groundTruth.push_back(*ptr);
	}

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
	SMR.cost = j1 + j2 + j3 + j4;

	printf("j1 = %f,j2 = %f,j3 = %f,j4 = %f,smr.cost = %f\n", j1, j2, j3, j4,
			SMR.cost);

// SMR backward
	vector<cuMatrix> dis;
	vector<cuMatrix> dis2;
	for (int i = 0; i < T; i++) {
		cuMatrix tmpdis = groundTruth[i] - p[i];
		cuMatrix tmpdis2 = Pow(tmpdis, 2);
		dis.push_back(tmpdis);
		dis2.push_back(tmpdis2);
	}
	//Smr t-forward
	cuMatrix Swl_tmp(dis[0].rows(), acti_l[acti_l.size() - 1][0].rows());
	for (int i = 0; i < T; i++) {
		Swl_tmp = Swl_tmp - dis[i] * acti_l[acti_l.size() - 1][i].t();
	}
	Swl_tmp = Swl_tmp / nSamples;
	SMR.W_lgrad = Swl_tmp + SMR.W_l * SMR.get_WeightDecay();
	cuMatrix Swld2_tmp(dis2[0].rows(), acti_l[acti_l.size() - 1][0].rows());
	for (int i = 0; i < T; i++) {
		Swld2_tmp = Swld2_tmp
				+ dis2[i] * Pow(acti_l[acti_l.size() - 1][i].t(), 2);
	}
	Swld2_tmp = Swld2_tmp / nSamples;
	SMR.W_ld2 = Swld2_tmp + SMR.get_WeightDecay();
	//Smr t-borward
	cuMatrix Swr_tmp(dis[0].rows(), acti_r[acti_r.size() - 1][0].rows());
	for (int i = 0; i < T; i++) {
		Swr_tmp = Swr_tmp - dis[i] * acti_r[acti_r.size() - 1][i].t();
	}
	Swr_tmp = Swr_tmp / nSamples;
	SMR.W_rgrad = Swr_tmp + SMR.W_r * SMR.get_WeightDecay();
	cuMatrix Swrd2_tmp(dis2[0].rows(), acti_r[acti_r.size() - 1][0].rows());
	for (int i = 0; i < T; i++) {
		Swrd2_tmp = Swrd2_tmp
				+ dis2[i] * Pow(acti_r[acti_r.size() - 1][i].t(), 2);
	}
	Swrd2_tmp = Swrd2_tmp / nSamples;
	SMR.W_rd2 = Swrd2_tmp + SMR.get_WeightDecay();



//BPTT for last hidden
	vector<cuMatrixVector> delta_l(acti_l.size());
	vector<cuMatrixVector> delta_ld2(acti_l.size());
	vector<cuMatrixVector> delta_r(acti_r.size());
	vector<cuMatrixVector> delta_rd2(acti_r.size());
	for (int i = 0; i < delta_l.size(); i++) {
		for (int j = 0; j < T; j++) {
			delta_l[i].push_back(new cuMatrix(SMR.W_l.cols(), dis[0].cols()));
			delta_ld2[i].push_back(new cuMatrix(SMR.W_l.cols(), dis[0].cols()));
			delta_r[i].push_back(new cuMatrix(SMR.W_r.cols(), dis[0].cols()));
			delta_rd2[i].push_back(new cuMatrix(SMR.W_r.cols(), dis[0].cols()));
		}
	}
	//time forward
	for (int i = T - 1; i >= 0; i--) {
		cuMatrix tmp = SMR.W_l.t() * dis[i] * (-1.0f);
		cuMatrix tmp2 = Pow(SMR.W_l.t(), 2) * dis2[i];
		if (i < T - 1) {
			tmp = tmp
					+ Hiddenlayers[Hiddenlayers.size() - 1].W_l.t()
							* (*delta_l[delta_l.size() - 1][i + 1]);
			tmp2 = tmp2
					+ Pow(Hiddenlayers[Hiddenlayers.size() - 1].W_l.t(), 2)
							* (*delta_ld2[delta_ld2.size() - 1][i + 1]);
		}
		tmp.copyTo(*delta_l[delta_l.size() - 1][i]);
		tmp2.copyTo(*delta_ld2[delta_ld2.size() - 1][i]);
		*delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() - 1][i]->Mul(
				dReLU(nonlin_l[nonlin_l.size() - 1][i]));
		*delta_ld2[delta_ld2.size() - 1][i] =
				delta_ld2[delta_ld2.size() - 1][i]->Mul(
						Pow(dReLU(nonlin_l[nonlin_l.size() - 1][i]), 2.0));
		if (Config::instance()->HiddenConfigs[HiddenNum - 1].get_WeightDecay()
				< 1.0) {
		}
	}
	//time backward
	for (int i = 0; i < T; i++) {
		cuMatrix tmp = SMR.W_r.t() * dis[i] * (-1.0f);
		cuMatrix tmp2 = Pow(SMR.W_r.t(), 2) * dis2[i];
		if (i > 0) {
			tmp = tmp
					+ Hiddenlayers[Hiddenlayers.size() - 1].W_r.t()
							* (*delta_r[delta_r.size() - 1][i - 1]);
			tmp2 = tmp2
					+ Pow(Hiddenlayers[Hiddenlayers.size() - 1].W_r.t(), 2)
							* (*delta_rd2[delta_rd2.size() - 1][i - 1]);
		}
		tmp.copyTo(*delta_r[delta_r.size() - 1][i]);
		tmp2.copyTo(*delta_rd2[delta_rd2.size() - 1][i]);
		*delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() - 1][i]->Mul(
				dReLU(nonlin_r[nonlin_r.size() - 1][i]));
		*delta_rd2[delta_rd2.size() - 1][i] =
				delta_rd2[delta_rd2.size() - 1][i]->Mul(
						Pow(dReLU(nonlin_r[nonlin_r.size() - 1][i]), 2.0));
		if (Config::instance()->HiddenConfigs[HiddenNum - 1].get_WeightDecay()
				< 1.0) {
		}
	}
//*************  hidden layers **********************
//****************************************************
//****************************************************
//****************************************************
//****************************************************
//***************************************************
	for (int i = HiddenNum - 1; i >= 0; i--) {
		// forward part.
		cuMatrix tmp;
		cuMatrix tmp2;
		if (i == 0) {
			tmp = *delta_l[i + 1][0] * acti_l[i][0].t();
			tmp2 = *delta_ld2[i + 1][0] * Pow(acti_l[i][0].t(), 2.0f);
			for (int j = 1; j < T; ++j) {
				tmp = tmp + *delta_l[i + 1][j] * acti_l[i][j].t();
				tmp2 = tmp2
						+ *delta_ld2[i + 1][j] * Pow(acti_l[i][j].t(), 2.0f);
			}
		} else {
			tmp = *delta_l[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
			tmp2 =
					*delta_ld2[i + 1][0]
							* (Pow(acti_l[i][0].t(), 2.0)
									+ Pow(acti_r[i][0].t(), 2.0));
			for (int j = 1; j < T; ++j) {
				tmp = tmp
						+ *delta_l[i + 1][j]
								* (acti_l[i][j].t() + acti_r[i][j].t());
				tmp2 = tmp2
						+ *delta_ld2[i + 1][j]
								* (Pow(acti_l[i][j].t(), 2.0)
										+ Pow(acti_r[i][j].t(), 2.0));
			}
		}
		Hiddenlayers[i].U_lgrad =
				tmp / nSamples
						+ Hiddenlayers[i].U_l
								* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].U_ld2 = tmp2 / nSamples
				+ Config::instance()->HiddenConfigs[i].get_WeightDecay();
		tmp = *delta_l[i + 1][T - 1] * acti_l[i + 1][T - 2].t();
		tmp2 = *delta_ld2[i + 1][T - 1] * Pow(acti_l[i + 1][T - 2].t(), 2.0);
		for (int j = T - 2; j > 0; j--) {
			tmp = tmp + *delta_l[i + 1][j] * acti_l[i + 1][j - 1].t();
			tmp2 = tmp2
					+ *delta_ld2[i + 1][j]
							* Pow(acti_l[i + 1][j - 1].t(), 2.0);
		}
		Hiddenlayers[i].W_lgrad =
				tmp / nSamples
						+ Hiddenlayers[i].W_l
								* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].W_ld2 = tmp2 / nSamples
				+ Config::instance()->HiddenConfigs[i].get_WeightDecay();
// backward part.
		if (i == 0) {
			tmp = *delta_r[i + 1][0] * acti_r[i][0].t();
			tmp2 = *delta_rd2[i + 1][0] * Pow(acti_r[i][0].t(), 2.0f);
			for (int j = 1; j < T; ++j) {
				tmp = tmp + *delta_r[i + 1][j] * acti_r[i][j].t();
				tmp2 = tmp2
						+ *delta_rd2[i + 1][j] * Pow(acti_r[i][j].t(), 2.0f);
			}
		} else {
			tmp = *delta_r[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
			tmp2 =
					*delta_rd2[i + 1][0]
							* (Pow(acti_l[i][0].t(), 2.0)
									+ Pow(acti_r[i][0].t(), 2.0));
			for (int j = 1; j < T; ++j) {
				tmp = tmp
						+ *delta_r[i + 1][j]
								* (acti_l[i][j].t() + acti_r[i][j].t());
				tmp2 = tmp2
						+ *delta_rd2[i + 1][j]
								* (Pow(acti_l[i][j].t(), 2.0)
										+ Pow(acti_r[i][j].t(), 2.0));
			}
		}
		Hiddenlayers[i].U_rgrad =
				tmp / nSamples
						+ Hiddenlayers[i].U_r
								* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].U_rd2 = tmp2 / nSamples
				+ Config::instance()->HiddenConfigs[i].get_WeightDecay();
		tmp = *delta_r[i + 1][0] * acti_r[i + 1][1].t();
		tmp2 = *delta_rd2[i + 1][0] * Pow(acti_r[i + 1][1].t(), 2.0);
		for (int j = 1; j < T - 1; j++) {
			tmp = tmp + *delta_r[i + 1][j] * acti_r[i + 1][j + 1].t();
			tmp2 = tmp2
					+ *delta_rd2[i + 1][j]
							* Pow(acti_r[i + 1][j + 1].t(), 2.0);
		}
		Hiddenlayers[i].W_rgrad =
				tmp / nSamples
						+ Hiddenlayers[i].W_r
								* Config::instance()->HiddenConfigs[i].get_WeightDecay();
		Hiddenlayers[i].W_rd2 = tmp2 / nSamples
				+ Config::instance()->HiddenConfigs[i].get_WeightDecay();
	}
}
