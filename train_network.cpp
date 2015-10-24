#include "train_network.h"
#include <iomanip>
void getNetworkCost(cuMatrixVector<double> &acti_0, cuMatrix<double> &sampleY,
		vector<HiddenConfig> &HiddenConfigs, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR) {
	int T = acti_0.size();
	int nSamples = acti_0[0]->cols;
	vector<cuMatrixVector<double> > acti_l;
	vector<cuMatrixVector<double> > acti_r;
	vector<cuMatrixVector<double> > nonlin_l;
	vector<cuMatrixVector<double> > nonlin_r;

	acti_l.push_back(acti_0);
	acti_r.push_back(acti_0);

//	getHandle();
	for (int i = 1; i <= HiddenConfigs.size(); i++) {
		//time forword
		cuMatrixVector<double> tmp_nonlinv;
		cuMatrixVector<double> tmp_acti;
		for (int j = 0; j < T; j++) {
			cuMatrix<double>* tmp1 = new cuMatrix<double>(
					Hiddenlayers[i - 1].U_l->rows, acti_l[i - 1][j]->cols, 1);
			cuMatrix<double>* tmp2 = new cuMatrix<double>(
					Hiddenlayers[i - 1].W_l->rows, acti_l[i - 1][j]->cols, 1);
			matrixMul(Hiddenlayers[i - 1].U_l, acti_l[i - 1][j], tmp1); //matrixMul(*x,*y,*z) z = x*y/
			if (j > 0) {
				matrixMul(Hiddenlayers[i - 1].W_l, tmp_acti[j - 1], tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
			if (i > 1) {
				matrixMul(Hiddenlayers[i - 1].U_l, acti_r[i - 1][j], tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
			delete tmp2;
			cuMatrix<double>* tmp_nonlin = new cuMatrix<double>(tmp1->rows,
					tmp1->cols, 1);
			checkCudaErrors(
					cudaMemcpy(tmp_nonlin->getDev(), tmp1->getDev(),
							sizeof(double) * tmp1->rows * tmp1->cols,
							cudaMemcpyDeviceToDevice));
			tmp_nonlinv.push_back(tmp_nonlin);

			ReLU(tmp1);

			if (HiddenConfigs[i - 1].get_DropoutRate() < 1.0) {
			} else {
				tmp_acti.push_back(tmp1);
			}
		}
		nonlin_l.push_back(tmp_nonlinv);
		acti_l.push_back(tmp_acti);
		tmp_nonlinv.clear();
		tmp_acti.clear();
//time backward
		cuMatrixVector<double> tmp_nonlinv2;
		cuMatrixVector<double> tmp_acti2;
		for (int x = 0; x < T; x++) {
			cuMatrix<double>* tmpmat = NULL;
			tmp_acti2.push_back(tmpmat);
			tmp_nonlinv2.push_back(tmpmat);
		}
		for (int j = T - 1; j >= 0; j--) {
			cuMatrix<double>* tmp1 = new cuMatrix<double>(
					Hiddenlayers[i - 1].U_r->rows, acti_r[i - 1][j]->cols, 1);
			cuMatrix<double>* tmp2 = new cuMatrix<double>(
					Hiddenlayers[i - 1].W_r->rows, acti_r[i - 1][j]->cols, 1);
			matrixMul(Hiddenlayers[i - 1].U_r, acti_r[i - 1][j], tmp1);
			if (j < T - 1) {
//				matrixMul(Hiddenlayers[i - 1].W_r, acti_r[i][j + 1], tmp2);
				matrixMul(Hiddenlayers[i - 1].W_r, tmp_acti2[j + 1], tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
			if (i > 1) {
				matrixMul(Hiddenlayers[i - 1].U_r, acti_l[i - 1][j], tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
			delete tmp2;
			cuMatrix<double>* tmp_nonlin = new cuMatrix<double>(tmp1->rows,
					tmp1->cols, 1);
			checkCudaErrors(
					cudaMemcpy(tmp_nonlin->getDev(), tmp1->getDev(),
							sizeof(double) * tmp1->rows * tmp1->cols,
							cudaMemcpyDeviceToDevice));
			tmp_nonlinv2[j] = tmp_nonlin;
			ReLU(tmp1);
			if (HiddenConfigs[i - 1].get_DropoutRate() < 1.0) {
			} else {
				tmp_acti2[j] = tmp1;
			}
		}
		nonlin_r.push_back(tmp_nonlinv2);
		acti_r.push_back(tmp_acti2);
		tmp_nonlinv2.clear();
		tmp_acti2.clear();
	}

//softmax forward
	cuMatrixVector<double> p;
	for (int i = 0; i < T; i++) {
		cuMatrix<double>* tmp1 = new cuMatrix<double>(SMR.W_l->rows,
				acti_l[acti_l.size() - 1][i]->cols, 1);
		cuMatrix<double>* tmp2 = new cuMatrix<double>(SMR.W_r->rows,
				acti_r[acti_r.size() - 1][i]->cols, 1);
		matrixMul(SMR.W_l, acti_l[acti_l.size() - 1][i], tmp1);
		matrixMul(SMR.W_r, acti_r[acti_r.size() - 1][i], tmp2);
		ElementAdd(tmp2, tmp1, tmp1);

		reduce_max(tmp1, tmp2);
		ElementDec(tmp1, tmp2, tmp1);
		ElementExp(tmp1, tmp1);
		reduce_sum(tmp1, tmp2);
		ElementDiv(tmp1, tmp2, tmp1);
		delete tmp2;
		p.push_back(tmp1);
	}
	cuMatrixVector<double> groundTruth;
	for (int i = 0; i < T; i++) {
		cuMatrix<double>* tmp = new cuMatrix<double>(SMR.get_NumClasses(),
				nSamples, 1);
		groundTruth.push_back(tmp);
	}
	groundTruth.toGpu();
	set_groundtruth(groundTruth, sampleY);
// cost fun
	double j1 = 0.0;
	double j2 = 0.0;
	double j3 = 0.0;
	double j4 = 0.0;
	double J1 = 0.0;
	for (int i = 0; i < T; i++) {
		cuMatrix<double>* tmp = new cuMatrix<double>(p[i]->rows, p[i]->cols, 1);
		ElementLog(p[i], tmp);
		ElementMul(groundTruth[i], tmp, tmp);

		j1 -= matrix_sum(tmp);
		delete tmp;
	}
	j1 /= nSamples;

	cuMatrix<double>* tmpj1 = new cuMatrix<double>(SMR.W_l->rows, SMR.W_l->cols,
			1);
	ElementPow(SMR.W_l, 2, tmpj1);
	j2 += matrix_sum(tmpj1);
	ElementPow(SMR.W_r, 2, tmpj1);
	j2 += matrix_sum(tmpj1);
	j2 = j2 * SMR.get_WeightDecay() / 2;
	delete tmpj1;

	for (int i = 0; i < Hiddenlayers.size(); i++) {
		cuMatrix<double>* tmp = new cuMatrix<double>(Hiddenlayers[i].W_l->rows,
				Hiddenlayers[i].W_l->cols, 1);
		ElementPow(Hiddenlayers[i].W_l, 2, tmp);

		j3 += matrix_sum(tmp);
		ElementPow(Hiddenlayers[i].W_r, 2, tmp);

		j3 += matrix_sum(tmp);
		j3 = j3 * HiddenConfigs[i].get_WeightDecay() / 2;
		delete tmp;
	}
	for (int i = 0; i < Hiddenlayers.size(); i++) {
		cuMatrix<double>* tmp = new cuMatrix<double>(Hiddenlayers[i].U_l->rows,
				Hiddenlayers[i].U_l->cols, 1);
		ElementPow(Hiddenlayers[i].U_l, 2, tmp);

		j4 += matrix_sum(tmp);
		ElementPow(Hiddenlayers[i].U_r, 2, tmp);

		j4 += matrix_sum(tmp);
		j4 = j4 * HiddenConfigs[i].get_WeightDecay() / 2;
		delete tmp;
	}
	SMR.cost = j1 + j2 + j3 + j4;
	printf("j1 = %f,j2 = %f,j3 = %f,j4 = %f,smr.cost = %f\n", j1, j2, j3, j4,
			SMR.cost);
	cuMatrixVector<double> dis;
	cuMatrixVector<double> dis2;
	for (int i = 0; i < T; i++) {
		cuMatrix<double>* tmp = new cuMatrix<double>(groundTruth[0]->rows,
				groundTruth[0]->cols, 1);
		ElementDec(groundTruth[i], p[i], tmp);
		dis.push_back(tmp);
	}
//	dis.toGpu();
	for (int i = 0; i < T; i++) {
		cuMatrix<double>* tmp = new cuMatrix<double>(groundTruth[0]->rows,
				groundTruth[0]->cols, 1);
		ElementPow(dis[i], 2, tmp);
		dis2.push_back(tmp);
	}

//	dis2.toGpu();

//SMR bp
//SMR.W_lgrad
	cuMatrix<double>* tmp3 = new cuMatrix<double>(
			acti_l[acti_l.size() - 1][0]->rows,
			acti_l[acti_l.size() - 1][0]->cols, 1);
	cuMatrix<double>* tmp2 = new cuMatrix<double>(groundTruth[0]->rows,
			acti_l[acti_l.size() - 1][0]->rows, 1);
	cuMatrix<double>* tmp1 = new cuMatrix<double>(SMR.W_l->rows, SMR.W_l->cols,
			1);
	cuMatrix<double>* tmp = new cuMatrix<double>(groundTruth[0]->rows,
			acti_l[acti_l.size() - 1][0]->rows, 1);
	tmp->toGpu();
	tmp1->toGpu();
	for (int i = 0; i < T; i++) {
		/*z = x * T(y)*/
		matrixMulTB(dis[i], acti_l[acti_l.size() - 1][i], tmp2);
		ElementDec(tmp, tmp2, tmp);
	}
	ElementDiv(tmp, nSamples, tmp);
	ElementMul(SMR.W_l, SMR.get_WeightDecay(), tmp1);
	ElementAdd(tmp, tmp1, SMR.W_lgrad);
	delete tmp2;
	delete tmp;
	delete tmp1;
//smr.W_ld2
	tmp2 = new cuMatrix<double>(groundTruth[0]->rows,
			acti_l[acti_l.size() - 1][0]->rows, 1);
	tmp = new cuMatrix<double>(groundTruth[0]->rows,
			acti_l[acti_l.size() - 1][0]->rows, 1);
	for (int i = 0; i < T; i++) {
		ElementPow(acti_l[acti_l.size() - 1][i], 2, tmp3);
		matrixMulTB(dis2[i], tmp3, tmp2);
		ElementAdd(tmp, tmp2, tmp);
	}
	ElementDiv(tmp, nSamples, tmp);
	ElementAdd(tmp, SMR.get_WeightDecay(), SMR.W_ld2);

	delete tmp2;
	delete tmp3;
	delete tmp;

//SMR.W_rgrad
	tmp1 = new cuMatrix<double>(groundTruth[0]->rows, groundTruth[0]->cols, 1);
	tmp2 = new cuMatrix<double>(groundTruth[0]->rows,
			acti_r[acti_r.size() - 1][0]->rows, 1);
	tmp = new cuMatrix<double>(groundTruth[0]->rows,
			acti_r[acti_r.size() - 1][0]->rows, 1);
	for (int i = 0; i < T; i++) {
		/*z = x * T(y)*/
		matrixMulTB(dis[i], acti_r[acti_r.size() - 1][i], tmp2);
		ElementDec(tmp, tmp2, tmp);
	}
	delete tmp1;
	tmp1 = new cuMatrix<double>(SMR.W_r->rows, SMR.W_r->cols, 1);
	ElementDiv(tmp, nSamples, tmp);
	ElementMul(SMR.W_r, SMR.get_WeightDecay(), tmp1);
	ElementAdd(tmp, tmp1, SMR.W_rgrad);
	delete tmp;
	delete tmp1;
	delete tmp2;
	tmp2 = new cuMatrix<double>(groundTruth[0]->rows,
			acti_r[acti_r.size() - 1][0]->rows, 1);
	tmp3 = new cuMatrix<double>(acti_r[acti_r.size() - 1][0]->rows,
			acti_r[acti_r.size() - 1][0]->cols, 1);
	tmp = new cuMatrix<double>(groundTruth[0]->rows,
			acti_r[acti_r.size() - 1][0]->rows, 1);
	for (int i = 0; i < T; i++) {
		ElementPow(acti_r[acti_r.size() - 1][i], 2, tmp3);
		matrixMulTB(dis2[i], tmp3, tmp2);
		ElementAdd(tmp, tmp2, tmp);
	}
	ElementDiv(tmp, nSamples, tmp);
	ElementAdd(tmp, SMR.get_WeightDecay(), SMR.W_rd2);
	delete tmp2;
	delete tmp3;
	delete tmp;
//BPTT for last hidden

	vector<cuMatrixVector<double> > delta_l(acti_l.size());
	vector<cuMatrixVector<double> > delta_ld2(acti_l.size());
	vector<cuMatrixVector<double> > delta_r(acti_r.size());
	vector<cuMatrixVector<double> > delta_rd2(acti_r.size());
	for (int i = 0; i < delta_l.size(); i++) {
		cuMatrix<double> *tmpmat = NULL;
		for (int j = 0; j < T; j++) {
			delta_l[i].push_back(tmpmat);
			delta_ld2[i].push_back(tmpmat);
			delta_r[i].push_back(tmpmat);
			delta_rd2[i].push_back(tmpmat);
		}
		delete tmpmat;
	}

//time forward
	tmp2 = new cuMatrix<double>(SMR.W_l->cols, dis[0]->cols, 1);
	tmp3 = new cuMatrix<double>(Hiddenlayers[Hiddenlayers.size() - 1].W_l->rows,
			Hiddenlayers[Hiddenlayers.size() - 1].W_l->cols, 1); //hiddenlayer.W_l^2
	cuMatrix<double>* tmp4 = new cuMatrix<double>(SMR.W_l->rows, SMR.W_l->cols,
			1); //SMR.W_l^2
	ElementPow(SMR.W_l, 2, tmp4);
	ElementPow(Hiddenlayers[Hiddenlayers.size() - 1].W_l, 2, tmp3);
	for (int i = T - 1; i >= 0; i--) {
		tmp = new cuMatrix<double>(SMR.W_l->cols, dis[0]->cols, 1);
		tmp1 = new cuMatrix<double>(tmp->rows, tmp->cols, 1);
		matrixMulTA(SMR.W_l, dis[i], tmp);
		ElementMul(tmp, -1.0, tmp);
		matrixMulTA(tmp4, dis2[i], tmp1);
		if (i < T - 1) {
			matrixMulTA(Hiddenlayers[Hiddenlayers.size() - 1].W_l,
					delta_l[delta_l.size() - 1][i + 1], tmp2);
			ElementAdd(tmp2, tmp, tmp);
			matrixMulTA(tmp3, delta_ld2[delta_ld2.size() - 1][i + 1], tmp2);
			ElementAdd(tmp2, tmp1, tmp1);
		}
		delta_l[delta_l.size() - 1][i] = tmp;
		delta_ld2[delta_ld2.size() - 1][i] = tmp1;
		cuMatrix<double>* tmpcm = dReLU(nonlin_l[nonlin_l.size() - 1][i]);
		ElementMul(delta_l[delta_l.size() - 1][i], tmpcm,
				delta_l[delta_l.size() - 1][i]);

		ElementPow(tmpcm, 2.0, tmpcm);
		ElementMul(delta_ld2[delta_ld2.size() - 1][i], tmpcm,
				delta_ld2[delta_ld2.size() - 1][i]);

		delete tmpcm;
		if (HiddenConfigs[HiddenConfigs.size() - 1].get_WeightDecay() < 1.0) {

		}
	}
	delete tmp2;
	delete tmp3;
	delete tmp4;
//time backward
	tmp2 = new cuMatrix<double>(SMR.W_r->cols, dis[0]->cols, 1);

	tmp3 = new cuMatrix<double>(Hiddenlayers[Hiddenlayers.size() - 1].W_r->rows,
			Hiddenlayers[Hiddenlayers.size() - 1].W_r->cols, 1); //hiddenlayer.W_r^2
	tmp4 = new cuMatrix<double>(SMR.W_r->rows, SMR.W_r->cols, 1); //SMR.W_r^2
	ElementPow(SMR.W_r, 2, tmp4);
	ElementPow(Hiddenlayers[Hiddenlayers.size() - 1].W_r, 2, tmp3);
	for (int i = 0; i < T; i++) {
		tmp = new cuMatrix<double>(SMR.W_r->cols, dis[0]->cols, 1);
		tmp1 = new cuMatrix<double>(tmp->rows, tmp->cols, 1);
		matrixMulTA(SMR.W_r, dis[i], tmp);
		ElementMul(tmp, -1.0, tmp);
		matrixMulTA(tmp4, dis2[i], tmp1);
		if (i > 0) {
			matrixMulTA(Hiddenlayers[Hiddenlayers.size() - 1].W_r,
					delta_r[delta_r.size() - 1][i - 1], tmp2);
			ElementAdd(tmp2, tmp, tmp);
			matrixMulTA(tmp3, delta_rd2[delta_rd2.size() - 1][i - 1], tmp2);
			ElementAdd(tmp2, tmp1, tmp1);
		}
		delta_r[delta_r.size() - 1][i] = tmp;
		delta_rd2[delta_rd2.size() - 1][i] = tmp1;
		cuMatrix<double>* tmpcm = dReLU(nonlin_r[nonlin_r.size() - 1][i]);
		ElementMul(delta_r[delta_r.size() - 1][i], tmpcm,
				delta_r[delta_r.size() - 1][i]);

		ElementPow(tmpcm, 2.0, tmpcm);
		ElementMul(delta_rd2[delta_rd2.size() - 1][i], tmpcm,
				delta_rd2[delta_rd2.size() - 1][i]);
//		cout << "delta_r[delta_r.size() - 1][i]:"<<matrix_sum(delta_r[delta_r.size() - 1][i])<<endl;
		delete tmpcm;
		if (HiddenConfigs[HiddenConfigs.size() - 1].get_WeightDecay() < 1.0) {

		}
	}
	delete tmp2;
	delete tmp3;
	delete tmp4;

//hiddenlayers
	for (int i = delta_l.size() - 2; i > 0; --i) {
		tmp3 = new cuMatrix<double>(Hiddenlayers[i].W_l->rows,
				Hiddenlayers[i].W_l->cols, 1); //Hiddenlayers[i].Wl ^2
		tmp4 = new cuMatrix<double>(Hiddenlayers[i].U_l->rows,
				Hiddenlayers[i].U_l->cols, 1); //Hiddenlayers[i].Ul ^2
		cuMatrix<double>* tmp5 = new cuMatrix<double>(Hiddenlayers[i].W_r->rows,
				Hiddenlayers[i].W_r->cols, 1); //Hiddenlayers[i].Wr^2
		cuMatrix<double>* tmp6 = new cuMatrix<double>(Hiddenlayers[i].U_r->rows,
				Hiddenlayers[i].U_r->cols, 1); //Hiddenlayers[i].Ur ^2
		ElementPow(Hiddenlayers[i - 1].W_l, 2, tmp3);
		ElementPow(Hiddenlayers[i - 1].U_l, 2, tmp4);
		ElementPow(Hiddenlayers[i - 1].W_r, 2, tmp5);
		ElementPow(Hiddenlayers[i - 1].U_r, 2, tmp6);
		//forward
		for (int j = T - 1; j >= 0; j--) {
			delta_l[i][j] = new cuMatrix<double>(Hiddenlayers[i].U_l->cols,
					delta_l[i + 1][j]->cols, 1);
			delta_ld2[i][j] = new cuMatrix<double>(Hiddenlayers[i].U_l->cols,
					delta_ld2[i + 1][j]->cols, 1);
			tmp2 = new cuMatrix<double>(Hiddenlayers[i].U_l->cols,
					delta_ld2[i + 1][j]->cols, 1);

			matrixMulTA(Hiddenlayers[i].U_l, delta_l[i + 1][j], delta_l[i][j]);
			matrixMulTA(tmp4, delta_ld2[i + 1][j], delta_ld2[i][j]);
			if (j < T - 1) {
				matrixMulTA(Hiddenlayers[i - 1].W_l, delta_l[i][j + 1], tmp2);
				ElementAdd(delta_l[i][j], tmp2, delta_l[i][j]);
				matrixMulTA(tmp3, delta_ld2[i][j + 1], tmp2);
				ElementAdd(delta_ld2[i][j], tmp2, delta_ld2[i][j]);
			}
			matrixMulTA(Hiddenlayers[i].U_r, delta_r[i + 1][j], tmp2);
			ElementAdd(delta_l[i][j], tmp2, delta_l[i][j]);
			matrixMulTA(tmp6, delta_rd2[i + 1][j], tmp2);
			ElementAdd(delta_ld2[i][j], tmp2, delta_ld2[i][j]);
			cuMatrix<double>* tmpcm = dReLU(nonlin_l[i - 1][j]);
			ElementMul(tmpcm, delta_l[i][j], delta_l[i][j]);
			ElementPow(tmpcm, 2, tmpcm);
			ElementMul(delta_ld2[i][j], tmpcm, delta_ld2[i][j]);
			if (HiddenConfigs[HiddenConfigs.size() - 1].get_WeightDecay()
					< 1.0) {

			}
			delete tmpcm;
			delete tmp2;
		}
		//backward
		for (int j = 0; j < T; ++j) {
			delta_l[i][j] = new cuMatrix<double>(Hiddenlayers[i].U_r->cols,
					delta_l[i + 1][j]->cols, 1);
			delta_ld2[i][j] = new cuMatrix<double>(Hiddenlayers[i].U_r->cols,
					delta_ld2[i + 1][j]->cols, 1);
			tmp2 = new cuMatrix<double>(Hiddenlayers[i].U_r->cols,
					delta_ld2[i + 1][j]->cols, 1);
			matrixMulTA(Hiddenlayers[i].U_r, delta_r[i + 1][j], delta_l[i][j]);
			matrixMulTA(tmp6, delta_rd2[i + 1][j], delta_ld2[i][j]);
			if (j > 0) {
				matrixMulTA(Hiddenlayers[i - 1].W_r, delta_r[i][j - 1], tmp2);
				ElementAdd(delta_l[i][j], tmp2, delta_l[i][j]);
				matrixMulTA(tmp5, delta_rd2[i][j - 1], tmp2);
				ElementAdd(delta_ld2[i][j], tmp2, delta_ld2[i][j]);
			}
			matrixMulTA(Hiddenlayers[i].U_l, delta_l[i + 1][j], tmp2);
			ElementAdd(delta_l[i][j], tmp2, delta_l[i][j]);
			matrixMulTA(tmp4, delta_ld2[i + 1][j], tmp2);
			ElementAdd(delta_ld2[i][j], tmp2, delta_ld2[i][j]);
//			delta_l[i][j] = tmp;
//			delta_ld2[i][j] = tmp1;
			cuMatrix<double>* tmpcm = dReLU(dReLU(nonlin_r[i - 1][j]));
			ElementMul(tmpcm, delta_r[i][j], delta_r[i][j]);
			ElementPow(tmpcm, 2, tmpcm);
			ElementMul(delta_rd2[i][j], tmpcm, delta_rd2[i][j]);
			if (HiddenConfigs[HiddenConfigs.size() - 1].get_WeightDecay()
					< 1.0) {

			}
			delete tmpcm;
			delete tmp2;
		}
		delete tmp3;
		delete tmp4;
		delete tmp5;
		delete tmp6;
	}

	for (int i = HiddenConfigs.size() - 1; i >= 0; i--) {

		tmp = new cuMatrix<double>(delta_l[i + 1][0]->rows, acti_l[i][0]->rows,
				1);
		tmp1 = new cuMatrix<double>(delta_ld2[i + 1][0]->rows,
				acti_l[i][0]->rows, 1);
		tmp2 = new cuMatrix<double>(delta_ld2[i + 1][0]->rows,
				acti_l[i][0]->rows, 1);
		tmp3 = new cuMatrix<double>(delta_ld2[i + 1][0]->rows,
				acti_l[i][0]->rows, 1);
		cuMatrix<double>* acti_l2 = new cuMatrix<double>(acti_l[i][0]->rows,
				acti_l[i][0]->cols, 1);
		cuMatrix<double>* acti_r2 = new cuMatrix<double>(acti_r[i][0]->rows,
				acti_r[i][0]->cols, 1);
		// bacward

		if (i == 0) {
			matrixMulTB(delta_l[i + 1][0], acti_l[i][0], tmp);
			ElementPow(acti_l[i][0], 2.0, acti_l2);
			matrixMulTB(delta_ld2[i + 1][0], acti_l2, tmp1);
			for (int j = 1; j < T; j++) {
				matrixMulTB(delta_l[i + 1][j], acti_l[i][j], tmp2);
				ElementAdd(tmp, tmp2, tmp);

				ElementPow(acti_l[i][j], 2.0, acti_l2);
				matrixMulTB(delta_ld2[i + 1][j], acti_l2, tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
		} else {
			ElementAdd(acti_l[i][0], acti_r[i][0], tmp2);
			matrixMulTB(delta_l[i + 1][0], tmp2, tmp);

			ElementPow(acti_l[i][0], 2.0, acti_l2);
			ElementPow(acti_r[i][0], 2.0, acti_r2);
			ElementAdd(acti_l2, acti_r2, acti_l2);
			matrixMulTB(delta_ld2[i + 1][0], acti_l2, tmp1);
			for (int j = 1; j < T; j++) {
				ElementAdd(acti_l[i][j], acti_r[i][j], tmp2);
				matrixMulTB(delta_l[i + 1][j], tmp2, tmp3);
				ElementAdd(tmp, tmp3, tmp);

				ElementPow(acti_l[i][j], 2.0, acti_l2);
				ElementPow(acti_r[i][j], 2.0, acti_r2);
				ElementAdd(acti_l2, acti_r2, acti_l2);
				matrixMulTB(delta_ld2[i + 1][j], acti_l2, tmp3);
				ElementAdd(tmp1, tmp3, tmp1);
			}
		}

		//		freeboth(acti_r2);
		ElementDiv(tmp, (double) nSamples, tmp);

		ElementMul(Hiddenlayers[i].U_l, HiddenConfigs[i].get_WeightDecay(),
				tmp2);
		ElementAdd(tmp, tmp2, Hiddenlayers[i].U_lgrad);
		ElementDiv(tmp1, (double) nSamples, tmp1);
		ElementAdd(tmp1, HiddenConfigs[i].get_WeightDecay(),
				Hiddenlayers[i].U_ld2);
		delete acti_l2;
		delete acti_r2;
		delete tmp;
		delete tmp1;
		delete tmp2;
		delete tmp3;
		acti_l2 = new cuMatrix<double>(acti_l[i + 1][T - 2]->rows,
				acti_l[i + 1][T - 2]->cols, 1);
		tmp = new cuMatrix<double>(delta_l[i + 1][T - 1]->rows,
				acti_l[i + 1][T - 2]->rows, 1);
		tmp1 = new cuMatrix<double>(delta_ld2[i + 1][T - 1]->rows,
				acti_l2->rows, 1);
		tmp2 = new cuMatrix<double>(tmp->rows, tmp->cols, 1);
		tmp3 = new cuMatrix<double>(tmp1->rows, tmp1->cols, 1);
		matrixMulTB(delta_l[i + 1][T - 1], acti_l[i + 1][T - 2], tmp);
		ElementPow(acti_l[i + 1][T - 2], 2.0, acti_l2);

		matrixMulTB(delta_ld2[i + 1][T - 1], acti_l2, tmp1);
		for (int j = T - 2; j > 0; j--) {
			matrixMulTB(delta_l[i + 1][j], acti_l[i + 1][j - 1], tmp2);
			ElementAdd(tmp, tmp2, tmp);
			ElementPow(acti_l[i + 1][j - 1], 2.0, acti_l2);
			matrixMulTB(delta_ld2[i + 1][j], acti_l2, tmp3);
			ElementAdd(tmp1, tmp3, tmp1);
		}

		ElementDiv(tmp, (double) nSamples, tmp);
		ElementMul(Hiddenlayers[i].W_l,
				(double) HiddenConfigs[i].get_WeightDecay(), tmp2);

		ElementAdd(tmp, tmp2, Hiddenlayers[i].W_lgrad);
		ElementDiv(tmp1, (double) nSamples, tmp1);
		ElementAdd(tmp1, (double) HiddenConfigs[i].get_WeightDecay(),
				Hiddenlayers[i].W_ld2);
		delete tmp;
		delete tmp1;
		delete tmp2;
		delete tmp3;
		delete acti_l2;
//		 backward

		acti_r2 = new cuMatrix<double>(acti_r[i][0]->rows, acti_r[i][0]->cols,
				1);
		acti_l2 = new cuMatrix<double>(acti_l[i][0]->rows, acti_l[i][0]->cols,
				1);
		tmp = new cuMatrix<double>(delta_r[i + 1][0]->rows, acti_l[i][0]->rows,
				1);
		tmp1 = new cuMatrix<double>(delta_rd2[i + 1][0]->rows,
				acti_r[i][0]->rows, 1);
		tmp2 = new cuMatrix<double>(delta_rd2[i + 1][0]->rows,
				acti_r[i][0]->rows, 1);
		tmp3 = new cuMatrix<double>(delta_rd2[i + 1][0]->rows,
				acti_r[i][0]->rows, 1);

		if (i == 0) {
			matrixMulTB(delta_r[i + 1][0], acti_r[i][0], tmp);
			ElementPow(acti_r[i][0], 2.0, acti_r2);
			matrixMulTB(delta_rd2[i + 1][0], acti_r2, tmp1);
			for (int j = 1; j < T; j++) {
				matrixMulTB(delta_r[i + 1][j], acti_r[i][j], tmp2);
				ElementAdd(tmp, tmp2, tmp);

				ElementPow(acti_r[i][j], 2.0, acti_r2);
				matrixMulTB(delta_rd2[i + 1][j], acti_r2, tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
		} else {

			ElementAdd(acti_l[i][0], acti_r[i][0], tmp2);
			matrixMulTB(delta_r[i + 1][0], tmp2, tmp);

			ElementPow(acti_l[i][0], 2.0, acti_l2);
			ElementPow(acti_r[i][0], 2.0, acti_r2);
			ElementAdd(acti_l2, acti_r2, acti_l2);
			matrixMulTB(delta_rd2[i + 1][0], acti_l2, tmp1);
			for (int j = 1; j < T; j++) {
				ElementAdd(acti_l[i][j], acti_r[i][j], tmp2);
				matrixMulTB(delta_r[i + 1][j], tmp2, tmp3);
				ElementAdd(tmp, tmp3, tmp);
				ElementPow(acti_l[i][j], 2.0, acti_l2);
				ElementPow(acti_r[i][j], 2.0, acti_r2);
				ElementAdd(acti_l2, acti_r2, acti_l2);
				matrixMulTB(delta_rd2[i + 1][j], acti_l2, tmp3);
				ElementAdd(tmp1, tmp3, tmp1);
			}
		}

		ElementDiv(tmp, (double) nSamples, tmp);
		ElementMul(Hiddenlayers[i].U_r, HiddenConfigs[i].get_WeightDecay(),
				tmp2);
		ElementAdd(tmp, tmp2, Hiddenlayers[i].U_rgrad);
		ElementDiv(tmp1, (double) nSamples, tmp1);
		ElementAdd(tmp1, HiddenConfigs[i].get_WeightDecay(),
				Hiddenlayers[i].U_rd2);

		delete tmp;
		delete tmp1;
		delete tmp2;
		delete tmp3;
		delete acti_r2;
		delete acti_l2;

		acti_r2 = new cuMatrix<double>(acti_r[i + 1][1]->rows,
				acti_r[i + 1][1]->cols, 1);
		tmp = new cuMatrix<double>(delta_l[i + 1][0]->rows,
				acti_r[i + 1][1]->rows, 1);
		tmp1 = new cuMatrix<double>(delta_rd2[i + 1][0]->rows, acti_r2->rows,
				1);
		tmp2 = new cuMatrix<double>(tmp->rows, tmp->cols, 1);
		tmp3 = new cuMatrix<double>(tmp1->rows, tmp1->cols, 1);

		matrixMulTB(delta_r[i + 1][0], acti_r[i + 1][1], tmp);
		ElementPow(acti_r[i + 1][1], 2.0, acti_r2);
		matrixMulTB(delta_rd2[i + 1][0], acti_r2, tmp1);
		for (int j = 1; j < T - 1; j++) {
			matrixMulTB(delta_r[i + 1][j], acti_r[i + 1][j + 1], tmp2);
			ElementAdd(tmp, tmp2, tmp);
			ElementPow(acti_r[i + 1][j + 1], 2.0, acti_r2);
			matrixMulTB(delta_rd2[i + 1][j], acti_r2, tmp3);
			ElementAdd(tmp1, tmp3, tmp1);
		}
		ElementDiv(tmp, (double) nSamples, tmp);
		ElementMul(Hiddenlayers[i].W_r,
				(double) HiddenConfigs[i].get_WeightDecay(), tmp2);
		ElementAdd(tmp, tmp2, Hiddenlayers[i].W_rgrad);
		ElementDiv(tmp1, (double) nSamples, tmp1);
		ElementAdd(tmp1, (double) HiddenConfigs[i].get_WeightDecay(),
				Hiddenlayers[i].W_rd2);

		delete acti_r2;
		delete tmp;
		delete tmp1;
		delete tmp2;
		delete tmp3;

	}

	for (int i = 0; i < dis2.size(); i++) {
		delete dis2[i];
		delete dis[i];
	}
	dis2.clear();
	dis.clear();
	for (int x = 1; x < acti_l.size(); x++) {
		for (int y = 0; y < acti_l[x].size(); y++) {
			delete acti_l[x][y];
			delete acti_r[x][y];
		}
		acti_l[x].clear();
		acti_r[x].clear();
	}

	for (int x = 0; x < nonlin_l.size(); x++) {
		for (int y = 0; y < nonlin_l[x].size(); y++) {
			delete nonlin_l[x][y];
			delete nonlin_r[x][y];
		}
		nonlin_l[x].clear();
		nonlin_r[x].clear();
	}

	for (int x = delta_l.size() -1 ; x >= 0; x--) {
		for (int y = 0; y < delta_l[x].size(); y++) {
			if (delta_l[x][y] != NULL) {
				delete delta_l[x][y];
			}

			if (delta_ld2[x][y] != NULL) {
				delete delta_ld2[x][y];
			}
			if (delta_r[x][y] != NULL) {
				delete delta_r[x][y];
			}
			if (delta_rd2[x][y] != NULL) {
				delete delta_rd2[x][y];
			}
		}
		delta_l[x].clear();
		delta_ld2[x].clear();
		delta_r[x].clear();
		delta_rd2[x].clear();
	}

	for (int x = 0; x < p.size(); x++) {
		delete p[x];
	}
	p.clear();
	for (int x = 0; x < groundTruth.size(); x++) {
		delete groundTruth[x];
	}
	groundTruth.clear();
}

void trainNetwork(vector<vector<int> > &trainX, vector<vector<int> > &trainY,
		vector<HiddenConfig> &HiddenConfigs, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR, vector<vector<int> > &testX,
		vector<vector<int> > &testY, vector<string> &re_word) {
	cuMatrix<double> v_smr_W_l(SMR.W_l->rows, SMR.W_l->cols, 1);
	cuMatrix<double> smrW_ld2(SMR.W_l->rows, SMR.W_l->cols, 1);     //r
	cuMatrix<double> v_smr_W_r(SMR.W_l->rows, SMR.W_l->cols, 1); //xxxxxxxxxxxxxxxxxxxxx
	cuMatrix<double> smrW_rd2(SMR.W_l->rows, SMR.W_l->cols, 1);
	cuMatrixVector<double> v_hl_W_l;
	cuMatrixVector<double> hlW_ld2;
	cuMatrixVector<double> v_hl_U_l;
	cuMatrixVector<double> hlU_ld2;
	cuMatrixVector<double> v_hl_W_r;
	cuMatrixVector<double> hlW_rd2;
	cuMatrixVector<double> v_hl_U_r;
	cuMatrixVector<double> hlU_rd2;
	for (int i = 0; i < Hiddenlayers.size(); i++) {
		cuMatrix<double>* tmpW = new cuMatrix<double>(Hiddenlayers[i].W_l->rows,
				Hiddenlayers[i].W_l->cols, 1);
		cuMatrix<double>* tmpU = new cuMatrix<double>(Hiddenlayers[i].U_l->rows,
				Hiddenlayers[i].U_l->cols, 1);
		v_hl_W_l.push_back(tmpW);
		v_hl_U_l.push_back(tmpU);
		hlW_ld2.push_back(tmpW);
		hlU_ld2.push_back(tmpU);
		v_hl_W_r.push_back(tmpW);
		v_hl_U_r.push_back(tmpU);
		hlW_rd2.push_back(tmpW);
		hlU_rd2.push_back(tmpU);
	}
	double Momentum_w = 0.5;
	double Momentum_u = 0.5;
	double Momentum_d2 = 0.5;
	double mu = 1e-2;
	int k = 0;

	for (int epo = 1; epo <= Config::instance()->get_training_epochs(); epo++) {
		for (; k <= Config::instance()->get_iter_per_epo() * epo; k++) {
//			for (int epo = 1; epo <= 10; epo++) {
//				for (; k <= 5; k++) {
			if (k > 30) {
				Momentum_w = 0.95;
				Momentum_u = 0.95;
				Momentum_d2 = 0.90;
			}
			cout << "epoch: " << epo << ", iter: " << k << endl;
			cuMatrixVector<double> acti_0;
			cuMatrix<double> sampleY(Config::instance()->get_ngram(),
					Config::instance()->get_batch_size(), 1);
			init_acti(acti_0, trainX, sampleY, trainY, re_word.size());
			getNetworkCost(acti_0, sampleY, HiddenConfigs, Hiddenlayers, SMR);

			for (int x = 0; x < acti_0.size(); x++) {
				delete acti_0[x];
			}
			acti_0.clear();
			freeboth(&sampleY);
			ElementMul(&smrW_ld2, Momentum_d2, &smrW_ld2);
			cuMatrix<double>* tmp = new cuMatrix<double>(smrW_ld2.rows,
					smrW_ld2.cols, 1);

			ElementMul(SMR.W_ld2, 1 - Momentum_d2, tmp);
			ElementAdd(&smrW_ld2, tmp, &smrW_ld2);
			ElementAdd(&smrW_ld2, mu, tmp);
			ElementDiv(SMR.lr_W, tmp, tmp);
			ElementMul(SMR.W_lgrad, tmp, tmp);
			ElementMul(tmp, 1.0 - Momentum_w, tmp);
			ElementMul(&v_smr_W_l, Momentum_w, &v_smr_W_l);
			ElementAdd(&v_smr_W_l, tmp, &v_smr_W_l);
			ElementDec(SMR.W_l, &v_smr_W_l, SMR.W_l);

			ElementMul(&smrW_rd2, Momentum_d2, &smrW_rd2);
			ElementMul(SMR.W_rd2, 1 - Momentum_d2, tmp);
			ElementAdd(&smrW_rd2, tmp, &smrW_rd2);
			ElementAdd(&smrW_rd2, mu, tmp);
			ElementDiv(SMR.lr_W, tmp, tmp);
			ElementMul(SMR.W_rgrad, tmp, tmp);
			ElementMul(tmp, 1.0 - Momentum_w, tmp);
			ElementMul(&v_smr_W_r, Momentum_w, &v_smr_W_r);
			ElementAdd(&v_smr_W_r, tmp, &v_smr_W_r);
			ElementDec(SMR.W_r, &v_smr_W_r, SMR.W_r);
			delete tmp;

//HiddenLayer Update
			for (int i = 0; i < Hiddenlayers.size(); i++) {
				cuMatrix<double>* hw = new cuMatrix<double>(
						Hiddenlayers[i].W_ld2->rows,
						Hiddenlayers[i].W_ld2->cols, 1);
				cuMatrix<double>* hu = new cuMatrix<double>(
						Hiddenlayers[i].U_ld2->rows,
						Hiddenlayers[i].U_ld2->cols, 1);
				ElementMul(hlW_ld2[i], Momentum_d2, hlW_ld2[i]);
				ElementMul(Hiddenlayers[i].W_ld2, 1.0 - Momentum_d2, hw);
				ElementAdd(hlW_ld2[i], hw, hlW_ld2[i]);
				ElementMul(hlU_ld2[i], Momentum_d2, hlU_ld2[i]);
				ElementMul(Hiddenlayers[i].U_ld2, 1.0 - Momentum_d2, hu);
				ElementAdd(hlU_ld2[i], hu, hlU_ld2[i]);
				ElementAdd(hlW_ld2[i], mu, hw);
				ElementAdd(hlU_ld2[i], mu, hu);
				ElementDiv(Hiddenlayers[i].lr_W, hw, hw);
				ElementDiv(Hiddenlayers[i].lr_U, hu, hu);

				ElementMul(Hiddenlayers[i].W_lgrad, hw, hw);
				ElementMul(Hiddenlayers[i].U_lgrad, hu, hu);
				ElementMul(hw, 1.0 - Momentum_w, hw);
				ElementMul(hu, 1.0 - Momentum_u, hu);
				ElementMul(v_hl_W_l[i], Momentum_w, v_hl_W_l[i]);
				ElementMul(v_hl_U_l[i], Momentum_u, v_hl_U_l[i]);
				ElementAdd(v_hl_W_l[i], hw, v_hl_W_l[i]);
				ElementAdd(v_hl_U_l[i], hu, v_hl_U_l[i]);
				ElementDec(Hiddenlayers[i].W_l, v_hl_W_l[i],
						Hiddenlayers[i].W_l);
				ElementDec(Hiddenlayers[i].U_l, v_hl_U_l[i],
						Hiddenlayers[i].U_l);
				delete hw;
				delete hu;

				hw = new cuMatrix<double>(Hiddenlayers[i].W_ld2->rows,
						Hiddenlayers[i].W_ld2->cols, 1);
				hu = new cuMatrix<double>(Hiddenlayers[i].U_ld2->rows,
						Hiddenlayers[i].U_ld2->cols, 1);
				ElementMul(hlW_rd2[i], Momentum_d2, hlW_rd2[i]);
				ElementMul(Hiddenlayers[i].W_rd2, 1.0 - Momentum_d2, hw);
				ElementAdd(hlW_rd2[i], hw, hlW_rd2[i]);
				ElementMul(hlU_rd2[i], Momentum_d2, hlU_rd2[i]);
				ElementMul(Hiddenlayers[i].U_rd2, 1.0 - Momentum_d2, hu);
				ElementAdd(hlU_rd2[i], hu, hlU_rd2[i]);

				ElementAdd(hlW_rd2[i], mu, hw);
				ElementAdd(hlU_rd2[i], mu, hu);
				ElementDiv(Hiddenlayers[i].lr_W, hw, hw);
				ElementDiv(Hiddenlayers[i].lr_U, hu, hu);
				ElementMul(Hiddenlayers[i].W_rgrad, hw, hw);
				ElementMul(Hiddenlayers[i].U_rgrad, hu, hu);
				ElementMul(hw, 1.0 - Momentum_w, hw);
				ElementMul(hu, 1.0 - Momentum_u, hu);
				ElementMul(v_hl_W_r[i], Momentum_w, v_hl_W_r[i]);
				ElementMul(v_hl_U_r[i], Momentum_u, v_hl_U_r[i]);
				ElementAdd(v_hl_W_r[i], hw, v_hl_W_r[i]);
				ElementAdd(v_hl_U_r[i], hu, v_hl_U_r[i]);
				ElementDec(Hiddenlayers[i].W_r, v_hl_W_r[i],
						Hiddenlayers[i].W_r);
				ElementDec(Hiddenlayers[i].U_r, v_hl_U_r[i],
						Hiddenlayers[i].U_r);
				delete hw;
				delete hu;
			}
		}
	}
	freeboth(&v_smr_W_l);
	freeboth(&smrW_ld2);
	freeboth(&v_smr_W_r);
	freeboth(&smrW_rd2);
	for (int i = 0; i < Hiddenlayers.size(); i++) {
		freeboth(v_hl_W_l[i]);
		freeboth(v_hl_U_l[i]);
		freeboth(hlW_ld2[i]);
		freeboth(hlU_ld2[i]);
		freeboth(v_hl_W_r[i]);
		freeboth(v_hl_U_r[i]);
		freeboth(hlW_rd2[i]);
		freeboth(hlU_rd2[i]);
	}
	init_testdata(testX,testY);

	if (!(Config::instance()->get_gradient())) {
		cout << "Test training data: " << endl;
		testNetwork(Hiddenlayers,SMR,HiddenConfigs,re_word.size(),trainX.size());
		cout << "Test test data: " << endl;
		testNetwork(Hiddenlayers,SMR,HiddenConfigs,re_word.size(),testX.size());

	}

	printf("end\n");
}

