#include "resultpredict.h"

void predict(cuMatrixVector<double> &sampleX, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR, vector<HiddenConfig> &HiddenConfigs, int* output,
		int offset) {
	int T = sampleX.size();
	int mid = (int) (T / 2.0);
	vector<cuMatrixVector<double> > acti_l;
	vector<cuMatrixVector<double> > acti_r;
	acti_l.push_back(sampleX);
	acti_r.push_back(sampleX);

	for (int i = 1; i <= HiddenConfigs.size(); i++) {
//FORDW
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

			ReLU(tmp1);
			if (HiddenConfigs[i - 1].get_DropoutRate() < 1.0) {
			} else {
				tmp_acti.push_back(tmp1);
			}
		}
		acti_l.push_back(tmp_acti);
		tmp_acti.clear();
		//BACKW
		cuMatrixVector<double> tmp_acti2;
		for (int x = 0; x < T; x++) {
			cuMatrix<double>* tmpmat = NULL;
			tmp_acti2.push_back(tmpmat);
		}
		for (int j = T - 1; j >= 0; j--) {
			cuMatrix<double>* tmp1 = new cuMatrix<double>(
					Hiddenlayers[i - 1].U_r->rows, acti_r[i - 1][j]->cols, 1);
			cuMatrix<double>* tmp2 = new cuMatrix<double>(
					Hiddenlayers[i - 1].W_r->rows, acti_r[i - 1][j]->cols, 1);
			matrixMul(Hiddenlayers[i - 1].U_r, acti_r[i - 1][j], tmp1);
			if (j < T - 1) {
				matrixMul(Hiddenlayers[i - 1].W_r, tmp_acti2[j + 1], tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
			if (i > 1) {
				matrixMul(Hiddenlayers[i - 1].U_r, acti_l[i - 1][j], tmp2);
				ElementAdd(tmp1, tmp2, tmp1);
			}
			delete tmp2;
			ReLU(tmp1);
			if (HiddenConfigs[i - 1].get_DropoutRate() < 1.0) {
			} else {
				tmp_acti2[j] = tmp1;
			}
		}
		acti_r.push_back(tmp_acti2);
		tmp_acti2.clear();
	}

	// softmax layer forward
	cuMatrix<double>* tmp1 = new cuMatrix<double>(SMR.W_l->rows,
			acti_l[acti_l.size() - 1][mid]->cols, 1);
	cuMatrix<double>* tmp2 = new cuMatrix<double>(SMR.W_r->rows,
			acti_r[acti_r.size() - 1][mid]->cols, 1);
	matrixMul(SMR.W_l, acti_l[acti_l.size() - 1][mid], tmp1);
	matrixMul(SMR.W_r, acti_r[acti_r.size() - 1][mid], tmp2);
	ElementAdd(tmp2, tmp1, tmp1);
	get_res_array(tmp1, output, offset);

	delete tmp1;
	delete tmp2;
	for (int i = 1; i < acti_l.size(); i++) {
		for (int j = 0; j < acti_l[i].size(); j++) {
			delete acti_l[i][j];
			delete acti_r[i][j];
		}
		acti_l[i].clear();
		acti_r[i].clear();
	}
}

void testNetwork(vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR,
		vector<HiddenConfig> &HiddenConfigs, int n, const int size) {
	int* res = new int[size];
	int* label = new int[size];
	int offset;
	int batch_size = Config::instance()->get_batch_size();
	int batch_amount = size / batch_size;
	for (int i = 0; i < batch_amount; i++) {
		offset = i * batch_size;
		cuMatrixVector<double> sampleX;
		getDataMat(sampleX, offset, batch_size, n, size);

		predict(sampleX, Hiddenlayers, SMR, HiddenConfigs, res, offset);
		for (int i = 0; i < sampleX.size(); i++)
			delete sampleX[i];
		sampleX.clear();
	}
	offset = batch_amount * batch_size;
	if (size % batch_size) {
		batch_size = size % batch_size;
		cuMatrixVector<double> sampleX;
		getDataMat(sampleX, offset, batch_size, n, size);
		predict(sampleX, Hiddenlayers, SMR, HiddenConfigs, res, offset);
		for (int i = 0; i < sampleX.size(); i++)
			delete sampleX[i];
		sampleX.clear();
	}
	set_label(label, size);
	int error = 0;
	for (int i = 0; i < size; i++) {
		if (label[i] != res[i]) {
			error++;
		}
	}
	double rate = size - error;
	rate = rate / size;
	cout << "sample num :" << size << ",error num:" << error << ",correct rate:"
			<< rate << endl;

}
