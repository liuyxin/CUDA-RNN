#include "resultPredict.h"

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
		cuMatrixVector sampleX;
		getDataMat(sampleX, offset, batch_size, wn, flag);

		predict(sampleX, Hiddenlayers, SMR, res, offset);
        for(int j = sampleX.size() - 1 ; j >= 0; j--){
            delete sampleX[j];
        }
    }
	offset = batch_amount * batch_size;
	if (size % batch_size) {
		batch_size = size % batch_size;
		cuMatrixVector sampleX;
		getDataMat(sampleX, offset, batch_size, wn, size);
		predict(sampleX, Hiddenlayers, SMR, res, offset);
        for(int j = sampleX.size() - 1 ; j >= 0; j--){
            delete sampleX[j];
        }
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
void predict(cuMatrixVector &sampleX, vector<HiddenLayer> &Hiddenlayers,
		SoftMax &SMR, int* output, int offset) {
	int T = sampleX.size();
	int mid = (int) (T / 2.0);
	int HiddenNum = Config::instance()->HiddenConfigs.size();
	std::vector<vector<cuMatrix> > acti_l(HiddenNum + 1);
	std::vector<vector<cuMatrix> > acti_r(HiddenNum + 1);
	for (int i = 0; i < T; i++) {
		cuMatrix* ptr = sampleX[i];
		acti_l[0].push_back(*ptr);
		acti_r[0].push_back(*ptr);
	}
	for (int i = 1; i <= HiddenNum; ++i) {
		//time forward
		for (int j = 0; j < T; j++) {
			acti_l[i].push_back(
					cuMatrix(Hiddenlayers[i - 1].U_l.rows(),
							acti_l[i - 1][0].cols()));
			acti_r[i].push_back(
					cuMatrix(Hiddenlayers[i - 1].U_r.rows(),
							acti_r[i - 1][0].cols()));
		}
		for (int j = 0; j < T; ++j) {
			cuMultiplication(Hiddenlayers[i - 1].U_l, acti_l[i - 1][j],
					acti_l[i][j]);

			if (j > 0) {
				acti_l[i][j] += Hiddenlayers[i - 1].W_l * acti_l[i][j - 1];
			}
			if (i > 1) {
				acti_l[i][j] += Hiddenlayers[i - 1].U_l * acti_r[i - 1][j];
			}
			acti_l[i][j].ReLU2(acti_l[i][j]);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate()
					< 1.0) {
				acti_l[i][j] *= Config::instance()->HiddenConfigs[i - 1].get_DropoutRate();
			}
		}
		//time backward
		for (int j = T - 1; j >= 0; --j) {
			cuMultiplication(Hiddenlayers[i - 1].U_r, acti_r[i - 1][j],
					acti_r[i][j]);
			if (j < T - 1) {
				acti_r[i][j] += Hiddenlayers[i - 1].W_r * acti_r[i][j + 1];
			}
			if (i > 1) {
				acti_r[i][j] += Hiddenlayers[i - 1].U_r * acti_l[i - 1][j];
			}
			acti_r[i][j].ReLU2(acti_r[i][j]);
			if (Config::instance()->HiddenConfigs[i - 1].get_DropoutRate() < 1.0) {
				acti_r[i][j] *= Config::instance()->HiddenConfigs[i - 1].get_DropoutRate();
			}
		}
	}
	cuMatrix M(SMR.W_l.rows(), acti_l[acti_l.size() - 1][mid].cols());
	cuMultiplication(SMR.W_l, acti_l[acti_l.size() - 1][mid], M);
	M += SMR.W_r * acti_r[acti_r.size() - 1][mid];
	get_res_array(M, output, offset);
}
