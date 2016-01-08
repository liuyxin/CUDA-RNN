#include "LayerInit.h"
void weightRandomInit(HiddenLayer &hidden, int inputsize, int hiddensize) {
	const float epsilon = 0.12;
	Mat tmp_ran = Mat::ones(hiddensize, inputsize, CV_32FC1);
	randu(tmp_ran, Scalar(-1.0), Scalar(1.0));
	tmp_ran = tmp_ran * epsilon;
	hidden.U_l = cuMatrix((float*) tmp_ran.data, hiddensize, inputsize);

	hidden.U_lgrad = cuMatrix(hiddensize, inputsize);
	hidden.U_ld2 = cuMatrix(hiddensize, inputsize);
	hidden.lr_U = Config::instance()->get_lrate_w();

	Mat tmp_ran1 = Mat::ones(hiddensize, hiddensize, CV_32FC1);
	randu(tmp_ran1, Scalar(-1.0), Scalar(1.0));
	tmp_ran1 = tmp_ran1 * epsilon;
	hidden.W_l = cuMatrix((float*) tmp_ran1.data, hiddensize, hiddensize);
	hidden.W_lgrad = cuMatrix(hiddensize, hiddensize);
	hidden.W_ld2 = cuMatrix(hiddensize, hiddensize);
	hidden.lr_W = Config::instance()->get_lrate_w();

	Mat tmp_ran2 = Mat::ones(hiddensize, inputsize, CV_32FC1);
	randu(tmp_ran2, Scalar(-1.0), Scalar(1.0));
	tmp_ran2 = tmp_ran2 * epsilon;
	hidden.U_r = cuMatrix((float*) tmp_ran2.data, hiddensize, inputsize);
	hidden.U_rgrad = cuMatrix(hiddensize, inputsize);
	hidden.U_rd2 = cuMatrix(hiddensize, inputsize);

	Mat tmp_ran3 = Mat::ones(hiddensize, hiddensize, CV_32FC1);
	randu(tmp_ran3, Scalar(-1.0), Scalar(1.0));
	tmp_ran3 = tmp_ran3 * epsilon;
	hidden.W_r = cuMatrix((float*) tmp_ran3.data, hiddensize, hiddensize);
	hidden.W_rgrad = cuMatrix(hiddensize, hiddensize);
	hidden.W_rd2 = cuMatrix(hiddensize, hiddensize);

}

void weightRandomInit(SoftMax &SMR, int nclasses, int nfeatures) {
	const float epsilon = 0.12;
	Mat tmp_ran = Mat::ones(nclasses, nfeatures, CV_32FC1);
	randu(tmp_ran, Scalar(-1.0), Scalar(1.0));
	tmp_ran = tmp_ran * epsilon;
	SMR.W_l = cuMatrix((float*) tmp_ran.data, nclasses, nfeatures);
	SMR.W_lgrad = cuMatrix(nclasses, nfeatures);
	SMR.W_ld2 = cuMatrix(nclasses, nfeatures);

	Mat tmp_ran1 = Mat::ones(nclasses, nfeatures, CV_32FC1);
	randu(tmp_ran1, Scalar(-1.0), Scalar(1.0));
	tmp_ran1 = tmp_ran1 * epsilon;
	SMR.W_r = cuMatrix((float*) tmp_ran1.data, nclasses, nfeatures);
	SMR.W_rgrad = cuMatrix(nclasses, nfeatures);
	SMR.W_rd2 = cuMatrix(nclasses, nfeatures);
	SMR.cost = 0.0;
	SMR.lr_W = Config::instance()->get_lrate_w();
}

void init_HLandSMR(vector<HiddenConfig> &HiddenConfigs,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR, int word_vec_len) {
	printf("init smr&hiddenlayer star!\n");
	if (HiddenConfigs.size() > 0) {
		HiddenLayer tpntw;
		weightRandomInit(tpntw, word_vec_len, HiddenConfigs[0].get_NeuronNum());
		Hiddenlayers.push_back(tpntw);
		for (int i = 1; i < HiddenConfigs.size(); i++) {
			HiddenLayer tpntw2;
			weightRandomInit(tpntw2, HiddenConfigs[i - 1].get_NeuronNum(),
					HiddenConfigs[i].get_NeuronNum());
			Hiddenlayers.push_back(tpntw2);
		}
	}
	if (HiddenConfigs.size() == 0) {
		weightRandomInit(SMR, SMR.get_NumClasses(), word_vec_len);
	} else {
		weightRandomInit(SMR, SMR.get_NumClasses(),
				HiddenConfigs[HiddenConfigs.size() - 1].get_NeuronNum());
	}
	printf("init smr&hiddenlayer done!\n");
}
