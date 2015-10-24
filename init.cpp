#include "init.h"

void weightRandomInit(HiddenLayer &hidden, int inputsize, int hiddensize) {
	const double epsilon = 0.12;
	Mat tmp_ran = Mat::ones(hiddensize, inputsize, CV_64FC1);
	randu(tmp_ran, Scalar(-1.0), Scalar(1.0));
	tmp_ran = tmp_ran * epsilon;
	hidden.U_l = new cuMatrix<double>((double*) tmp_ran.data, hiddensize,
			inputsize, 1);
	hidden.U_l->toGpu();
	hidden.U_lgrad = new cuMatrix<double>(hiddensize, inputsize, 1);
	hidden.U_ld2 = new cuMatrix<double>(hiddensize, inputsize, 1);
	hidden.lr_U = Config::instance()->get_lrate_w();

	Mat tmp_ran1 = Mat::ones(hiddensize, hiddensize, CV_64FC1);
	randu(tmp_ran1, Scalar(-1.0), Scalar(1.0));
	tmp_ran1 = tmp_ran1 * epsilon;
	hidden.W_l = new cuMatrix<double>((double*) tmp_ran1.data, hiddensize,
			hiddensize, 1);
	hidden.W_l->toGpu();
	hidden.W_lgrad = new cuMatrix<double>(hiddensize, hiddensize, 1);
	hidden.W_ld2 = new cuMatrix<double>(hiddensize, hiddensize, 1);
	hidden.lr_W = Config::instance()->get_lrate_w();

	Mat tmp_ran2 = Mat::ones(hiddensize, inputsize, CV_64FC1);
	randu(tmp_ran2, Scalar(-1.0), Scalar(1.0));
	tmp_ran2 = tmp_ran2 * epsilon;
	hidden.U_r = new cuMatrix<double>((double*) tmp_ran2.data, hiddensize,
			inputsize, 1);
	hidden.U_r->toGpu();
	hidden.U_rgrad = new cuMatrix<double>(hiddensize, inputsize, 1);
	hidden.U_rd2 = new cuMatrix<double>(hiddensize, inputsize, 1);

	Mat tmp_ran3 = Mat::ones(hiddensize, hiddensize, CV_64FC1);
	randu(tmp_ran3, Scalar(-1.0), Scalar(1.0));
	tmp_ran3 = tmp_ran3 * epsilon;
	hidden.W_r = new cuMatrix<double>((double*) tmp_ran3.data, hiddensize,
			hiddensize, 1);
	hidden.W_r->toGpu();
	hidden.W_rgrad = new cuMatrix<double>(hiddensize, hiddensize, 1);
	hidden.W_rd2 = new cuMatrix<double>(hiddensize, hiddensize, 1);
//    hidden.lr_W = Config::instance()->get_lrate_w();
}

//smr.W_l = Mat::ones(nclasses, nfeatures, CV_64FC1);
//randu(smr.W_l, Scalar(-1.0), Scalar(1.0));
//smr.W_l = smr.W_l * epsilon;
//smr.W_lgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
//smr.W_ld2 = Mat::zeros(smr.W_l.size(), CV_64FC1);
//
//smr.W_r = Mat::ones(nclasses, nfeatures, CV_64FC1);
//randu(smr.W_r, Scalar(-1.0), Scalar(1.0));
//smr.W_r = smr.W_r * epsilon;
//smr.W_rgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
//smr.W_rd2 = Mat::zeros(smr.W_r.size(), CV_64FC1);
//
//smr.cost = 0.0;
//smr.lr_W = lrate_w;
void weightRandomInit(SoftMax &SMR, int nclasses, int nfeatures) {
	const double epsilon = 0.12;
	Mat tmp_ran = Mat::ones(nclasses, nfeatures, CV_64FC1);
	randu(tmp_ran, Scalar(-1.0), Scalar(1.0));
	tmp_ran = tmp_ran * epsilon;
	SMR.W_l = new cuMatrix<double>((double*) tmp_ran.data, nclasses, nfeatures,
			1);
	SMR.W_l->toGpu();
	SMR.W_lgrad = new cuMatrix<double>(nclasses, nfeatures, CV_64FC1);
	SMR.W_ld2 = new cuMatrix<double>(nclasses, nfeatures, CV_64FC1);

	Mat tmp_ran1 = Mat::ones(nclasses, nfeatures, CV_64FC1);
	randu(tmp_ran1, Scalar(-1.0), Scalar(1.0));
	tmp_ran1 = tmp_ran1 * epsilon;
	SMR.W_r = new cuMatrix<double>((double*) tmp_ran1.data, nclasses, nfeatures,
			1);
	SMR.W_r->toGpu();
	SMR.W_rgrad = new cuMatrix<double>(nclasses, nfeatures, CV_64FC1);
	SMR.W_rd2 = new cuMatrix<double>(nclasses, nfeatures, CV_64FC1);
	SMR.cost = 0.0;
	SMR.lr_W = Config::instance()->get_lrate_w();
}

void init_HLandSMR(vector<HiddenConfig> &HiddenConfigs,
		vector<HiddenLayer> &Hiddenlayers, SoftMax &SMR, int word_vec_len) {
	printf("init smr&hiddenlayer star!\n");
//		int hiddenlayer_num = 1;
	HiddenLayer tpntw;
	weightRandomInit(tpntw, word_vec_len, HiddenConfigs[0].get_NeuronNum());
//	weightRandomInit(tpntw, word_vec_len, 512);
	Hiddenlayers.push_back(tpntw);
//        for(int i = 1; i < hiddenlayer_num; i++){
//            Hl tpntw2;
//            weightRandomInit(tpntw2, hiddenConfig[i - 1].NumHiddenNeurons, hiddenConfig[i].NumHiddenNeurons);
//            HiddenLayers.push_back(tpntw2);
//        }
	weightRandomInit(SMR, SMR.get_NumClasses(),
			HiddenConfigs[0].get_NeuronNum());
	printf("init smr&hiddenlayer done!\n");
}

void init_acti(cuMatrixVector<double>& acti_0, vector<vector<int> >& trainX,
		cuMatrix<double>& sampleY, vector<vector<int> >& trainY, int n) {
//	printf("star init_acti()...\n");

	if (Config::instance()->get_dev_inputX() == NULL) {
		int *host_X;
		int *host_Y;
		host_X = (int *) malloc(
				sizeof(int) * trainX.size() * Config::instance()->get_ngram());
		host_Y = (int *) malloc(
				sizeof(int) * trainY.size() * Config::instance()->get_ngram());
		for (int i = 0; i < trainX.size(); i++) {
			memcpy(host_X + i * 5, &trainX[i][0], sizeof(int) * 5);
		}
		for (int i = 0; i < trainY.size(); i++) {
			memcpy(host_Y + i * 5, &trainY[i][0], sizeof(int) * 5);
		}

		for (int i = 0; i < Config::instance()->get_ngram(); i++) {
			cuMatrix<double> *tmp = new cuMatrix<double>(n,
					Config::instance()->get_batch_size(), 1);
			acti_0.push_back(tmp);
		}
		acti_0.toGpu();
		set_acti0(acti_0, host_X,
				sizeof(int) * trainX.size() * Config::instance()->get_ngram(),
				sampleY, host_Y,
				sizeof(int) * trainY.size() * Config::instance()->get_ngram());
		free(host_X);
		free(host_Y);
	} else {
		for (int i = 0; i < Config::instance()->get_ngram(); i++) {
			cuMatrix<double> *tmp = new cuMatrix<double>(n,
					Config::instance()->get_batch_size(), 1);
			acti_0.push_back(tmp);
		}
		acti_0.toGpu();
		set_acti0(acti_0, sampleY);
	}
}

void init_testdata(vector<vector<int> > &testX, vector<vector<int> > &testY) {
	int *host_X = (int *) malloc(
			sizeof(int) * testX.size() * Config::instance()->get_ngram());
	int *host_Y = (int *) malloc(
			sizeof(int) * testY.size() * Config::instance()->get_ngram());
	for (int i = 0; i < testX.size(); i++) {
		memcpy(host_X + i * 5, &testX[i][0], sizeof(int) * 5);
	}
	for (int i = 0; i < testY.size(); i++) {
		memcpy(host_Y + i * 5, &testY[i][0], sizeof(int) * 5);
	}
	Config::instance()->testX2gpu(host_X,
			sizeof(int) * testX.size() * Config::instance()->get_ngram());
	Config::instance()->testY2gpu(host_Y,
			sizeof(int) * testY.size() * Config::instance()->get_ngram());
	free (host_X);
	free (host_Y);
}
