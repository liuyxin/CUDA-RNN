#include "cuMatrixVector.h"
#include "Base.h"
#include "readdata.h"
#include "Layer.h"
#include "cuMatrixVector.h"
#include "Config.h"
#include "init.h"
#include "train_network.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include "hardware.h"
using namespace std;

int main() {

	getDevicesinfo();
	vector<vector<singleWord> > traindata;
	vector<vector<singleWord> > testdata;
	vector<string> re_label;
	map<string, int> labelmap;
	map<string, int> wordmap;
	vector<string> re_word;
	readdata("dataset/news_tagged_data.txt", traindata, testdata, re_label,
			labelmap);
	cout << "traindata.size() :" << traindata.size() << endl
			<< "testdata.size()  :" << testdata.size() << endl
			<< "re_label.size()  :" << re_label.size() << endl
			<< "labelmap.size()  :" << labelmap.size() << endl;
//  softmaxConfig.NumClasses = re_label.size();
	removeNumber(traindata);
	getWordMap(traindata, wordmap, re_word);
	cout << "wordmap.size()     :" << wordmap.size() << endl
			<< "re_wordmap.size()  :" << re_word.size() << endl;
	vector<vector<int> > trainX;
	vector<vector<int> > trainY;
	vector<vector<int> > testX;
	vector<vector<int> > testY;
	resolutioner(traindata, trainX, trainY, wordmap);
	resolutioner(testdata, testX, testY, wordmap);
	vector<HiddenConfig> HiddenConfigs;
	vector<HiddenLayer> Hiddenlayers;
	HiddenConfigs.push_back(*HiddenConfig::instance());
	SoftMax SMR;
	init_HLandSMR(HiddenConfigs, Hiddenlayers, SMR, re_word.size());

	trainNetwork(trainX, trainY, HiddenConfigs,Hiddenlayers, SMR, testX, testY, re_word);

	return 0;
}
