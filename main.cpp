#include <iostream>
#include "cuMatrixVector.h"
#include "cuMatrix.h"
#include "readdata.h"
#include "Layer.h"
#include "Config.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include "hardware.h"
#include "Config.h"
#include "LayerInit.h"
#include "InputInit.h"
#include "TrainNetwork.h"
#include "test.h"
using namespace std;
void PrintLayerConfigs(SoftMax& SMR) {
	cout
			<< "************************** HiddenLayer parameter **************************"
			<< endl;
	for (int i = 0; i < Config::instance()->HiddenConfigs.size(); i++) {
		printf("Hidden %d :\n", i);
		cout << "NeuronNum = "
				<< Config::instance()->HiddenConfigs[i].get_NeuronNum() << endl;
		cout << "WeightDecay = "
				<< Config::instance()->HiddenConfigs[i].get_WeightDecay()
				<< endl;
		cout << "DropoutRate = "
				<< Config::instance()->HiddenConfigs[i].get_DropoutRate()
				<< endl;
	}
	printf("\n");
	cout
			<< "*************************** SOFTMAX parameter ***************************"
			<< endl;
	cout << "class num = " << SMR.get_NumClasses() << endl;
	cout << "WeightDecay= " << SMR.get_WeightDecay() << endl;
}

//cout << "wordmap.size()     :" << wordmap.size() << endl
//		<< "re_wordmap.size()  :" << re_word.size() << endl;
//int main() {
//	getDevicesinfo();
//	vector<vector<singleWord> > traindata;
//	vector<vector<singleWord> > testdata;
//	vector<string> re_label;
//	map<string, int> labelmap;
//	map<string, int> wordmap;
//	vector<string> re_word;
//	vector<HiddenLayer> Hiddenlayers;
//	SoftMax SMR;
//	Config::instance()->init("config.txt", SMR);
//	readdata("dataset/news_tagged_data.txt", traindata, testdata, re_label,
//			labelmap);
//	removeNumber(traindata);
//
//	getWordMap(traindata, wordmap, re_word);
//	vector<vector<int> > trainX;
//	vector<vector<int> > trainY;
//	vector<vector<int> > testX;
//	vector<vector<int> > testY;
//	resolutioner(traindata, trainX, trainY, wordmap);
//	resolutioner(testdata, testX, testY, wordmap);
//	Config::instance()->set_word_num(re_word.size());
//	Config::instance()->set_trainX_num(trainX.size());
//	Config::instance()->set_testX_num(testX.size());
//	cout<< Config::instance()->get_wordNum()<<endl
//		<<Config::instance()->trainXNum()<<endl
//		<<Config::instance()->testXNum()<<endl;
//	init_HLandSMR(Config::instance()->HiddenConfigs, Hiddenlayers, SMR,
//			re_word.size());
//
//	Data2GPU(trainX, trainY, testX, testY);
//	trainNetwork(Hiddenlayers,SMR,re_word.size());
//	return 0;
//}
//int main(){
//	Devices::instance();
//	printf("available = %llu\n",Devices::instance()->getAvailableMemory());
//	cuMatrix4d mat(3,2,3,3);
//	init32(mat);
//	mat.printMat();
//	cuMatrix res(3,18);
//	cuMatrix4dRightTrans(mat,res);
//	printf("\n\n");
//	res.printMat();
//	return 0;
//}
