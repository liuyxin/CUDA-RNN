#ifndef CONFIG_H__
#define CONFIG_H__
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuMatrix.h"
#include "cuMatrixVector.h"
#include "Layer.h"
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2
using namespace std;

class Config {
public:
	vector<HiddenConfig> HiddenConfigs;
	Config() :
			test_num(0), train_num(0),word_num(0),trainX_num(0),testX_num(0) {
	}
	static Config* instance() {
		static Config* config = new Config();
		return config;
	}
	bool get_use_log() {
		return non_linearity;
	}
	void set_use_log(bool i) {
		use_log = i;
	}
	int get_non_linearity() {
		return non_linearity;
	}
	void set_non_linearity(int i) {
		non_linearity = i;
	}
	int get_batch_size() {
		return batch_size;
	}
	void set_batch_size(int i) {
		batch_size = i;
	}
	int get_training_epochs() {
		return training_epochs;
	}
	void set_training_epochs(int i) {
		training_epochs = i;
	}
	int get_iter_per_epo() {
		return iter_per_epo;
	}
	void set_iter_per_epo(int i) {
		iter_per_epo = i;
	}
	double get_lrate_w() {
		return lrate_w;
	}
	void set_lrate_w(int i) {
		lrate_w = i;
	}
	double get_lrate_b() {
		return lrate_b;
	}
	void set_lrate_b(int i) {
		lrate_b = i;
	}
	int get_ngram() {
		return ngram;
	}
	void set_ngram(int i) {
		ngram = i;
	}
	float get_TRAINING_PERCENT() {
		return training_percent;
	}
	void set_TRAINING_PERCENT(float i) {
		training_percent = i;
	}
	void set_traintest_num(int sample_num) {
		train_num = sample_num * training_percent;
		test_num = sample_num - train_num;
	}
	int get_test_num() {
		return test_num;
	}
	int get_train_num() {
		return train_num;
	}
	void set_trainX_num(int i){
		trainX_num = i;
	}
	int trainXNum(){
		return trainX_num;
	}
	void set_testX_num(int i){
		testX_num = i;
	}
	int testXNum(){
		return testX_num;
	}
	void set_word_num(int i){
		word_num = i;
	}
	int get_wordNum(){
		return word_num;
	}
	void init(string path, SoftMax &SMR);
private:
	bool use_log;
	int non_linearity;
	int batch_size;
	int training_epochs;
	int iter_per_epo;
	double lrate_w;
	double lrate_b;
	int ngram;
	float training_percent;
	int train_num;
	int test_num;
	int word_num;
	int trainX_num;
	int testX_num;
	string m_configStr;
	string read_2_string(string File_name);
	void deleteSpace();
	void deleteComment();
	bool get_word_bool(string &str, string name);
	int get_word_int(string &str, string name);
	float get_word_float(string &str, string name);
	int get_word_type(string &str, string name);
	void get_layers_config(string &str, SoftMax &SMR);
};

#endif
