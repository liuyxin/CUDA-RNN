#ifndef CONFIG_H
#define CONFIG_H
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "Base.h"
#include "cuMatrixVector.h"
#include "Layer.h"
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

class Config {
public:
	vector<HiddenLayer*> Hiddens;
	vector<SoftMax*> SMS;
	Config() :
			is_gradient_checking(false), use_log(false), non_linearity(NL_RELU), batch_size(
					50), training_epochs(30), iter_per_epo(100), lrate_w(3e-3), lrate_b(
					1e-3), ngram(5), training_percent(0.80), test_num(0), train_num(
					0), randproductor(NULL), dev_inputX(NULL), dev_inputY(NULL), dev_testX(
					NULL), dev_testY(NULL), sizex(0), sizey(0), sizetx(0), sizety(
					0) {
	}
	static Config* instance() {
		static Config* config = new Config();
		return config;
	}

	bool get_gradient() {
		return is_gradient_checking;
	}
	void set_gradien(bool i) {
		is_gradient_checking = i;
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
	void randproductor_init() {
		if (train_num == 0) {
			printf(
					"void randproductor_init() error: train_num = 0, run Config::set_traintest_num(int sample_num) first\n");
			exit(0);
		} else {
			if (randproductor == NULL) {
				randproductor = (int *) malloc(train_num * sizeof(int));
				for (int i = 0; i < train_num; i++)
					randproductor[i] = i;
			}
		}
	}

	int* &get_rand(bool x = 1) {
		if (randproductor == NULL) {
			printf(
					"int* get_rand() error: randproductor = NULL, run Config::randproductor_init() first\n");
			exit(0);
		} else if (x) {
			random_shuffle(randproductor, randproductor + train_num);
			return randproductor;
		} else {
			return randproductor;
		}
	}

	void testX2gpu(int *host_, int size) {
		MemoryMonitor::instance()->gpuMalloc((void**) &dev_testX, size);
		cudaError_t cudaStat;
		sizetx = size;
		cudaStat = cudaMemcpy(dev_testX, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			MemoryMonitor::instance()->freeGpuMemory(dev_testX);
			exit(0);
		}
	}
	void testY2gpu(int *host_, int size) {
		MemoryMonitor::instance()->gpuMalloc((void**) &dev_testY, size);
		cudaError_t cudaStat;
		sizety = size;
		cudaStat = cudaMemcpy(dev_testY, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			MemoryMonitor::instance()->freeGpuMemory(dev_testY);
			exit(0);
		}
	}

	int* &get_dev_testX() {
		return dev_testX;
	}
	int* &get_dev_testY() {
		return dev_testY;
	}
	void inputx2gpu(int *host_, int size) {
		MemoryMonitor::instance()->gpuMalloc((void**) &dev_inputX, size);
		cudaError_t cudaStat;
		sizex = size;
		cudaStat = cudaMemcpy(dev_inputX, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			MemoryMonitor::instance()->freeGpuMemory(dev_inputX);
			exit(0);
		}
	}
	void inputy2gpu(int *host_, int size) {
		MemoryMonitor::instance()->gpuMalloc((void**) &dev_inputY, size);
		cudaError_t cudaStat;
		sizey = size;
		cudaStat = cudaMemcpy(dev_inputY, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			MemoryMonitor::instance()->freeGpuMemory(dev_inputY);
			exit(0);
		}
	}

	int* &get_dev_inputX() {
		return dev_inputX;
	}
	int* &get_dev_inputY() {
		return dev_inputY;
	}
	int	get_sizetx() {
		return sizetx;
	}
	int	get_sizety() {
		return sizety;
	}
	int	get_sizex() {
		return sizex;
	}
	int	get_sizey() {
		return sizey;
	}
private:
	bool is_gradient_checking;
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
	int *randproductor;
	int *dev_inputX;
	int *dev_inputY;
	int *dev_testX;
	int *dev_testY;
	int sizex;
	int sizey;
	int sizetx;
	int sizety;
};

#endif
