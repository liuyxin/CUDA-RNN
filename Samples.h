#ifndef SAMPLES_H
#define SAMPLES_H
#include "Config.h"
#include <algorithm>
#include <stdio.h>
class Samples {
public:
	Samples() :
			randproductor(NULL), dev_trainX(NULL), dev_trainY(
			NULL), dev_testX(NULL), dev_testY(NULL), sizex(0), sizey(0), sizetx(
					0), sizety(0) {
	}
	static Samples* instance() {
		static Samples* samples = new Samples();
		return samples;
	}
	void randproductor_init() {
		if (Config::instance()->get_train_num() == 0) {
			printf(
					"void randproductor_init() error: train_num = 0, run Config::set_traintest_num(int sample_num) first\n");
			exit(0);
		} else {
			if (randproductor == NULL) {
				randproductor = (int *) malloc(Config::instance()->get_train_num() * sizeof(int));
				for (int i = 0; i < Config::instance()->get_train_num(); i++)
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
			random_shuffle(randproductor, randproductor + Config::instance()->get_train_num());
			return randproductor;
		} else {
			return randproductor;
		}
	}

	void testX2gpu(int *host_, int size) {
		cudaError_t cudaStat = cudaMalloc((void**) &dev_testX, size);
		if (cudaStat != cudaSuccess) {
			printf("Samples::testX2gpu() failed\n");
			exit(0);
		}
		sizetx = size;
		cudaStat = cudaMemcpy(dev_testX, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}
	void testY2gpu(int *host_, int size) {
		cudaError_t cudaStat = cudaMalloc((void**) &dev_testY, size);
		if (cudaStat != cudaSuccess) {
			printf("Samples::testY2gpu() failed\n");
			exit(0);
		}
		sizety = size;
		cudaStat = cudaMemcpy(dev_testY, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}

	int* &get_testX() {
		return dev_testX;
	}
	int* &get_testY() {
		return dev_testY;
	}
	void trainX2gpu(int *host_, int size) {
		cudaError_t cudaStat = cudaMalloc((void**) &dev_trainX, size);
		if (cudaStat != cudaSuccess) {
			printf("Samples::testY2gpu() failed\n");
			exit(0);
		}
		sizex = size;
		cudaStat = cudaMemcpy(dev_trainX, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}
	void trainY2gpu(int *host_, int size) {
		cudaError_t cudaStat = cudaMalloc((void**) &dev_trainY, size);
		if (cudaStat != cudaSuccess) {
			printf("Samples::testY2gpu() failed\n");
			exit(0);
		}
		sizey = size;
		cudaStat = cudaMemcpy(dev_trainY, host_, size, cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("set_gpudata::toGPU data upload failed\n");
			exit(0);
		}
	}

	int* &get_trainX() {
		return dev_trainX;
	}
	int* &get_trainY() {
		return dev_trainY;
	}
	int get_sizetx() {
		return sizetx;
	}
	int get_sizety() {
		return sizety;
	}
	int get_sizex() {
		return sizex;
	}
	int get_sizey() {
		return sizey;
	}
private:
	int *randproductor;
	int *dev_trainX;
	int *dev_trainY;
	int *dev_testX;
	int *dev_testY;
	int sizex; //trainx
	int sizey;
	int sizetx;//testx
	int sizety;
};

#endif
