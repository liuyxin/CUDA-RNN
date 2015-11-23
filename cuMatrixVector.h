#ifndef CUMATRIXVECTOR_H
#define CUMATRIXVECTOR_H
#include <vector>
#include <stdio.h>
#include "cuMatrix.h"
#include <assert.h>
using namespace std;

class cuMatrixVector {
public:
	cuMatrixVector() {
		m_host = NULL;
		m_dev = NULL;
	}
	~cuMatrixVector() {
		if (m_host != NULL)
			free(m_host);
		if (m_dev != NULL)
			cudaFree(m_dev);
//		for(int i = 0 ; i < m_vec.size() ; i++){
//			delete m_vec[i];
//		}
		m_vec.clear();
	}
	cuMatrix*& operator[](size_t index) {
		if (index >= m_vec.size()) {
			printf("cuMatrixVector operator[] error\n");
			exit(0);
		}
		return m_vec[index];
	}
	void toGpu() {
		cudaError_t cudaStat;
		m_host = (float**) malloc(m_vec.size() * sizeof(float*));
		if (!m_host) {
			printf("cuMatrixVector malloc m_host fail\n");
			exit(0);
		}
		cudaStat = cudaMalloc((void**) &m_dev, m_vec.size() * sizeof(float*));
		if (cudaStat != cudaSuccess) {
			printf("cuMatrixVector cudaMalloc m_dev fail\n");
			exit(0);
		}
		for (int p = 0; p < (int) m_vec.size(); p++) {
			m_host[p] = m_vec[p]->getDev();
		}
		cudaStat = cudaMemcpyAsync(m_dev, m_host, sizeof(float*) * m_vec.size(),
				cudaMemcpyHostToDevice,0);
		if (cudaStat != cudaSuccess) {
			printf("cuMatrixVector::toGpu cudaMemcpy fail\n");
			exit(0);
		}
	}

	void push_back(cuMatrix* p) {
		m_vec.push_back(p);
	}
	size_t size() {
		return m_vec.size();
	}

	float** &get_host() {
		assert(m_host != NULL);
		return m_host;
	}
	float** &get_devPoint() {
		assert(m_dev != NULL);
		return m_dev;
	}
private:
	vector<cuMatrix*> m_vec;
	float** m_host; //point 2 cuMatrix->Data->getDev()
	float** m_dev;
};
#endif
