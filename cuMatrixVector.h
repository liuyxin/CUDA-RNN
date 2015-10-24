#ifndef __CU_MATRIX_VECTOR_H_
#define __CU_MATRIX_VECTOR_H_

#include <vector>
#include "Base.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "MemoryMonitor.h"
using namespace std;

template<class T>
class cuMatrixVector {
public:
	cuMatrixVector() :
			m_hstPoint(0), m_devPoint(0) {
	}
	~cuMatrixVector() {
		MemoryMonitor::instance()->freeCpuMemory(m_hstPoint);
		MemoryMonitor::instance()->freeGpuMemory(m_devPoint);
		m_vec.clear();
	}
	cuMatrix<T>*& operator[](size_t index) {
		if (index >= m_vec.size()) {
			//Assert(true);
			printf("cuMatrix Vector operator[] error\n");
			exit(0);
		}
		return m_vec[index];
	}
	size_t size() {
		return m_vec.size();
	}

	void push_back(cuMatrix<T>* m) {
		m_vec.push_back(m);
	}

	void toGpu() {
		cudaError_t cudaStat;

		m_hstPoint = (T**) MemoryMonitor::instance()->cpuMalloc(
				m_vec.size() * sizeof(T*));
		if (!m_hstPoint) {
			printf("cuMatrixVector<T> malloc m_hstPoint fail\n");
			exit(0);
		}

		cudaStat = MemoryMonitor::instance()->gpuMalloc((void**) &m_devPoint,
				sizeof(T*) * m_vec.size());
		if (cudaStat != cudaSuccess) {
			printf("cuMatrixVector<T> cudaMalloc m_devPoint fail\n");
			exit(0);
		}

		for (int p = 0; p < (int) m_vec.size(); p++) {
			m_hstPoint[p] = m_vec[p]->getDev();
		}

		cudaStat = cudaMemcpy(m_devPoint, m_hstPoint, sizeof(T*) * m_vec.size(),
				cudaMemcpyHostToDevice);
		if (cudaStat != cudaSuccess) {
			printf("cuMatrixVector::toGpu cudaMemcpy w fail\n");
			exit(0);
		}
	}
	void reverse_(){
		reverse(m_vec.begin(),m_vec.end());
		toGpu();
	}
	T** &get_hstPoint() {
		return m_hstPoint;
	}
	T** &get_devPoint() {
		return m_devPoint;
	}
	void clear() {
		MemoryMonitor::instance()->freeCpuMemory(m_hstPoint);
		MemoryMonitor::instance()->freeGpuMemory(m_devPoint);
		m_vec.clear();
	}
private:
	vector<cuMatrix<T>*> m_vec;
	T** m_hstPoint;
	T** m_devPoint;
};

#endif
