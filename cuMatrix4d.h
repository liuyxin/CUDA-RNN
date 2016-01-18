#ifndef CUMATRIX4D_H
#define CUMATRIX4D_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include "helper_cuda.h"
#include "MemoryMonitor.h"
//#include "hardware.h"
#include "cuMatrix.h"
using namespace std;
class cuMatrix4d {
	public:
		std::shared_ptr<MatData> data;
		cuMatrix4d(int r = 0 ,int c = 0 ,int ch = 0 ,int t = 0){
			row = r;
			col = c;
			channal = ch;
			timeStep = t;
			area2 = r * c;
			area3 = r * c * ch;
			len_ = area3 * t;		
			size = len_ * sizeof(float);
			data = std::make_shared< MatData >(r,c,ch,t);
		}	
		cuMatrix4d(std::shared_ptr<MatData> td,int r,int c,int ch,int t){
			row = r;
			col = c;
			channal = ch;
			timeStep = t;
			area2 = r * c;
			area3 = r * c * ch;
			len_ = area3 * t;		
			size = len_ * sizeof(float);
			data = td;
		}	
		int rows() {
			return row;
		}
		int cols() {
			return col;
		}
		int channals(){
			return channal;
		}
		int ts(){
			return timeStep;
		}
		int area3D(){
			return area3;
		}	
		int area2D(){
			return area2;
		}
		int len(){
			return len_;
		}
		unsigned int sizes() {
			return size;
		}
		float* getDev() {
			return data->getDev();
		}
		void printMat(){
			assert(data->getDev()!=NULL);
			int size2 = area2 * sizeof(float);
			float* f[2];
			f[0] = (float*)malloc(size2);
			f[1] = (float*)malloc(size2);
			int t = 0;
			cudaError_t cudaStat;
			cudaStat = cudaMemcpy(f[t], data->getDev(), size2,
					cudaMemcpyDeviceToHost);
			if (cudaStat != cudaSuccess) {
				printf("cuMatrix4d::printMat,timeStep 0,channal 0 failed\n");
				exit(0);
			}
			for(int i = 0 ; i < timeStep ;i ++){
				for(int j = 0 ; j < channal ; j++){
					int offset = i * area3 + j * area2 + area2;
					if(offset < area3 * timeStep){
						cudaStat = cudaMemcpyAsync(f[1-t], data->getDev()+offset, size2,
								cudaMemcpyDeviceToHost);
						if (cudaStat != cudaSuccess) {
							printf("cuMatrix4d::printMat,timeStep %d,channal %d failed\n",i,j);
							exit(0);
						}
					}
					float* tmp = f[t];
					printf("timeStep %d,channal %d:\n",i,j);
					for(int x = 0 ; x < row ; x ++){
						for(int y = 0 ; y < col ; y ++)
						{
							printf("%f,",tmp[x*col + y]);
						}
						printf("\n");
					}
					t = 1 - t;
				}
			}	
			delete [] f[1];
			delete [] f[0];
		}
		float& getSum();
		cuMatrix4d Mul(cuMatrix4d m);
		cuMatrix4d t();

	private:
		int row;
		int col;
		int channal;
		int timeStep;
		int area2;
		int area3;
		int len_;
		float sum;
		unsigned int size;
};

#endif
