#ifndef HARDWARE_H
#define HARDWARE_H
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
using namespace std;

class Devices {
public:
	Devices(){
			get_info();
	}
	static Devices* instance() {
		static Devices* dev = new Devices();
		return dev;
	}

	void get_info() {
		cudaGetDeviceCount(&count);
		cudaGetDeviceProperties(&prop, count - 1);
	}
	void print_devname() {
		cout << "device name:" << prop.name << endl;
	}
	int max_ThreadsPerBlock() {
		cout << "max threads per block:" << prop.maxThreadsPerBlock << endl;
		return prop.maxThreadsPerBlock;
	}
	//max thread num in every dim;
	int* max_ThreadsDim() {
		cout << "max threads dims:" << prop.maxThreadsDim[0] << "  "
				<< prop.maxThreadsDim[1] << "  " << prop.maxThreadsDim[2]
				<< endl;
		return prop.maxThreadsDim;
	}
	int* max_GridDims() {
		cout << "max grid dims:" << prop.maxGridSize[0] << "  "
				<< prop.maxGridSize[1] << "  " << prop.maxGridSize[2] << endl;
		return prop.maxGridSize;
	}
	size_t get_sharedmemorysize(){
		printf("sharedmemory size:%d\n",prop.sharedMemPerBlock);
		return prop.sharedMemPerBlock;
	}
	cudaDeviceProp get_prop(){
		return prop;
	}
private:
	cudaDeviceProp prop;
	int count;
};
void getDevicesinfo();

#endif
