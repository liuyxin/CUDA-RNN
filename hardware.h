#ifndef HARDWARE_H
#define HARDWARE_H
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <nvml.h>
using namespace std;

class Devices {
	public:
		Devices(){
			devInit();
		}
		~Devices(){
			nvmlReturn_t result;
			result = nvmlShutdown();
			if (NVML_SUCCESS != result)
				printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

		}
		static Devices* instance() {
			static Devices* dev = new Devices();
			return dev;
		}
		const char * convertToComputeModeString(nvmlComputeMode_t mode)
		{
			switch (mode)
			{
				case NVML_COMPUTEMODE_DEFAULT:
					return "Default";
				case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
					return "Exclusive_Thread";
				case NVML_COMPUTEMODE_PROHIBITED:
					return "Prohibited";
				case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
					return "Exclusive Process";
				default:
					return "Unknown";
			}
		}

		void devInit() {
			nvmlReturn_t result;
			unsigned int i;
			result = nvmlInit();
			if (NVML_SUCCESS != result)
			{
				printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));

				printf("Press ENTER to continue...\n");
				getchar();
				exit(0);
			}
			result = nvmlDeviceGetCount(&count);
			printf("count = %d\n",count);
			if (NVML_SUCCESS != result)
			{
				printf("Failed to query device count: %s\n", nvmlErrorString(result));
				exit(0);
			}
			//cudaGetDeviceCount(&count);
			chosenDev = 0 ;
			for(i = 0 ; i < count ; i ++){
				result = nvmlDeviceGetHandleByIndex(i, &device);
				if (NVML_SUCCESS != result)
				{
					printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
					exit(0);
				}
				nvmlMemory_t mem;
				result = nvmlDeviceGetMemoryInfo(device , &mem);
				if (NVML_SUCCESS != result)
				{
					printf("Failed to get memInfo of device %i: %s\n", i, nvmlErrorString(result));
					exit(0);
				}
				printf("Device %u ,total:%llu,free:%llu,used:%llu,\n",i,mem.total,mem.free,mem.used);  
				if(mem.free > availableMemory){
					availableMemory = mem.free;
					chosenDev = i;
				}
			}
			cudaError_t s;
			s = cudaSetDevice((int)chosenDev);
			if(s != cudaSuccess){
				printf("cudaSetDevice() fail:!\n");
				exit(0);
			}
			else{
				printf("cudaSetDevice(%d),success!\n",chosenDev);
			}
			result = nvmlDeviceGetHandleByIndex(chosenDev, &device);
			if (NVML_SUCCESS != result)
			{
				printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
				exit(0);
			}
			cudaGetDeviceProperties(&prop, chosenDev);
		}
		void printDeviceInfo() {
			cout << "device name:" << prop.name << endl;
		}
		unsigned long long getAvailableMemory(){
			nvmlMemory_t mem;
			if(NVML_SUCCESS != nvmlDeviceGetMemoryInfo(device,&mem)){
				printf("getAvailableMemory() failed!\n");
				exit(0);
			}
			availableMemory = mem.free;
			return mem.free;
		}
		int maxThreadNum() {
			return prop.maxThreadsPerBlock;
		}
		int* blockDim() {
			return prop.maxThreadsDim;
		}
		int* griDim() {
			return prop.maxGridSize;
		}
		size_t get_sharedmemorysize(){
		//	printf("sharedmemory size:%d\n",prop.sharedMemPerBlock);
			return prop.sharedMemPerBlock;
		}
		cudaDeviceProp get_prop(){
			return prop;
		}
		unsigned long long availableMemory;
	private:
		cudaDeviceProp prop;
		nvmlDevice_t device;
		unsigned int count;
		unsigned int chosenDev;
};


#endif
