#ifndef LAYER_H
#define LAYER_H
#include "cuMatrix.h"

class HiddenConfig{
public:
	HiddenConfig(int nn, float wd , float dr){
		NeuronNum = nn;
		WeightDecay = wd;
		DropoutRate = dr;
	}

	static HiddenConfig* instance(){
		static HiddenConfig* tmp = new HiddenConfig(0,0,0);
		return tmp;
	}
	void set_NeuronNum(int i) {
		NeuronNum = i;
	}
	void set_WeightDecay(float i){
		WeightDecay = i;
	}
	void set_DropoutRate(float i){
		DropoutRate = i;
	}
	int get_NeuronNum() {
		return NeuronNum;
	}
	float get_WeightDecay(){
		return WeightDecay;
	}
	float get_DropoutRate(){
		return DropoutRate;
	}
private:
	int NeuronNum;
	float WeightDecay;
	float DropoutRate;
};
class HiddenLayer {
public:
	cuMatrix W_l; // weight between current time t with previous time t-1
	cuMatrix U_l;  // weight between hidden layer with previous layer
	cuMatrix W_lgrad;
	cuMatrix U_lgrad;
	cuMatrix W_ld2;
	cuMatrix U_ld2;
	cuMatrix W_r; // weight between current time t with previous time t-1
	cuMatrix U_r;  // weight between hidden layer with previous layer
	cuMatrix W_rgrad;
	cuMatrix U_rgrad;
	cuMatrix W_rd2;
	cuMatrix U_rd2;
	float lr_W;
	float lr_U;
};


class SoftMax {
public:
	SoftMax():NumClasses(10),WeightDecay(1e-6){}
	cuMatrix W_l;
	cuMatrix W_lgrad;
	cuMatrix W_ld2;
	cuMatrix W_r;
	cuMatrix W_rgrad;
	cuMatrix W_rd2;
    float cost;
    float lr_W;

	void set_NumClasses(int i){
		NumClasses = i;
	}
	void set_WeightDecay(float i){
		WeightDecay = i;
	}
	int get_NumClasses() {
		return NumClasses;
	}
	float get_WeightDecay(){
		return WeightDecay;
	}

private:
    int NumClasses ;
    float WeightDecay ;
};
#endif
