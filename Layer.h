#ifndef LAYER_H
#define LAYER_H
#include "Base.h"

class HiddenConfig{
public:
	HiddenConfig(int nn, double wd , double dr){
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
	void set_WeightDecay(double i){
		WeightDecay = i;
	}
	void set_DropoutRate(double i){
		DropoutRate = i;
	}
	int get_NeuronNum() {
		return NeuronNum;
	}
	double get_WeightDecay(){
		return WeightDecay;
	}
	double get_DropoutRate(){
		return DropoutRate;
	}
	void set_inputs(cuMatrix<double>* t)
	{
		inputs = t;
	}
	void set_outputs(cuMatrix<double>* t)
	{
		outputs = t;
	}
	cuMatrix<double>* get_inputs()
	{
		return inputs;
	}
	cuMatrix<double>* get_outputs(cuMatrix<double>* t)
	{
		return outputs;
	}
private:
	int NeuronNum;
	double WeightDecay;
	double DropoutRate;
	cuMatrix<double>* inputs;
	cuMatrix<double>* outputs;
};
class HiddenLayer {
public:
	cuMatrix<double>* W_l; // weight between current time t with previous time t-1
	cuMatrix<double>* U_l;  // weight between hidden layer with previous layer
	cuMatrix<double>* W_lgrad;
	cuMatrix<double>* U_lgrad;
	cuMatrix<double>* W_ld2;
	cuMatrix<double>* U_ld2;
	cuMatrix<double>* W_r; // weight between current time t with previous time t-1
	cuMatrix<double>* U_r;  // weight between hidden layer with previous layer
	cuMatrix<double>* W_rgrad;
	cuMatrix<double>* U_rgrad;
	cuMatrix<double>* W_rd2;
	cuMatrix<double>* U_rd2;
	double lr_W;
	double lr_U;
};


class SoftMax {
public:
	SoftMax():NumClasses(10),WeightDecay(1e-6),inputs(NULL),outputs(NULL){}
	cuMatrix<double>* W_l;
	cuMatrix<double>* W_lgrad;
	cuMatrix<double>* W_ld2;
	cuMatrix<double>* W_r;
	cuMatrix<double>* W_rgrad;
	cuMatrix<double>* W_rd2;
    double cost;
    double lr_W;

	void set_NumClasses(int i){
		NumClasses = i;
	}
	void set_WeightDecay(double i){
		WeightDecay = i;
	}
	int get_NumClasses() {
		return NumClasses;
	}
	double get_WeightDecay(){
		return WeightDecay;
	}
	void set_inputs(cuMatrix<double>* t)
	{
		inputs = t;
	}
	void set_outputs(cuMatrix<double>* t)
	{
		outputs = t;
	}
	cuMatrix<double>* get_inputs()
	{
		return inputs;
	}
	cuMatrix<double>* get_outputs(cuMatrix<double>* t)
	{
		return outputs;
	}
private:
    int NumClasses ;
    double WeightDecay ;
	cuMatrix<double>* inputs;
	cuMatrix<double>* outputs;
};

#endif
