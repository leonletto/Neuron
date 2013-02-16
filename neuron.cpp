#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "neuron.h"

neuron::neuron() {
	nInputs = 0;
	bias = 0;
	xw = 0;
	output = 0;
	AF = 0;
	threshold = 0;
	nExamples = 0;
}

neuron::neuron(int n) {
	nInputs = n;
	bias = 0;
	xw = 0;
	output = 0;
	AF = 0;
	threshold = 0;
	nExamples = 0;
	inputs.reserve(n);
	weights.reserve(n);
}

neuron::neuron(int n, vector<double> w, double b) {
	nInputs = n;
	bias = b;
	xw = 0;
	output = 0;
	AF = 0;
	threshold = 0;
	nExamples = 0;
	inputs.reserve(n);
	weights.reserve(n);
	weights.assign(w.begin(),w.end());
}

neuron::~neuron() {

}

void neuron::setSize(int n) {
	nInputs = n;
	weights.clear();
	weights.reserve(n);
	inputs.clear();
	inputs.reserve(n);
}

void neuron::setWeights(vector<double> w) {
	if(nInputs == 0 || w.size() != nInputs)
		return;
	
	weights.clear();
	weights.assign(w.begin(),w.end());		
}

void neuron::setInputs(vector<double> in) {
	if(nInputs == 0 || in.size() != nInputs)
		return;

	inputs.clear();
	inputs.assign(in.begin(),in.end());	
}

void neuron::setInput(double in) {
	inputs.push_back(in);
}

void neuron::clearInputs() {
	inputs.clear();
}

void neuron::setBias(double b) {
	bias = b;
}

void neuron::setThreshold(double off) {
	threshold = off;
}

void neuron::setActivationFunc(int af, double th) {
	AF = af;
	threshold = th;
}

void neuron::showData() {
	cout<<"Weights:";
	for (unsigned int i = 0; i < nInputs; ++i)
	{
		cout<<" "<<weights[i];
	}
	cout<<endl<<"Bias: "<<bias<<endl;


	cout<<"Inputs:";
	for (unsigned int i = 0; i < inputs.size(); ++i)
	{
		cout<<" "<<inputs[i];
	}
	cout<<endl<<"Output: "<<output<<endl;

	cout<<"Threshold: "<<threshold<<"\tAF: "<<AF<<endl;
}

double neuron::getOutput() {
	return output;
}

double neuron::compute() {
	for(unsigned int i = 0; i<nInputs; ++i)
		xw += inputs[i]*weights[i];
	xw -= bias;

	switch(AF) {
		case STEP:
		output = step(xw);
		break;
		case SLOP:
		output = slope(xw);
		break;
		case SIGM:
		output = sigmoid(xw);
		break;
		case BSIG:
		output = biSigmoid(xw);
		break;
		default:
		output = 0;
	}

	return output;
}

void neuron::setExamples(int n, vector<vector<double> > ex, vector<double> targ) {
	nExamples = n;
	if(nExamples == 0 || ex.size() != nExamples || targ.size() != nExamples)
		return;

	examples.clear();
	targets.clear();
	
	examples.reserve(n);
	for (unsigned int i = 0; i < nExamples; ++i)
	{
		examples.push_back(ex.at(i));
	}

	targets.reserve(n);
	targets.assign(targ.begin(),targ.end());
}

void neuron::showExamples() {
	for (unsigned int i = 0; i < nExamples; ++i)
	{
		cout<<"Example: "<<i<<endl;
		for (unsigned int j = 0; j < nInputs; ++j)
		{
			cout<<" "<<examples.at(i).at(j);
		}
		cout<<" "<<targets.at(i)<<endl;
	}
}

void neuron::trainNeuron(int nIt) {
	for (unsigned int j = 0; j < nInputs; ++j)
	{
		cout<<" "<<weights.at(j);
	}
	cout<<endl<<"Bias: "<<bias<<endl<<endl;

	int n = 0;
	double error = 0.0;
	double y;

	while(n < nIt) {
		cout<<"Epoch: "<<n<<endl;
		for (unsigned int i = 0; i < nExamples; ++i)
		{
			xw = 0.0;
			for (unsigned int j = 0; j < nInputs; ++j)
			{
				xw += examples.at(i).at(j)*weights.at(j);
			}

			xw -= bias;

			switch(AF) {
				case STEP:
				y = step(xw);
				break;
				case SLOP:
				y = slope(xw);
				break;
				case SIGM:
				y = sigmoid(xw);
				break;
				case BSIG:
				y = biSigmoid(xw);
				break;
				default:
				y = 0;
			}

			error = ALPHA*(targets.at(i) - y);

			switch(AF) {
				case STEP:
				y = dStep(xw);
				break;
				case SLOP:
				y = dSlope(xw);
				break;
				case SIGM:
				y = dSigmoid(xw);
				break;
				case BSIG:
				y = dBiSigmoid(xw);
				break;
				default:
				y = 0;
			}

			error *= y;

			for (unsigned int j = 0; j < nInputs; ++j)
			{
				weights.at(j) += examples.at(i).at(j)*error;
			}
			bias -= error;

			cout<<"Weights:";
			for (unsigned int j = 0; j < nInputs; ++j)
			{
				cout<<" "<<weights.at(j);
			}
			cout<<endl<<"Bias: "<<bias<<endl;
			cout<<"Error: "<<error<<endl<<endl;
		}
		n++;
	}
}

void neuron::randomWeights() {
	srand(time(NULL));

	weights.clear();

    for (unsigned int i = 0; i < nInputs; ++i)
    {
        weights.push_back((rand()%11)/10.0);
    }

    bias = (rand()%11)/10.0;
    //bias = 1.0;
}

//Funciones de Activacion
double neuron::step(double y) {
	double o;

	if(y-threshold < 0)
		o = 0.0;
	else
		o = 1.0;

	return o;
}

double neuron::slope(double y) {
	double o;

	o = threshold*y;

	return o;
}

double neuron::sigmoid(double y) {
	double o;

	o = 1/(1+exp(-threshold*y));

	return o;
}

double neuron::biSigmoid(double y) {
	double o;

	o = (2/(1+exp(-threshold*y)))-1.0;

	return o;
}

//Derivadas
double neuron::dStep(double y) {
	double dAF;

	dAF = 1.0;
	return dAF;
}

double neuron::dSlope(double y) {
	double dAF;

	dAF = 1.0;
	return dAF;
}

double neuron::dSigmoid(double y) {
	double dAF;

	dAF = sigmoid(y)*(1 - sigmoid(y));
	return dAF;
}

double neuron::dBiSigmoid(double y) {
	double dAF;

	dAF = 0.5*(1.0 + biSigmoid(y))*(1.0 - biSigmoid(y));
	return dAF;
}