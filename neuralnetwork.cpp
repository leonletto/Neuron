#include "neuralnetwork.h"

NeuralNetwork() {
	nInputs = 0, nInputL = 0, nHiddenL = 0, nOutputL = 0;
	bias = 0;
	xw = 0;
	output = 0;
	AF = 0;
	treshold = 0;
	nExamples = 0;
}

NeuralNetwork(int nIn, int nIL , int inHL, int nOL) {
	nInputs = nIn, nInputL = nIL, nHiddenL = nHL, nOutputL = nOL;
	bias = 0;
	xw = 0;
	output = 0;
	AF = 0;
	treshold = 0;
	nExamples = 0;
	inputs.resize(nIn);
	inputLayer.resize(nIL);
	for (unsigned int i = 0; i < nInputL; ++i)
	{
		inputLayer.at(i).setSize(nInputs);
	}
	hiddenLayer.resize(nHL);
	for (unsigned int i = 0; i < nHiddenL; ++i)
	{
		inputLayer.at(i).setSize(nInputL);
	}
	outputLayer.resize(nOL);
	for (unsigned int i = 0; i < nOutputL; ++i)
	{
		inputLayer.at(i).setSize(nHiddenL);
	}
	ILOutpus.resize();
	HLOutpus.resize();
	OLOutpus.resize();
}

~NeuralNetwork() {

}

void setNetworkSetSize(int nIn, int nIL , int inHL, int nOL) {
	inputs.resize(nIn);
	inputLayer.resize(nIL);
	for (unsigned int i = 0; i < nInputL; ++i)
	{
		inputLayer.at(i).setSize(nInputs);
	}
	hiddenLayer.resize(nHL);
	for (unsigned int i = 0; i < nHiddenL; ++i)
	{
		inputLayer.at(i).setSize(nInputL);
	}
	outputLayer.resize(nOL);
	for (unsigned int i = 0; i < nOutputL; ++i)
	{
		inputLayer.at(i).setSize(nHiddenL);
	}
}

void setInputs(vector<double> in) {
	if(nInputs == 0 || in.size() != nInputs)
		return;

	inputs.clear();
	inputs.assign(in.begin(),in.end());
}

void setActivationFunc(int af, double th) {
	for (unsigned int i = 0; i < nInputL; ++i)
	{
		inputLayer.at(i).setActivationFunc(af,th);
	}
	hiddenLayer.resize(nHL);
	for (unsigned int i = 0; i < nHiddenL; ++i)
	{
		inputLayer.at(i).setActivationFunc(af,th);
	}
	outputLayer.resize(nOL);
	for (unsigned int i = 0; i < nOutputL; ++i)
	{
		inputLayer.at(i).setActivationFunc(af,th);
	}
}

void showData();

void setExamples(int n, vector<vector<double> > ex, vector<double> targ);

void trainNetwork(int nIt);

double computeNetwork();