#include <iostream>
#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork() {
	neuron nIn(IN);
	nIn.setActivationFunc(AF,TH);
	neuron nHid(IL);
	nHid.setActivationFunc(AF,TH);
	neuron nOut(HL);
	nOut.setActivationFunc(AF,TH);

	nExamples = 0;
	inputLayer.assign(IL, nIn);
	hiddenLayer.assign(HL, nHid);
	outputLayer.assign(OL, nOut);
	inputs.assign(IN,0.0);
	//ILOutputs.assign(IL,0.0);
	//HLOutputs.assign(HL,0.0);
	//OLOutputs.assign(OL,0.0);
}

NeuralNetwork::~NeuralNetwork() {

}

void NeuralNetwork::setInputs(vector<double> in) {
	if(in.size() != IN)
		return;

	inputs.assign(in.begin(),in.end());
}

void NeuralNetwork::setInput(double in) {
	if(inputs.size() >= IN)
		return;
	inputs.push_back(in);
}

void NeuralNetwork::clearInputs() {
	inputs.clear();
}

void NeuralNetwork::showData() {
	cout<<"Input Layer:"<<endl;
	for (unsigned int i = 0; i < inputLayer.size(); ++i)
	{
		inputLayer.at(i).showData();
	}
	cout<<endl;
	cout<<"Hidden Layer:"<<endl;
	for (unsigned int i = 0; i < hiddenLayer.size(); ++i)
	{
		hiddenLayer.at(i).showData();
	}
	cout<<endl;
	cout<<"Output Layer:"<<endl;
	for (unsigned int i = 0; i < outputLayer.size(); ++i)
	{
		outputLayer.at(i).showData();
	}
	cout<<endl;
	cout<<"Inputs:"<<endl;
	for (unsigned int i = 0; i < inputs.size(); ++i)
	{
		cout<<" "<<inputs.at(i);
	}
	cout<<endl;
}

void NeuralNetwork::setExamples(int n, vector<vector<double> > ex, vector<double> targ) {
	nExamples = n;
	if(nExamples == 0 || ex.size() != nExamples || targ.size() != nExamples)
		return;
	
	examples.reserve(n);
	examples.assign(ex.begin(),ex.end());

	targets.reserve(n);
	targets.assign(targ.begin(),targ.end());
}

void NeuralNetwork::trainNetwork(int nIt) {

}

double NeuralNetwork::computeNetwork() {

}