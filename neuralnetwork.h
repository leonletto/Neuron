#include "neuron.h"

#define IL 1
#define HL 1
#define OL 1

class NeuralNetwork {
public:
	NeuralNetwork();
	NeuralNetwork(int nIn, int nIL , int inHL, int nOL);
	~NeuralNetwork();
	void setNetworkSetSize(int nIn, int nIL , int inHL, int nOL);
	void setInputs(vector<double> in);
	void setActivationFunc(int af, double th);
	void showData();
	void setExamples(int n, vector<vector<double> > ex, vector<double> targ);
	void trainNetwork(int nIt);
	double computeNetwork();

private:
	unsigned int nInputs;
	int AF;
	unsigned int nExamples;
	double bias;
	vector<neuron> inputLayer;
	vector<neuron> hiddenLayer;
	vector<neuron> outputLayer;
	vector<double> inputs;
	vector<vector<double> > examples;
	vector<double> targets;
	vector<double> ILOutputs;
	vector<double> HLOutputs;
	vector<double> OLOutputs;
	double threshold;
};