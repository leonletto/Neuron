#include "neuron.h"

#define IL 4
#define HL 8
#define OL 1
#define IN 2
#define TH 1.0
#define AF SLOP

class NeuralNetwork {
public:
	NeuralNetwork();
	~NeuralNetwork();
	void setInputs(vector<double> in);
	void setInput(double in);
	void clearInputs();
	void showData();
	void setExamples(int n, vector<vector<double> > ex, vector<double> targ);
	void trainNetwork(int nIt);
	double computeNetwork();

private:
	unsigned int nExamples;
	vector<neuron> inputLayer;
	vector<neuron> hiddenLayer;
	vector<neuron> outputLayer;
	vector<double> inputs;
	vector<vector<double> > examples;
	vector<double> targets;
	vector<double> ILOutputs;
	vector<double> HLOutputs;
	vector<double> OLOutputs;
};