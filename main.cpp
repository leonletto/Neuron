#include <iostream>
//#include "neuron.h"
#include "neuralnetwork.h"

using namespace std;

int main() {
	/*vector<double> a,b,t;
	//Inputs
	a.push_back(0.1);
	a.push_back(1.0);
	//Weights
	b.push_back(0.5);
	b.push_back(0.3);

	t.reserve(2);
	vector<vector<double> > e;
	e.reserve(5);
	t.push_back(0.1);
	t.push_back(1.0);
	e.push_back(t);
	t.clear();

	t.push_back(1.0);
	t.push_back(0.5);
	e.push_back(t);
	t.clear();

	t.push_back(0.7);
	t.push_back(0.3);
	e.push_back(t);
	t.clear();

	t.push_back(0.1);
	t.push_back(0.1);
	e.push_back(t);
	t.clear();

	t.push_back(0.2);
	t.push_back(0.1);
	e.push_back(t);
	t.clear();

	t.reserve(5);
	t.push_back(0.537);
	t.push_back(0.610);
	t.push_back(0.559);
	t.push_back(0.47);
	t.push_back(0.4825);

	neuron n1(2);
	//n1.showData();

	n1.setActivationFunc(SIGM,1.0);

	n1.setExamples(5, e, t);
	n1.showExamples();

	int nIt;

	cout<<"Number of iterations: ";
	cin>>nIt;
	n1.randomWeights();
	n1.trainNeuron(nIt);

	//n1.setWeights(b);
	//n1.setInputs(a);
	//n1.setBias(0.2);
	n1.showData();

	n1.clearInputs();
	n1.setInput(0.1);
	n1.setInput(1.0);

	n1.compute();
	n1.showData();*/

	NeuralNetwork nn1;
	nn1.showData();

	return 0;
}