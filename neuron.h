#include <vector>

#define STEP 1
#define SLOP 2
#define SIGM 3
#define BSIG 4

#define ALPHA 0.2

using namespace std;

class neuron {
public:
	neuron();
	neuron(int n);
	neuron(int n, vector<double> w, double b);
	~neuron();
	void setSize(int n);
	void setWeights(vector<double> w);
	void setInputs(vector<double> in);
	void setInput(double in);
	void clearInputs();
	void setBias(double b);
	void setThreshold(double off);
	void setActivationFunc(int af, double th);
	void showData();
	double getOutput();
	double compute();
	void setExamples(int n, vector<vector<double> > ex, vector<double> targ);
	void showExamples();
	void randomWeights();
	void trainNeuron(int nIt);

private:
	double step(double y);
	double slope(double y);
	double sigmoid(double y);
	double biSigmoid(double y);
	double dStep(double y);
	double dSlope(double y);
	double dSigmoid(double y);
	double dBiSigmoid(double y);

	unsigned int nInputs;
	int AF;
	unsigned int nExamples;
	double bias;
	vector<double> inputs;
	vector<double> weights;
	vector<vector<double> > examples;
	vector<double> targets;
	double xw;
	double output;
	double threshold;
};