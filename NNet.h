#ifndef NNET_H
#define NNET_H

#include <vector>
#include "TrainData.h"

//const int LAYERCNT = 3;


/* 
 * Structure modeling single neuron
 */

//solitury function
bool CmpVec(std::vector<double> &v1, std::vector<double> &v2, double eps=0.2);

struct sfcNeuron
{
    
    std::vector<double> weights1;
    double sum;
 
    void PrintWeights(void);;
    int InitWeightsOut(unsigned weig1_Cnt);
    int InitWeightsHid(unsigned weig1_Cnt);
    int UpdHidden(std::vector<double> *i, double alpha=1, double eps=0.001);

    int UpdOutput(std::vector<double> *d, unsigned winner, unsigned index, double beta = 1, double eps =  0.001);
    
};

/*
 * Basic CPN network modeling class
 */

class sfcNeuralNet
{


public:
    
    sfcNeuralNet(unsigned neuPerLayer[], unsigned layers);
    double * GetWeights(void) const;
    void PrintWeights(void);
//    int TrainForwOnly();
    int TrainFOCPN(sfcTrainData &tData,sfcTrainData &testData, double alpha, double beta, double eps);
    int InitWeights(sfcTrainData &data); 
    int GetResponse(std::vector<double> *input, std::vector<double> *output); 
    double TestNet(sfcTrainData &testData);


private:

//    unsigned neuronCnt[3];


    std::vector< std::vector<sfcNeuron> > net;
    double VecDistance(std::vector<double> *i, std::vector<double> *w1);
    double palpha, pbeta;

    //tolerance for settled network
    double epsilon;
    unsigned inputLayer;
    unsigned OUTPUT;
};

#endif
