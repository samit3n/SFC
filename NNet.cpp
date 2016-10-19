/*
 * =====================================================================================
 *
 *       Filename:  NNet.cpp
 *
 *    Description:  CPN neural network implementation
 *                  Part of project for Soft Computing course
 *
 *        Version:  1.0
 *        Created:  10/11/14 20:32:54
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Vojtech Dvoracek (), xdvora0y@stud.fit.vutbr.cz
 *   Organization:  FIT BUT
 *
 * =====================================================================================
 */

#include <iostream>
#include "NNet.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "TrainData.h"
#include <algorithm>
#include <limits>

/*
 * Printing weights of single neuron
 */

void sfcNeuron::PrintWeights(void)
{
    std::cout << "Weight vector 1" << std::endl << "[";
    
    for(std::vector<double>::const_iterator it = weights1.begin(); it != weights1.end();++it){
        std::cout << *it << ", ";
    }
    
    std::cout << "]\n";
}

/*
 * Updates hidden layer neuron weighs by delta rule
 * @par1: input data vector
 * @par2: alpha learnign coef. (default 1, decreasing)
 * @par2: threshold of weight change
*/

int sfcNeuron::UpdHidden(std::vector<double> *i, double alpha, double eps)
{
    
    double tmp;
    unsigned changed = 0;
    
    
    for(unsigned j = 0; j < weights1.size();j++){

        tmp = weights1[j] + alpha*( (*i)[j] - weights1[j] );

        if ( std::fabs(tmp - weights1[j] ) <  eps)
            changed++; //value is settled
    
        weights1[j] = tmp;
    }

    return changed;
}

/*
 * Updates output layer neuron weighs by delta rule
 * @par1: output data vector
 * @par2: beta learnign coef. (default 1, decreasing)
 * @par2: threshold of weight change
*/
int sfcNeuron::UpdOutput(std::vector<double> *d, unsigned winner, unsigned index, double beta, double eps)
{

    double tmp;
    unsigned changed = 0;
    
    //WEIGHT vector 1
    tmp = weights1[winner] + beta*( (*d)[index] - weights1[winner] );
    
    if ( std::fabs(tmp - weights1[winner]) < eps )
        changed++; //value is settled
    
    weights1[winner] = tmp;
    
    return changed;
}

/*
 * Init weights in neuron of hidden layer
 * @par1: desired number of weights (one per input)
 */


int sfcNeuron::InitWeightsHid(unsigned weig1_Cnt)
{

    int wSum;
    weights1.resize(weig1_Cnt);

    for (int i = 0; i < weig1_Cnt;i++){
    
       weights1[i] = double( rand() ) / RAND_MAX;
            
        wSum++;
    }
    
    return wSum;
}

/*
 * Init weights in neuron of output layer
 * @par1: desired number of weights (one per input)
 */
int sfcNeuron::InitWeightsOut(unsigned weig1_Cnt)
{
    int wSum;
    weights1.resize(weig1_Cnt);

    for (int i = 0; i < weig1_Cnt;i++){
    
        weights1[i] = double( rand() ) / RAND_MAX;
            
        wSum++;
    }
  
    return wSum;
}
    

/***********************************
 * NEURAL NET methos
 **********************************/

sfcNeuralNet::sfcNeuralNet(unsigned neuPerLayer[], unsigned layers)
{

    inputLayer = neuPerLayer[0]; 
    
    net.resize(layers-1);  //omitting input layer
    
    for(int i = 0 ; i < layers-1;i++){
        net[i].resize(neuPerLayer[i+1]);
    }

    epsilon = 0.001;
    OUTPUT = net.size() -1 ; //OUTPUT layer index
    //std::cout << OUTPUT << "\n";
}

/*
 * Init wghts in whole network
 * Calss sfcNeuron::InitWeigh.. methods
 * @par1: training data
 */
int sfcNeuralNet::InitWeights(sfcTrainData &data)
{
    srand(time(NULL));
    unsigned sum;
    unsigned wghtsCnt;
    
    //init in hidden and output layer
    //inits hidden layer to input data samples
    //improves learning
    
    std::vector<double> x,y;
    std::vector<char> touch;
    touch.assign(net[0].size(),0);
    int cls, tSum = 0;

    while(data.GetTrainPair(x,y)){
        for(int c = 0;c < (net[0].size() /10) ;c++ ){
        
            for(unsigned j = 0; j < y.size();j++){
                    if (y[j] == 1){
                        cls = j+c*10;
            }
            if (touch[cls] == 0){
                if (x.size() != inputLayer){
                    std::cerr << "Input vector doesn't match input neurons\n";
                    return 1;
                }
                net[0][cls].weights1.assign(x.begin(),x.end());
                touch[cls] = 1;
            }
            for(std::vector<char>::iterator it = touch.begin();it != touch.end();it++)
                tSum += *it;

            if (tSum == net[0].size())
                break;
            else
                tSum = 0;
            }
        }
    }
    data.Reset();

    for(unsigned lay = 1; lay < net.size();lay++){
        
        wghtsCnt = net[lay -1].size();
        for(unsigned neu = 0; neu < net[lay].size();neu++){
            net[lay][neu].InitWeightsOut(wghtsCnt);
        }
    }

    return 0;
}

void sfcNeuralNet::PrintWeights(void)
{
    
    std::cout << "Input layer: " << inputLayer << " neurons" << std::endl; 
    for(int lay = 0; lay < net.size();lay++){
        std::cout << "====== Layer " << lay << " ====" << std::endl;
        for(int neu = 0; neu < net[lay].size();neu++){
            std::cout << "====== Neuron " << neu << " ====" << std::endl;
            net[lay][neu].PrintWeights();
        }
    }
}


//++++++++++++  NETWORK TRAINING +++++++++++++++++++

/*
 * Computes euclidean vecotr distance
 * @pars: std::vector pointers of doubles
 * ret:  double distance
 */
double sfcNeuralNet::VecDistance(std::vector<double> *v1, std::vector<double> *v2)
{

    if( v1->size() != v2->size()){
        std::cout << v1->size() << std::endl;
        std::cout << v2->size() << std::endl;
    
        throw 1;
    }

    double sum1 = 0;
    double tmp;
    
    for(unsigned j = 0; j < v1->size();j++){
        tmp = (*v1)[j] - (*v2)[j];
        sum1 += tmp * tmp;
    }

    return sqrt(sum1);
}

/*
 * Compares two double vectors
 * @par3: tolerance
 */
bool CmpVec(std::vector<double> &v1, std::vector<double> &v2, double eps)
{
    
    int match = true;
    
    if (v1.size() != v2.size())
        return false;
    
    for (unsigned j = 0; j < v1.size();j++){

        if (v1[j] != v2[j]){
            match = false;

            if ( std::fabs(v1[j] - v2[j]) < eps){
                 match = true;    
            }
        }
    }
    
    return match;
}
/*
 * Computes response to given input
 * @par1: input
 * @par2: network response
 */
int sfcNeuralNet::GetResponse(std::vector<double> *input, std::vector<double> *output)
{
    //hidden layer competition
    
    double min = -1;
    unsigned winner;
    output->clear();
    double dist;
    double sum;

    for(unsigned lay = 0; lay < net.size();lay++){

        if (lay == OUTPUT){
            for(unsigned n = 0; n < net[lay].size();n++){
                if (net[lay][n].weights1[winner] < 0.5)
                    output->push_back(0);
                else
                    output->push_back(1);
            }
        }else{
            for(unsigned j = 0; j < net[lay].size();j++){

                dist = VecDistance(input,&net[lay][j].weights1);
            
                if (min < 0){
                    min = dist;       
                    winner = j;

                }else if ( dist < min){
                    min = dist;
                    winner = j;
                }
            }
        }

        //output layer computation
    }

    return 0;    
}
/*
 * Testing network response by dataset
 * @par1: test data set
 */
double sfcNeuralNet::TestNet(sfcTrainData &testData)
{
    unsigned testOk = 0, testCnt = 0;
    double succ;


    std::cout << "Testing network on testset\n";
    
    std::vector<double> ti, td, out;
 
    while (testData.GetTrainPair(ti, td) ){
         
        GetResponse(&ti, &out);
        
        if (CmpVec(out, td,0)){
            testOk++;   
        }
        testCnt++;
    }
    testData.Reset();
    succ = (double(testOk)/testCnt)*100;

    std::cout << "Success " << testOk << " of " << testCnt << " " << succ << "%" << std::endl;
    
    return succ;
       
}
/*
 * Forward onyl CPN training
 * @par1: training data reference
 * @par2: testData ref
 * @par3,4: learn. coeficients
 * @par5:  tolerance
 */

int sfcNeuralNet::TrainFOCPN(sfcTrainData &tData,sfcTrainData &testData,  double alpha, double beta, double eps)
{

    unsigned bestNeuHid, bestNeuOut;

    //vector variables
    std::vector<double> i;
    std::vector<double> d;


    //competition variables
    double distance;
    double shortest = -1;
    unsigned winner = 5;
    //testing variables
    unsigned cnt = 0;
    double succ = 0, lastSucc = -1;

    
    //debug    
    std::vector<int> wcnt;
    std::vector<unsigned> clses;
    clses.assign(10,0);
    wcnt.assign(net[0].size(),0);
    int cls;
    unsigned usedPairs = 0;
    unsigned tmp1 = 0, tmp2 = 0;

    
    for (int lay = 0; lay < net.size();lay++){
        std::cout << "========================\nTraining network layer " << lay << "\n========================\n";
        
        while( tData.GetTrainPair(i,d, &cls) ){
            if (cnt % 5000 == 0)
                std::cout << "Training " << cnt << "\n"; 

            clses[cls]++;
            cnt++;
            shortest = -1;
            
            for(unsigned r = 0; r < net[lay].size();r++){ //all neus in layer
    
                try{
                    if(lay == OUTPUT){
                        distance = VecDistance(&i,&net[lay-1][r].weights1);
                    }else{
                        distance = VecDistance(&i,&net[lay][r].weights1);
                    }
                }catch(int e){
    
                    std::cout << "Caught " << e << " VecDistance: vector dim. does not match" << std::endl;
                    std::cout << "x " << d.size() << "y " << net[lay][r].weights1.size() << std::endl;
                    return 1;
                }
                   
                if (shortest  < 0){
                   shortest = distance;   
                }else if ( distance < shortest) { //finding minimal distance as inverse value
                   shortest = distance;
                   winner = r;
                }   
            }
            if (winner == cls)      
                tmp1++;
            tmp2++;
            wcnt[winner]++;
    
            if(lay == OUTPUT ){ //OUTPUT layer
                for(unsigned x = 0; x < net[lay].size();x++){

                    net[lay][x].UpdOutput(&d, winner,x, beta, eps);
                }
                beta *= 0.9;
            }else{ //HIDDEN layer
                
                net[lay][winner].UpdHidden(&i, alpha, eps);
                alpha *= 0.9;
            }
        }
        

        tData.Reset(); //rewind train set to beginingi
        cnt = 0;
    }
     
    std::cout << "Assign: total " << tmp2 << " correct " << tmp1 << "\n"; 
    return 0;
}


