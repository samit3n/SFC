#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "NNet.h"
#include "TrainData.h"
#include <vector>
#include <cmath>
#include <getopt.h>

   
const char * HELP = 
"\n"
"    CPN neural network simulator recognizing\n"
"    handwritten digits from MNIST database\n"
"   -t train data file\n"
"   -l train labels file"
"   -T test data file\n"
"   -L test label files\n"
"   -i real data file (optional)\n"
"   -o output file (optional)\n";


 

int main(int argc, char **argv)
{

    char * trainF, *trainLabelF,*testF,* testLabelF, *inF,* outF;
    trainF = trainLabelF = testF= testLabelF = inF = outF = NULL;
    unsigned trDataLimit = 0;
    int c;
    
    //ARG PARSING 
    while ((c = getopt(argc, argv, "t:l:T:L:i:o:s:")) != -1){
        
        switch(c){
            case 't':   trainF = optarg;
                        break;
            case 'l':   trainLabelF = optarg;
                        break; 
            case 'T':   testF = optarg;
                        break;
            case 'L':   testLabelF = optarg;
                        break;
            case 'i':   inF = optarg;
                        break;
            case 'o':   outF = optarg;
                        break;
            case 's':   {std::istringstream ss(optarg);
                        if((ss >> trDataLimit).fail()){
                            std::cerr << "Err on arg \'s\'\n";
                            return 1;
                        }
                        break;
                        }
            default: std::cout << HELP;
        }
    }

    if(!trainF){
        std::cerr << "No train data given!\n";
        return 1;
    }
    if(!trainLabelF){
        std::cerr << "No train labels given!\n";
        return 1;
    }
    if(!testF){
        std::cerr << "No test data given!\n";
        return 1;
    }
    if(!testLabelF){
        std::cerr << "No test labels given!\n";
        return 1;
    }
    

    //NEURAL NETWORKING STUFF

    sfcTrainData data, testData, realData;
    std::ofstream outFile;
    data.ParseData(trainF, trainLabelF);
    testData.ParseData(testF, testLabelF);

    if (inF != NULL)
        realData.ParseData(inF, testLabelF); 
        //testLabelF only fake here to make things work
    if(outF != NULL){
        try{
            outFile.open(outF);
        }catch(std::ios_base::failure e){
            std::cerr << "Error opening file " << outF << "\n";
        }
    }
    
    if(trDataLimit > 0)
        data.SetLimit(trDataLimit, 10);

    std::vector<double> i, ti, in;
    std::vector<double> d, td, out;
    //network init
    unsigned lays[] = {161,10,10};
    sfcNeuralNet net(lays, 3);
    net.InitWeights(data); 
    net.TrainFOCPN(data, testData,1,1,0);
    net.TestNet(testData);
    
    if (inF != NULL){
        unsigned seq = 1;
        while(realData.GetTrainPair(in,d)){ //d is unused as well
            net.GetResponse(&in, &out);
            for (unsigned cnt = 0;cnt < out.size();cnt++){
                if(out[cnt] == 1){
                    if(outFile.is_open())
                        outFile << seq << ".\t" << cnt << "\n";
                    else
                        std::cout << seq << ".\t" << cnt << "\n";
                    seq++;
                }
            
            }
        }
    }

    return 0;
}
