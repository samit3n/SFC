/*
 * =====================================================================================
 *
 *       Filename:  TrainDataMNIST.h
 *
 *    Description:  MNIST training data handling class.
 *
 *        Version:  1.0
 *        Created:  06/12/14 19:59:00
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Vojtech Dvoracek (), xdvora0y@stud.fit.vutbr.cz
 *   Organization:  FIT VUT BRNO
 *
 * =====================================================================================
 */


#ifndef SFC_DATAPROC
#define SFC_DATAPROC

#include <vector>
#include "MNISTParser.h"


const int DATA_SIZE_CHANGED = -1;
const int PARSING_ERROR = -2;

class sfcTrainData
{
//training data handling
//
public:
    
    sfcTrainData();
    int ParseData(const char * images, const char * labels);
    bool GetTrainPair(std::vector<double> &i, std::vector<double> &d, int * cls=NULL );
    int FeatExtract(std::vector<double> &i, const float * tmp);
    void Reset(void);
    void Print(int item = -1) const;
    void SetLimit(unsigned count, unsigned classes);

private:
    
    //matrix of filenames
    //rows represents class of data, cols represents filenames
    
    
    MNISTDataset mnist;
    
    //data position coordinates as matrix indices
    // 1,2 1st class 2nd vector -> data[0][1]
    //

    const float * data; //image data
    const uint8_t * metaData; //image classes
    unsigned imgCount;

    unsigned width, height;

    unsigned sampleLimit;
    std::vector<unsigned> cntPerClass;
    

    unsigned dClass;
    unsigned position;

    unsigned dataSize; //train vector size
    

};

#endif
