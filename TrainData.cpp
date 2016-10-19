/*
 * =====================================================================================
 *
 *       Filename:  TrainDataMNIST.cpp
 *
 *    Description:  MNIST training data handling class. 
                    Includes feature extraction method.
 *
 *        Version:  1.0
 *        Created:  06/12/14 19:58:50
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Vojtech Dvoracek (), xdvora0y@stud.fit.vutbr.cz
 *   Organization:  FIT VUT BRNO
 *
 * =====================================================================================
 */

#include "TrainData.h"
#include <iostream>
#include <cstdio>
#include <algorithm>

sfcTrainData::sfcTrainData()
{
    dClass = 0;
    position = 0;
    sampleLimit = 0;
    
}
/*
 * Set amount of training data used from
 * training set
 * Equal amount from all categories is used
 * if limit == 0, all data are used
 * @par1: limit
 * @par2: number of classes
 */
void sfcTrainData::SetLimit(unsigned count, unsigned classes)
{
    sampleLimit = count;
    cntPerClass.assign(classes,0);
}

/*
 * MNIST data parsing method
 * works above MNISTParser
 * @par1,2: image and label filenames - passing on
 * */

int sfcTrainData::ParseData(const char * images, const char * labels)
{
    
    if (mnist.Parse(images, labels) != 0){
        std::cout << "Mnist parsing error" << std::endl;
        return 1;
    }

    dataSize = mnist.GetImageSize();
    data = mnist.GetImageData();
    metaData = mnist.GetCategoryData();
    imgCount = mnist.GetImageCount();
    width = mnist.GetImageWidth();
    height = mnist.GetImageHeight();
    
    
    return 0;    
}
/*
 * Main method for input data handling
 * Provides feature extract as well
 *
 * @par1: for input data store
 * @par2: for output data
 * @par3: pointer - class of data will be stored in
 */

int sfcTrainData::FeatExtract(std::vector<double> &i, const float * tmp)
{


    unsigned w,h, sum = 0;
    int last_edge;
    unsigned edge = 0;


    /*
     * 2D - Histogram generation
     * horizontal - amount of non-zero vals in row
     * vertial - the same in col
     * counting of edges in earch row and column
     */ 
    for(h = 0; h < height;h++ ){
        for(w = 0;w < width;w++){
            if ( *(tmp+h*width+w) != 0 ){
                sum += 1;
                if (last_edge == 0){
                    edge++;
                    last_edge = 1;
                }
            }else{
                if (last_edge == 1){
                    edge++;
                    last_edge = 0;
                }
            }
        }
        i.push_back(sum);
        i.push_back(edge);
        edge = 0;
        sum = 0;
    }
    
    for(w = 0; w < width;w++ ){
        for(h = 0;h < height;h++){
            if ( *(tmp+h*width+w) != 0 ){
                sum += 1;
                if (last_edge == 0){
                    edge++;
                    last_edge = 1;
                }
            }else{
                if (last_edge == 1){
                    edge++;
                    last_edge = 0;
                }
            }   
        }
        i.push_back(sum);
       i.push_back(edge);
        edge = 0;
        sum = 0;
    }

   
    sum = 0;
/*
 * creates sum of black pixel
 * in every 4-pixel square
 */ 
    for(unsigned x = 0; x < 7;x++){
        for(unsigned y = 0; y < 7;y++){
            for(w = 0;w < 4;w++){
                for(h = 0; h < 4;h++){ 
                    if ( *(tmp+h+w*width+x*7+y*7) != 0 ){
                        sum ++;
                   }
                }
            }
            i.push_back(sum);
            sum = 0;
        }
    }

        
    //input vector normalization
    //divides by highest possible value so creates <0,1>
    for (unsigned j = 0; j < i.size();j++){
        i[j] = i[j] / 28;
    }

    return 0;

}

bool sfcTrainData::GetTrainPair(std::vector<double> & i, std::vector<double> & d, int * cls)
{
    if (position == imgCount -1){
        return false; //out of training data    
    }
        
    i.clear();
    const float * tmp = &data[position*dataSize];
    FeatExtract(i, tmp);
    
    //feature extraction


    d.assign(10,0);
    d.erase(d.begin() + metaData[position]);
    d.insert(d.begin() + metaData[position], 1);
    
    if (cls != NULL){
        *cls = metaData[position];
    //     std::cout << *cls << ",";
    }

    bool flag = true;
    
    if (sampleLimit != 0){
        cntPerClass[(char) metaData[position]]++;
    
        for (std::vector<unsigned>::iterator it = cntPerClass.begin(); it != cntPerClass.end();++it){
            if(*it < sampleLimit)
                flag = false;
        }
        if(flag == true)
            return false;
    }
    position++;
   
    return true;
    
}

void sfcTrainData::Print(int item) const
{
    
    if (item == -1)
        item = position;

    const float * tmpdata = &data[item*dataSize];
    std::cout << "Class: ";
    printf("%d\n",metaData[item]);

    for (unsigned i = 0; i < height;i++){
        for (unsigned j = 0; j < width;j++){
        
            printf("%3d", (uint8_t) tmpdata[i*width+j]);

       }
       printf("\n");
    }
}

void sfcTrainData::Reset()
{
    position = 0; //resets sample pointer to beginning
    cntPerClass.assign(cntPerClass.size(),0);
}


