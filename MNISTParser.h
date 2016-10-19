/*
    Copyright 2014 Henry Tan
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
    http ://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef MNIST_PARSER
#define MNIST_PARSER

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory>
#include <assert.h>

#define nullptr NULL


//
// C++ MNIST dataset parser
// 
// Specification can be found in http://yann.lecun.com/exdb/mnist/
//
class MNISTDataset final
{
public:

	
    MNISTDataset();
    ~MNISTDataset();

    void Print();
    size_t GetImageWidth() const;
    size_t GetImageHeight() const;
    size_t GetImageCount() const;
    size_t GetImageSize() const;
    const float* GetImageData() const;
    const uint8_t* GetCategoryData() const;

    //
    // Parse MNIST dataset
    // Specification of the dataset can be found in:
    // http://yann.lecun.com/exdb/mnist/
    //
    int Parse(const char* imageFile, const char* labelFile);

private:
    void Initialize(const size_t width, const size_t height, const size_t count);

    // The total number of images
    size_t m_count;

    // Dimension of the image data
    size_t m_width;
    size_t m_height;
    size_t m_imageSize;

    float* m_imageBuffer;

    static const int c_categoryCount = 10;

    // 1-of-N label of the image data (N = 10) 
    uint8_t* m_categoryBuffer;

    // The entire buffers that stores both the image data and the category data
    float* m_buffer;
};

#endif
