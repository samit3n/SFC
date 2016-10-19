SFC Project
===========

Author: Vojtech Dvoracek

This project implements classification of handwritten digits using
forward-only Counterpropagation network.

Running:

 1. make
 2. run
     1. ./SFC -t \<training data\> -l \<training labels\> -T \<testing data\> -L \<testing labels\>
     2. ./SFC -t \<training data\> -l \<training labels\> -T \<testing data\> -L \<testing label\> -i \<real data\> -o \<classification output\>

* Option 2.1 will train and test given model.
* Option 2.2 will train, test and evaluate real data.

Datasets can be obtained from http://yann.lecun.com/exdb/mnist/.
They are part of MNIST handwritten digits competition project.

