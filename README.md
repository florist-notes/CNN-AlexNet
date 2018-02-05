# CNN-AlexNet
🕵🏻 Model 1: AlexNet : Image Classification

<b>Paper : </b>[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
<b>Talk :</b> [NIPS 2012](http://videolectures.net/machine_krizhevsky_imagenet_classification/?q=imagenet) ; <b>Slide :</b>[link](https://github.com/SKKSaikia/CNN-AlexNet/blob/master/nips/machine_krizhevsky_imagenet_classification_01.pdf)

2012 [ILSVRC](http://www.image-net.org/challenges/LSVRC/) ([ImageNet](http://www.image-net.org/) Large-Scale Visual Recognition Challenge) <b>Winner</b>.

It is a benchmark competition where teams across the world compete to classify, localize, detect ... images of 1000 categories, taken from the imagenet dataset.The imagenet dataset holds 15M images of 22K categories but for this contest: 1.2M images in 1K categories were chosen.Their goal was, Classification i.e, make 5 guesses about label for an image.Team "SuperVision (AlexNet) ", achieved top 5 test error rate of 15.4% ( next best entry achieved an error of 26.2% ) more than 10.8 percentage points ahead of the runner up. This was a huge success. Check ILSVRC 2012 [results](http://image-net.org/challenges/LSVRC/2012/results.html).

This paper is important as it stands as a stepping stone for CNNs in Computer Vision community. It was record breaking, new and exciting.

Overview
-
AlexNet is a Convolutional Neural Network architecture, introduced in 2012 by [Alex Krizhevsky](https://scholar.google.com/citations?user=xegzhJcAAAAJ), [Ilya Sutskever](https://scholar.google.com/citations?user=x04W_mMAAAAJ) and [Geoffrey Hinton](https://scholar.google.co.uk/citations?user=JicYPdAAAAAJ). It has 7 hidden weight layers & contains 
● 650,000 neurons
● 60,000,000 parameters
● 630,000,000 connections. In simple terms, it is a model to correctly classify images.

Architecture:
-
Alexnet contained <b>only 8 layers</b>, first 5 were convolutional layers followed by fully connected layers. It had max-pooling layers and dropout layers in between.
<img src="https://github.com/SKKSaikia/CNN-AlexNet/blob/master/img/arch-simple.jpg">

But wait,

What is Convolutional Layer ?
:
What is Fully Connected Layer ?
:
What is Max-pooling Layer ?
:
What is Dropout Layer ?
:
What is Normalization Layer ?
:

The Network had a very similar architecture to [LeNet](https://github.com/SKKSaikia/CNN-LeNet), but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer.

<img src="https://github.com/SKKSaikia/CNN-AlexNet/blob/master/img/arch.jpg">

model.summary():
-

    Input Image size : 227 x 227 x 3
    
    ● CONV1
    Output (from Conv1): 55 x 55 x 96  
    First Layer Conv1 has 96 11x11 filters at stride 4, pad 0
    
    Output (from Pool1): 27 x 27 x 96
    Max Pool 1 has 3 x 3 filter applied at stride 2
    
    Ouput ( from Normalization Layer ): 27 x 27 x 96
    
    ●CONV2

    Output (from Conv2): 27 x 27 x 256  
    Second Layer Conv2 has 256 5x5 filters at stride 1, pad 2
    
    Output (from Pool2): 13 x 13 x 256
    Max Pool 2 has 3 x 3 filter applied at stride 2
    
    Ouput ( from Normalization Layer ): 13 x 13 x 256
    
    ●CONV3
    
    Output (from Conv3): 13 x 13 x 384
    Third Layer Conv3 has 384 3x3 filters at stride 1, pad 1
    
    ●CONV4
    
    Output (from Conv4): 13 x 13 x 384
    Fourth Layer Conv4 has 384 3x3 filters at stride 1, pad 1
    
    ●CONV5
    
    Output (from Conv5): 13 x 13 x 256
    Fifth Layer Conv5 has 256 3x3 filters at stride 1, pad 1
    
    Output (from Pool3): 6 x 6 x 256
    Max Pool 3 has 3 x 3 filter applied at stride 2
    
    
    ●FC6
    Fully Connected Layer 6 : 4096 neurons
    
    ●FC7
    Fully Connected Layer 7 : 4096 neurons
    
    ●FC8
    Fully Connected Layer 7 : 1000 neurons ( class scores )
    
    
  
Practical:
-


Take Away:
-


References
-

    CS231n : Lecture 9 | CNN Architectures - AlexNet
  
  
