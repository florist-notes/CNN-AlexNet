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
Later in 2014, Alex once again shows a unique way to parrallelize CNNs in his paper, "[One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)"

Architecture:
-
Alexnet contained <b>only 8 layers</b>, first 5 were convolutional layers followed by fully connected layers. It had max-pooling layers and dropout layers in between. A simple skeleton looks like :
<img src="https://github.com/SKKSaikia/CNN-AlexNet/blob/master/img/arch-simple.jpg">

But wait,

What are Convolutional, Fully Connected, Max-pooling (P), Dropout & Normalization (N) Layers ?
:Find the Answers in CS231n's [blog](https://cs231n.github.io/convolutional-networks/) on CNN.

The Network had a very similar architecture to [LeNet](https://github.com/SKKSaikia/CNN-LeNet), but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer). The Architecture can be summarized as :

        
    ( Image ) ->CONV1->P1->N1  ->CONV2->P2->N2 ->CONV3 ->CONV4 ->CONV5->P3 ->FC6 ->FC7 ->FC8 -> ( Label )

<img src="https://github.com/SKKSaikia/CNN-AlexNet/blob/master/img/arch.jpg">

But why does the architecture diagram in the paper looks so scary ? 

<img src="https://github.com/SKKSaikia/CNN-AlexNet/blob/master/img/alexnet.png">

It is because, the figure shoes training as well, training was done in 2 GPUs. One GPU runs the layer parts at the top of the figure while the other runs the layer parts at the bottom. The GPUs communicate only at certain layers. 

model.summary():
-

    Input Image size : 227 x 227 x 3
    
    ● CONV1
    Output (from Conv1): 55 x 55 x 96                      //55 = (227-11)/(4+1) = (Image size - Filter size)/stride+1
    First Layer Conv1 has 96 11x11x3 filters at stride 4, pad 0
    
    Output (from Pool1): 27 x 27 x 96
    Max Pool 1 has 3 x 3 filter applied at stride 2
    
    Ouput ( from Normalization Layer ): 27 x 27 x 96
    
    ●CONV2

    Output (from Conv2): 27 x 27 x 256  
    Second Layer Conv2 has 256 5x5x48 filters at stride 1, pad 2
    
    Output (from Pool2): 13 x 13 x 256
    Max Pool 2 has 3 x 3 filter applied at stride 2
    
    Ouput ( from Normalization Layer ): 13 x 13 x 256
    
    ●CONV3
    
    Output (from Conv3): 13 x 13 x 384
    Third Layer Conv3 has 384 3x3x256 filters at stride 1, pad 1
    
    ●CONV4
    
    Output (from Conv4): 13 x 13 x 384
    Fourth Layer Conv4 has 384 3x3x192 filters at stride 1, pad 1
    
    ●CONV5
    
    Output (from Conv5): 13 x 13 x 256
    Fifth Layer Conv5 has 256 3x3x192 filters at stride 1, pad 1
    
    Output (from Pool3): 6 x 6 x 256
    Max Pool 3 has 3 x 3 filter applied at stride 2
    
    
    ●FC6
    Fully Connected Layer 6 : 4096 neurons
    
    ●FC7
    Fully Connected Layer 7 : 4096 neurons
    
    ●FC8
    Fully Connected Layer 7 : 1000 neurons ( class scores )
    
    
Important Points:
-

    ● ReLU non linearity is applied to the output of every convolutional layer and fully connected layer.
    ● Rectified Linear Units (first use), overlapping pooling, dropout (0.5) trick to avoid overfitting
    ● Layer 1 (Convolutional) : 55*55*96 = 290,400 neurons & each has 11*11*3 = 363 weights and 1 bias i.e, 
      290400 * 364 = 105,705,600 paramaters on the first layer of the AlexNet alone!
    ● Training on multiple GPUs ( 2 NVIDIA GTX 580 3 GB GPU ) for 5-6 days.
      Top-1 and Top-5 error rates decreases by 1.7% & 1.2% respectively comparing to the net trained with 
      one GPU and half neurons!!
    ● Local Response Normalization
      Response normalization reduces top-1 and top-5 error rates by 1.4% and 1.2% , respectively.
    ● Overlapping Pooling ( s x z , where s < z ) compared to the non-overlapping scheme s = 2, z = 2
      top-1 and top-5 error rates decrease by 0.4% and 0.3%, respectively.
    ● Reducing Overfitting
      Heavy Data Augmentation!
        - 60 million parameters, 650,000 neurons (Overfits a lot.)
        - Crop 224x224 patches (and their horizontal reflections.)
        - At test time, average the predictions on the 10 patches.
    ● Reducing Overfitting - Dropout
    ● Stochastic Gradient Descent Learning
    ● batch size = 128
    ● 96 Convolutional Kernels ( 11 x 11 x 3 size kernels. ) :
        - top 48 kernels on GPU 1 : color-agnostic
        - bottom 48 kernels on GPU 2 : color-specific.
    ● In the paper, they say "Depth is really important.removing a single convolutional layer degrades
      the performance."
    
    

Practical:
-

| Net | Backend | Weights |
|--------|------------|---------|
|AlexNet|[Tensorflow](https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py)|[Weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)|
|AlexNet|[Caffe](http://dandxy89.github.io/ImageModels/alexnet/)|[Weights](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)|

Let's build the Alexnet in Keras ('tf' backend) and test it on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The script is available here : [alexnet-keras.py]()

Training on the CIFAR-10 dataset gives an accuracy of 78%

We can find other AlexNet trained datasets like the [MIT-places](http://places.csail.mit.edu/downloadCNN.html) dataset.

References
-

    CS231n : Lecture 9 | CNN Architectures - AlexNet
  
  
