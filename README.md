# CNN-AlexNet
🕵🏻 Model 1: AlexNet : Image Classification

<b>Paper : </b>[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
<b>Talk :</b> [NIPS 2012](http://videolectures.net/machine_krizhevsky_imagenet_classification/?q=imagenet) ; <b>Slide :</b>[link](https://github.com/SKKSaikia/CNN-AlexNet/blob/master/nips/machine_krizhevsky_imagenet_classification_01.pdf)

2012 [ILSVRC](http://www.image-net.org/challenges/LSVRC/) ([ImageNet](http://www.image-net.org/) Large-Scale Visual Recognition Challenge) <b>Winner</b>.

It is a benchmark competition where teams across the world compete to classify, localize, detect ... images of 1000 categories, taken from the imagenet dataset.The imagenet dataset holds 15M images of 22K categories but for this contest: 1.2M images in 1K categories were chosen.Their goal was, Classification i.e, make 5 guesses about label for an image.Team "SuperVision (AlexNet) ", achieved top 5 test error rate of 15.4% ( next best entry achieved an error of 26.2% ). This was a huge success. Check ILSVRC 2012 [results](http://image-net.org/challenges/LSVRC/2012/results.html).

AlexNet is a CNN architecture introduced in 2012 by [Alex Krizhevsky](https://scholar.google.com/citations?user=xegzhJcAAAAJ), [Ilya Sutskever](https://scholar.google.com/citations?user=x04W_mMAAAAJ) and [Geoffrey Hinton](https://scholar.google.co.uk/citations?user=JicYPdAAAAAJ).

Architecture:
-
<img src="https://github.com/SKKSaikia/CNN-AlexNet/blob/master/img/arch.jpg">

It has 5 Convolution layers,max-pooling layers, dropout layers and 3 Fully connected layers.
