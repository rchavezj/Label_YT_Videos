# Classifying Video Labels from Youtube 

Deep Learning algorithm (Deep Neural Net + LSTM) to label a genre on a youtube video. Used deep learning (PyTorch & Keras) to extract spatial (pixels) and sequential strings (audio). Later concatenate onto a fully connected network to output the video label genre (E.g. Games, Art & Entertainment, etc.). Link to [my paper](https://github.com/rchavezj/Label_YT_Videos/blob/master/Paper.pdf).

![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/resource/feature_engineering.png)

## [Wandb Results](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report)

Below are graphs I gathered from the app wandb (Weights & Biasis) to monitor the performance of each deep learning algorithm coded in pytorch. If you wish to visualize the performance, [click here!](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report) I also done experiments for the same models in [keras](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FKeras%20Report). 

![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_1.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_2.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_3.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_4.png)

<img src="https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_cpu.png" width="440" height="275"> <img src="https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_gpu.png" width="440" height="275">


### PyTorch Report
|                                  | Loss     |  Accuracy  | Learn Rate | Epoch | Batch Size |   GPU Usage  |   CPU Usage  |   System Memory  |
| :---:                            |  :----:  |    :---:   |  :----:    | :---: |    :---:   |    :----:    |    :----:    |    :----:        |
| Neural Net                       |  1.75%    |    45%    |    0.01    | 300   |     30     |    29%       |    15.31%    |    13.89%        |
| Multi-Bidirectional LSTM         |  7.89%   |    64%     |    0.003   | 300   |     64     |    55%       |    12.65%    |    12.26%        |
| Stream LSTM                      |  7.98%   |    64%     |    0.03    | 300   |     64     |     0%       |     97%      |     70.60%       |
| Neual Net + Stream LSTM Concat   |  8.91%    |    95%    |  0.00045   | 300   |     64     |   24.13%     |   12.67%     |     16.49%       |

### Keras Report
|                                  | Loss     |  Accuracy  |  Epoch  | Batch Size |   GPU Usage  |   CPU Usage  |  System Memory  |
| :---:                            |  :----:  |    :---:   | :----:  |    :---:   |    :----:    |    :----:    |    :----:       |
| Neural Net                       |  68.5%   |    66%     |   300   |     84     |     0.13%    |    64.97%    |    14.37%       |
| Multi-Bidirectional LSTM         |  56%     |    77%     |   300   |     64     |     0%       |    99.17%    |    31.19%       |
| Stream LSTM                      |  58.6%   |    84%     |   500   |     64     |     0%       |    93.37%    |    26.07%       |
| Neual Net + Stream LSTM Concat   |  68.6%   |    73%     |   100   |     20     |     0%       |    96.84%    |    28.45%       |

## Neural Net with LSTM Stream
Documentations for my designs are available on [my paper](https://github.com/rchavezj/Label_YT_Videos/blob/master/Paper.pdf).
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/resource/nn_lstm_stream.png)


## Best Deep Learning Model: <br /> Neural Net + Stream LSTM Concatenated (Tensorboard)
Documentations for my designs are available on [my paper](https://github.com/rchavezj/Label_YT_Videos/blob/master/Paper.pdf).
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/code/tensorboard_images/graph_nn_stream_lstm.png)


## Download yt8m dataset
The total size of the frame-level features is 1.53 Terabytes. They are broken into 3844 shards which can be subsampled to reduce the dataset size. 

To download the Frame-level dataset using the download script, navigate your terminal to a directory where you would like to download the data. For example:

```console
mkdir -p ~/v2/yt8m/frame; cd ~/v2/yt8m/frame
```
Then download the training and validation data. Note: Make sure you have 1.53TB of free disk space to store the frame-level feature files. Download the entire dataset as follows:

```console
curl data.yt8m.org/download.py | partition=2/frame/train mirror=us python 
curl data.yt8m.org/download.py | partition=2/frame/validate mirror=us python 
curl data.yt8m.org/download.py | partition=2/frame/test mirror=us python
```

The total size of the video-level features is 31 Gigabytes. They are broken into 3844 shards which can be subsampled to reduce the dataset size. 

To download the Video-level dataset using the download script, navigate your terminal to a directory where you would like to download the data. For example:
```console
mkdir -p ~/v2/yt8m/video; cd ~/v2/yt8m/video 
```
Then download the training and validation data. Note: Make sure you have 31GB of free disk space to store the video-level feature files. Download the entire dataset as follows:
```console
curl data.yt8m.org/download.py | partition=2/video/train mirror=us python 
curl data.yt8m.org/download.py | partition=2/video/validate mirror=us python 
curl data.yt8m.org/download.py | partition=2/video/test mirror=us python
```

The above uses the us mirror. If you are located in Europe or Asia, please swap the mirror flag us with eu or asia, respectively.

## Bibliography
[1] Abu-El-Haija, Sami. “Youtube-8M: A Large-Scale  Video Classification 
Benchmark.” Google. 2016. 

[2] Tsang, SH. “Review: Alexnet, CaffeNet - Winner of ILSVRC 2012 (image 
classification)”.https://medium.com/coinmonks/paper-review-of-alexnet-caffenet-winner-in-ilsvrc-2012-image-classification-b93598314160

[3] O’Shea, Keiron. “An Introduction to Convolutional Neural Network.” 
Aberystwyth University. Dec, 2015. 

[4] Krizhevsky, Alex. “ImageNet Classification with Deep Convolutional Neural Network.” 
Neural Information Processing System (NIPS). 2012. 

[5] C. Feichtenhofer, A. Pinz, and A. Zisserman. “Convolutional two-stream network 
fusion for video action recognition.” CVPR, 2016. 

[6] Jia, Chengcheng. “Stacked Denoising Tensor Auto-Encoder for Action Recognition	
with Spatiotemporal Corruptions.” IEEE. 2017. 

[7] Kim, Minhoe. “Building Encoder and Decoder with Deep Neural Networks: On the 
way to Reality.” IEEE. 2018.		

[8] Metz, Luke & Maheswaranathan, Niru. “Learning Unsupervised Learning Rule.”
	Google Brain. 2018.

[9] Denil, Misha. “Predicting Parameters in Deep Learning.”
	University of Oxford. 2013.

[10] Hetherly, Jeffrey. “Using Deep Learning to Reconstruct High-Resolution Audio.” 
https://blog.insightdatascience.com/using-deep-learning-to-reconstruct-high-resolution-audio-29deee8b7ccd

[11] Kelly, Brendan. “Deep Learning-Guided Image Reconstruction from Incomplete 
Data.” Arxiv. 2017. 

[12] Kuleshov, Volodymyr. “Audio Super-Resolution Using Neural Nets.” International 
Conference on Learning Representation (ICLR). 2017. 

[13] Goodfellow, Ian. “Understanding and Improving Interpolation in Autoencoders via 
an Adversarial Regularizer.” 2018. 

[14] Kanska, Katarzyna and Golinski, Pawel. “Using Deep Learning for single Image 
Super Resolution.” 
https://deepsense.ai/using-deep-learning-for-single-image-super-resolution/

[15] Provost, Foster. “Machine Learning from Imbalanced Data Sets 101.” 
New York University. 2016. 

[16] Chawla, Nitesh. “Smote: Synthetic Minority Over-Sampling Technique.” 
	Technique. Journal of Artificial Intelligence Research. 2002. 

[17] Oyelade, OJ. “Application of K-means Clustering algorithm for prediction of 
Students’ Academic Performance.” International Journal of Computer Science	
and Information Security(IJCSIS). 2010. 

[18] I. Laptev, M. Marszalek, C. Schmid, and B. Rozenfeld. “Learning realistic human 
actions From movies.” CVPR, 2008.

[19] H.Wangand, C.Schmid. “Action Recognition with Improved 
Trajectories.” ICCV, 2013.

[20] Yeom, Samuel. “Privacy Risk in Machine Learning: Analyzing the Connection to 
Overfitting.” IEEE 31st Computer Security Foundations Symposium. 2018. 

[21] Feichtenhofer, Christoph. “Convolutional Two-Stream Network Fusion for Video 
Action Recognition.” CVPR, 2016. 

[22] Sherstinsky, Alex. “Fundamentals of Recurrent Neural Network (RNN) and Long 
Short-Term Memory (LSTM) Network.” Arxiv. 2018. 

[23] Abbas, Alhabib. “Vectors of Locally Aggregated Centers for Compact Video 
Representation.” International Conference on Multimedia and Expo (ICME).	
2015.

[24] Liu, Lingqiao. “Compositional Model Based Fisher Vector Coding for Image	
Classification.” IEEE Transaction on Pattern Analysis and Machine Intelligence.	
2017. 

[25]  Richard, Alexander. “A Bag-of-Words Equivalent Recurrent Neural Network for 
Action Recognition. “ Arxiv from the University of Bonn. 2017. 

[26] Olena. “GPU vs CPU Computing: What to choose?”  Medium. 2018. 
https://medium.com/altumea/gpu-vs-cpu-computing-what-to-choose-a9788a2370c4 

[27] Ha, Anthony. “Weights & Biases raises $5M to build development tools for machine 
Learning”. Techcrunch Article. 2018. https://techcrunch.com/2018/05/31/weights-biases-raises-5m-to-build-development-tools-for-machine-learning/

[28] Vincent, James. “NVIDIA has created the first video game demo using AI-generated 
Graphics.” The verge. 2018. 
https://www.theverge.com/2018/12/3/18121198/ai-generated-video-game-graphics-nvidia-driving-demo-neurips

[29] Mueller, Franziska. “GANerated Hands for Real-Time 3D Hand Tracking from 
Monocular RGB.” CVPR. 2018

[30] Liu, Dan-Ching. “A Practical Guide to ReLU.” Medium article. 2017. 
	https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7

[31] Li, Fei-Fei. “Neural Networks Part 1: Setting up the Architecture.” CS 231 
Convolutional Neural Networks for Visual Recognition”. 

[32] Hyndman, Rob. “How to choose the number of hidden layers and nodes in a 
feedforward neural Network.” Stack exchange website. 
https://stats.stackexchange.com/questions/181/how-to-choose-the-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

[33] Vazquez-Reina, Amelio. “Why are non zero-centered activation functions a problem 
in Backpropagation?” Stack exchange website. 
https://stats.stackexchange.com/questions/237169/why-are-non-zero-centered-activation-functions-a-problem-in-backpropagation

[34] Chung, Junyoung. “Empirical Evaluation of Gated Recurrent Neural Networks on 
Sequence Modeling.” NIPS. 2014

[35] He, Kaiming. “Deep Residual Learning for Image Recognition.” 
	ILSVRC. 2016. 

[36] Zagoruyko, Sergey and Chintala, Soumith. “A MultiPath Network for Object 
Detection.” Facebook AI Research (FAIR). 2016. 
https://www.youtube.com/watch?time_continue=2&v=0eLXNFv6aT8

[37] AEndrs. “Low GPU usage by keras / tensorflow?” 
	Stackoverflow discussion. 2017
https://stackoverflow.com/questions/44563418/low-gpu-usage-by-keras-tensorflow

[38] Krishnan, Gokula. “Difference between the Functional API and the Sequential
          API”. Google group discussion. 2016.  
          https://groups.google.com/forum/#!topic/keras-users/C2qX_Umu0hU
	  
[39] Peterstone. “Saving an Object (Data persistence).” Stackoverflow discussion. 2010.
	https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence

[40] Lu, Milo. “How can we define one-to-one, one-to-many, many-to-one, and 
many-to-many lstm neural networks in keras? [duplicate].” Stackoverflow discussion. 2018

[41] Gal, Yarin. “Dropout as a Bayesian Approximation: Representing Model Uncertainty 
in Deep Learning.” NIPS Conference. 2018

[42] Silver, David. “Mastering the game of Go with Deep Neural Nets with Tree Search”.
Nature Internal Journal of Science. 2016

[43] Silver, David. “Mastering Chess and Shogi by Self-Play with a General 
Reinforcement Learning Algorithm”. DeepMind. 2017

[44] Ray, Tiernan. “Fast.ai’s software could radically democratize AI”. zdnet. 2018
https://www.zdnet.com/article/fast-ais-new-software-could-radically-democratize-ai

[45] Johnson, Khari. “Facebook launches Pytorch 1.0 with Integrations for Google 
Cloud, AWS, and Azure Machine Learning”. venturebeat.com.
https://venturebeat.com/2018/10/02/facebook-launches-pytorch-1-0-integrations-for-google-cloud-aws-and-azure-machine-learning/?fbclid=IwAR0ZbFcn9U-plAx5uiKEsbosACSTjvoNruQsJkesgRbbqSHYx67Mu2M7_YE

[46] Philipp Schmidt, kaggle, 
	https://www.kaggle.com/philschmidt/youtube8m-eda
