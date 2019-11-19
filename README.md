# Classifying Video Labels from Youtube 

![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/resource/feature_engineering.png)

<strong> Goal</strong>: Taught an AI to label a genre (E.g., Makeup, Games, Art & Entertainment, etc.) on a youtube video (Audio, RGB, Frames) using a series of deep learning algorithms and compare each one to understand their strengths and weaknesses, and scalability properties. This algorithm can potentially automate repetitive labor organizing youtube videos with similar content in a recommendation search engine and classify copyright material. I will be using Google’s yt8m dataset [1] initially 0.5 petabytes of data, compressed to 1.5 terabytes for the research dataset. There will be two types of features: video-level (2-dimensions) and frame-level (3-dimensions) data. I coded eight deep learning models: 4 in Keras and the same models in PyTorch to compare not only models but frameworks against each other. More details provided under <strong> [Comparing models & frameworks](#Comparing-models-and-frameworks) </strong> section. Link to the <strong>[code](https://github.com/rchavezj/Label_YT_Videos/blob/master/code/Algorithms.ipynb)</strong>

<u>The pipeline would be</u>: </br>
> (1) Youtube videos uploaded from users </br></br>
> (2) Google compressing the content</br></br>
> (3) Reconstructing the data and features for a machine learning pipeline </br></br>
> (4) Output a vector from a model with one element containing the highest probability for the predicted genre label. </br></br>
> (5) Perform a comparison of both models and frameworks </br> </br>


### Contribution: 
Based on interactions and learning from other Kaggle competitors, I applied a best-practices a set of feature engineering and visualizations to the dataset. I contributed the following: feature engineering, comparing models, comparing frameworks (Keras & PyTorch), integration, optimization, implementing distributed training in PyTorch, and evaluating performance (lost, accuracy, CPU/GPU/Hardware usage). </br> 


### Comparing models and frameworks: 
I coded eight deep learning models: 4 in Keras and the same models in PyTorch. I chose to compare not only models but frameworks as well based on research I did on algorithms crashing in production [49]. Tensorflow (Keras) has the largest user base and most traction currently. I want to compare it with PyTorch which is an imperative framework that performs computation as you type it.  Tensorflow (Keras) uses symbolic programming: only computing your code at the end of each graph session [38]. Tensorflow is evolving to become more “PyTorch” like with eager execution [51], however that’s still in alpha and didn’t exist at the time my project started. The value of comparing frameworks was also recognized by google engineers as they were working along a similar path in parallel to my work [51]. More information about the difference between each framework and their performance is written in my paper. My experiments reveal surprising results.  

### Integration: 
We are dealing with compressed data containing two important features we need to aggregate: video-level and frame-level content. We need to integrate three different algorithms into the pipeline:

> (1) Reconstruct the compressed content [1] with initialized parameters. The number of features is smaller than the number of class labels (output genre labels) which is why I had to reconstruct the data. Add more dimensions into the input data. More details are available in the paper under the <strong>Feature Engineering</strong> section. The output of the reconstructed features in step (1) are separately sent into their respected models in (2) and (3).   </br></br>
> (2) Compute sequential data (frame-level) for each video gathered from yt8m. Each video is at least 100 second long to be utilized within the dataset. We can use temporal models (Recurrent Neural Net).  </br></br>
> (3) Instead of 100 second per video, Google created video-level features extracting a task-independent fixed-length vector per frame. In other words video-level features have one less dimension from compressing frame-level sequential content. We can train this data using classifiers like logistic regression or any non-linear design.    </br></br>

Both models <strong>(2)</strong> and <strong>(3)</strong> models contain a softmax approximator model to determine the label of a genre from a youtube video. You can develop an <strong>algorithm (section below) </strong> with both models separately or combined (concatenate). 

### Methods:
Google compressed their yt8m data using Principle Component Analysis (PCA) so the dimension of our input features is 100 for RGB content and 1028 for audio, both of which are less than 3864 which is the vector size of the class labels for the video genre (E.g., Games, Art & Entertainment, etc.). We need to come up with an unsupervised learning technique to reconstruct the data with additional weight parameters to make sure the compressed input content could fit the output dimension. After the number of input features for both RGB and audio has increased, we can now send both features into their respected deep learning algorithm. After reconstructing the data we are still given a challenging problem to select an algorithm that can scale to large amounts of data. Fortunately, deep learning algorithms have been getting remarkable results in the AI community for scalability [2]. For video-level we can use logistic regression or a deep neural net to compute compressed pixels. As for frame-level, sequential features (100 second videos) can also be scaled through temporal algorithms with: RNN’s, LSTM, Self Attention, GRU, and Markov chains. Each has their own limitations computing long sequences of video frames.  </br> </br>

### Feature engineering:  
I reconstructed the data using an autoencoder with additional weight parameters to make sure the compressed input content could fit a large genre label. The output vector mapping the relationship between each genre (E.g., Games, Art & Entertainment, etc.) to their respected youtube video. Once the data has been reconstructed with additional parameters to match the same dimension as the output, we can then choose a series of deep learning algorithms, which is covered in the next section.   </br> </br>

### Algorithms: 
Below are four algorithms I coded to aggregate my data. Later I compare each framework (PyTorch & Keras) with the same models to see which one was most efficient for research and or production. </br>

> (1) Fully Connected Net: Aggregating compressed spatial features from youtube videos </br></br>
> (2) Bidirectional LSTM: We are using a Bidirectional LSTM to aggregate sequential content of video frames. Each youtube video has been cut down to 100 seconds of frames. Any video less than 100 seconds is not part of the dataset to balance the distribution. For example, if there was a 10 second youtube video of nascar-racing inside the dataset while another video with 100 seconds of a video game of cars, the algorithm would likely have a bias prediction labeling a nascar-race as a video game. </br></br>
> (3) Stream-LSTM is similar to Bidirectional except we have the approximators for audio and rgb separately sent into their own fully connected nets. The approximators are later combined into a concatenate function to find the average. For example if the approximator for one genre label in a rgb vector was 79% while the audio was 99%, concatenating both would be 89%.   </br></br>
> (4) Fully Connected Net (Video-level) concatenated with a Stream-LSTM (frame-level). It’s a combination of Algorithm (1) & (3) softmax approximation using concatenation.

## Weights & Biasis (Wandb) Results

Below are graphs I gathered from the app wandb (Weights & Biasis) to monitor the performance of each deep learning algorithm coded in PyTorch and in Keras. If you wish to visualize the performance, [click here!](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report) I also done experiments for the same models in [keras](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FKeras%20Report).  

### [PyTorch Report](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_1.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_2.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_3.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_4.png)

|                                  |   Optimizer   | Loss     |  Accuracy  | Learn Rate | Epoch | Batch Size |   GPU Usage  |   CPU Usage  |   System Memory  |
| :---:                            | :----:   |  :----:  |    :---:   |  :----:    | :---: |    :---:   |    :----:    |    :----:    |    :----:        |
| Neural Net                       |  SGD  |  1.75%    |    45%    |    0.01    | 300   |     30     |    29%       |    15.31%    |    13.89%        |
| Multi-Bidirectional LSTM         |  SGD  |  7.89%   |    64%     |    0.003   | 300   |     64     |    55%       |    12.65%    |    12.26%        |
| Stream LSTM                      |  SGD  |  7.98%   |    64%     |    0.03    | 300   |     64     |     0%       |     97%      |     70.60%       |
| Neual Net + Stream LSTM Concat   |  SGD  |  8.91%    |    95%    |  0.00045   | 300   |     64     |   24.13%     |   12.67%     |     16.49%       |



### [Tensorflow/Keras Report](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FKeras%20Report)

<img src="https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/keras_parallel_coordinates.png">

<img src="https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/keras_loss.png" width="430"/><img src="https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/keras_accuracy.png" width="430"/>


|                                  | Opt  | Loss     |  Accuracy  |  Epoch  | Batch Size |   GPU Usage  |   CPU Usage  |  System Memory  |
| :---:                            |  :----:   |  :----:   |    :---:   | :----:  |    :---:   |    :----:    |    :----:    |    :----: | 
| Neural Net                       |  Adam     |    68.5%  |    66%     |   300   |     84     |     0.13%    |    64.97%    |    14.37%       |
| Multi-Bidirectional LSTM         |  Adam     |    56%  |    77%     |   300   |     64     |     0%       |    99.17%    |    31.19%       |
| Stream LSTM                      |  SGD      |    58.6%  |    84%     |   500   |     64     |     0%       |    93.37%    |    26.07%       |
| Neual Net + Stream LSTM Concat   |  Adam     |    68.6%  |    73%     |   100   |     20     |     0%       |    96.84%    |    28.45%       |

## Fully Connected Net concatenated with a Stream-LSTM
Documentations for my designs are available on [my paper](https://github.com/rchavezj/Label_YT_Videos/blob/master/Paper.pdf).
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/resource/nn_lstm_stream.png)
 

## Best Deep Learning Model: <br /> Fully Connected Net concatenated with a Stream-LSTM (Tensorboard)
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

[5] C.Feichtenhofer, A. Pinz, and A. Zisserman. “Convolutional two-stream network 
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
	
[47] NLP’s ImageNet moment has arrived, 2018
	http://ruder.io/nlp-imagenet/
	
[48] T.Q. Chen, Ricky and Rubanova, Yulia. “Neural Ordinary Differential Equations”, 
Neuralps Conference. 2019.

[49] Talby, David. “Why Machine Learning Models Crash and Burn In Production”. 
Forbes.com.https://www.forbes.com/sites/forbestechcouncil/2019/04/03/why-mac
hine-learning-models-crash-and-burn-in-production/#48dd10272f43

[50] Makadia, Mitul. “Top 8 Deep Learning Frameworks”. Dzone.com
	https://dzone.com/articles/8-best-deep-learning-frameworks

[51] Aggarwal, Keshav. “A brief guide to Tensorflow Eager Execution”. Medium.com
https://towardsdatascience.com/eager-execution-tensorflow-8042128ca7be?gi=6
4b7427864b6

[52] ArthurDent (anonymous username). “Keras Multiple ‘Softmax’ in last layer 
possible”. Datascience.stackexchange.com article. 
https://datascience.stackexchange.com/users/40180/arthurdent

[53] Sunnak, Abhishek. “Evolution of Natural Language Generation”. Medium Article
https://medium.com/sfu-big-data/evolution-of-natural-language-generation-c5d72
95d6517

[54] Madsen, Andreas. “Visualizing memorization in RNNs.” Distill journal
	2019. https://distill.pub/2019/memorization-in-rnns/


[55] Rosenblatt, Frank. “The Perceptron: A Probabilistic Model for information storage 
and organization in the brain”. Cornell Aeronautical Laboratory. 1958. 
	https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf


[56] Geirhos, Robert. “Imagenet-Trained CNNs are biased towards texture; increasing 
	shape bias improve accuracy and robustness”. University of Tubingen
	ICLR 2019. 
	
[57] Jeremy Howard: Deep Learning Frameworks - TensorFlow, PyTorch, fast.ai | AI Podcast Clips
	https://www.youtube.com/watch?v=XHyASP49ses
