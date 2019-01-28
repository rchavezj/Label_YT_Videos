# Classifying Genre Labels from Youtube Videos

Deep Learning algorithm (Deep Neural Net + LSTM) to label youtube genre labels based on their video frame. Using deep learning (PyTorch + Keras) to extract spatial (pixels) and sequential strings (audio). Later concatenate onto a fully connected network to output the video label genre (E.g. Games, Art & Entertainment, etc.)

## [Wandb Results](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_1.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_2.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_3.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_4.png)

Above are graphs I gathered from the app wandb (Weights & Biasis) to monitor the performance of each deep learning algorithm coded in pytorch. If you wish to visualize the performance, [click here!](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report) I also done experiments for the same deep learning model in keras that can be [found here](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FKeras%20Report). More info on wandb can be [found here](https://www.youtube.com/watch?v=zOB_fZPTeiI&t=108s)

### Pytorch Report
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

## Best Deep Learning Model: <br /> Neural Net + Stream LSTM Concatenated (Tensorboard)

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
