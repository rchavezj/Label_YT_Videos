# Label_YT_Videos

## Wandb Results
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_1.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_2.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_3.png)
![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/wandb_results/pytorch_results_pt_4.png)

Above are graphs I gathered from the app wandb (Weights & Biasis) to monitor the performance of my deep learning algorithms coded in pytorch. If you wish to visualize each algorithm individually, [click here!](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report)

## Neural Net + Stream LSTM Concatenated (Tensorboard)

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
