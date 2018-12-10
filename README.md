# Label_YT_Videos

![alt text](https://github.com/rchavezj/Label_YT_Videos/blob/master/pytorch_results.png "Pytorch Results")

Above image is an analytic tool I ran called wandb (Weights and Biasis) to monitor the performance of my deep learning algorithms coded in pytorch. If you wish to visualize each algorithm individually, [click here!](https://app.wandb.ai/rchavezj/label_yt_videos/reports?view=rchavezj%2FPytorch%20Report)


To download the Frame-level dataset using the download script, navigate your terminal to a directory where you would like to download the data. For example:
mkdir -p ~/data/yt8m/frame; cd ~/data/yt8m/frame
Then download the training and validation data. Note: Make sure you have 1.53TB of free disk space to store the frame-level feature files. Download the entire dataset as follows:

```console
curl data.yt8m.org/download.py | partition=2/frame/train mirror=us python 
curl data.yt8m.org/download.py | partition=2/frame/validate mirror=us python 
curl data.yt8m.org/download.py | partition=2/frame/test mirror=us python
```

The above uses the us mirror. If you are located in Europe or Asia, please swap the mirror flag us with eu or asia, respectively.

To download 1/100-th of the training data from the US use:
```console
curl data.yt8m.org/download.py | shard=1,100 partition=2/frame/train mirror=us python
```
