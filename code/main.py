# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../v2"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
import tensorflow as tf
# now, let's read the frame-level data
# due to execution time, we're only going to read the first video
video_files = os.listdir("../v2/video/")
frame_files = os.listdir("../v2/frame/")


# Global variables
num_labels = 4716

def createTargetVec(labels):
    out = np.zeros((1, num_labels))
    for label in labels:
        out[0,label] = 1
    return out

def get_dataset(frames, samples=5, k=0):
    rgb_input = np.empty((samples, 100, 1024))
    audio_input = np.empty((samples, 100, 128))
    label_output = np.empty((samples, num_labels))
    sess = tf.InteractiveSession()
    for example in tf.python_io.tf_record_iterator(frames):        
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        labels = tf_seq_example.context.feature['labels'].int64_list.value
        rgb_frame = np.zeros((100, 1024))
        audio_frame = np.zeros((100, 128))
        for i in range(100):
            rgb_frame[i] = tf.cast(tf.decode_raw(
                   tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                          ,tf.float32).eval()
            audio_frame[i] = tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                           ,tf.float32).eval()
        rgb_input[k] = rgb_frame
        audio_input[k] = audio_frame
        label_output[k] = createTargetVec(labels[:])
        k += 1
        progress = (k / samples) * 100
        if int(progress) % 10 == 0:
            print("Progress", progress, "%")
        if k >= samples:
            break
    sess.close()
    return audio_input, rgb_input, label_output


def main():
    # video_lvl_record = "../v2/video/train-1.tfrecord"
    frame_lvl_record = "../v2/frame/train4f.tfrecord"

    audio_input, rgb_input, label_output = get_dataset(frame_lvl_record)

    X_train_rgb, X_test_rgb, _, _ = train_test_split(rgb_input, label_output, test_size=0.2, random_state=42)
    X_train_audio, X_test_audio, y_train, y_test = train_test_split(audio_input, label_output, test_size=0.2, random_state=42)

    print(len(X_train_rgb), 'train rgb')
    print(len(X_test_rgb), 'test rgb')

    print(len(X_train_audio), 'train sequences')
    print(len(X_test_audio), 'test sequences')

    print('X_train_rgb shape:', X_train_rgb.shape)
    print('X_test_rgb shape:', X_test_rgb.shape)

    print('X_train_audio shape:', X_train_audio.shape)
    print('X_test_audio shape:', X_test_audio.shape)

    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    from keras.layers import LSTM
    from keras.layers import Dense
    from keras.models import Sequential

    model = Sequential()
    model.add(LSTM(128, input_shape=X_train_audio.shape[1:], return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train_audio, y_train, validation_data=(X_test_audio, y_test), epochs=60, batch_size=64)


if __name__ == '__main__':
    main()

