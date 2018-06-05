# End to End Learning for Self-Driving Cars

The goal of this project was to train a end-to-end deep learning model that would let a car drive itself around the track in a driving simulator.

## Project structure

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `data.py`                    | Methods related to data augmentation, preprocessing and batching.                  |
| `model.py`                   | Implements model architecture and runs the training pipeline.                      |
| `model.json`                 | JSON file containing model architecture in a format Keras understands.             |
| `model.h5`                   | Model weights.                                                                     |
| `weights_logger_callback.py` | Implements a Keras callback that keeps track of model weights throughout training. |
| `drive.py`                   | Implements driving simulator callbacks, essentially communicates with the driving simulator app providing model predictions based on real-time data simulator app is sending. |
| `output_video`               | Performance the record of vehicle autonomously driving around the track one over one lap

## Data collection and balancing

The provided driving simulator had two different tracks. I used first of them for collecting training data.

The driving simulator would save frames from three front-facing "cameras", recording data from the car's point of view; as well as various driving statistics like throttle, speed and steering angle. I am going to use camera data as model input and expect it to predict the steering angle in the `[-1, 1]` range.

I have collected a dataset containing approximately **8 minutes worth of driving data** around track 1. This would contain both driving in _"smooth"_ mode (staying right in the middle of the road for the whole lap), and _"recovery"_ mode (letting the car drive off center and then interfering to steer it back in the middle). 

Just as one would expect, resulting dataset was extremely unbalanced and had a lot of examples with steering angles close to `0`. So I applied a designated random sampling which ensured that the data is as balanced across steering angles as possible. This process included splitting steering angles. 



Please, mind that I balanced dataset across _absolute_ values, as by applying horizontal flip during augmentation I end up using both positive and negative steering angles for each frame.

## Data augmentation and preprocessing

After balancing 8 minutes worth of driving data I ended up with **4425 samples**, which most likely wouldn't be enough for the model to generalise well. However, as many pointed out, there a couple of augmentation tricks that should extend the dataset significantly:

- **Left and right cameras**. Along with each sample I receive frames from 3 camera positions: left, center and right. Although I am only going to use central camera while driving, I can still use left and right cameras data during training after applying steering angle correction, increasing number of examples by a factor of 3.
```python
cameras = ['left', 'center', 'right']
steering_correction = [.25, 0., -.25]
camera = np.random.randint(len(cameras))
image = mpimg.imread(data[cameras[camera]].values[i])
angle = data.steering.values[i] + steering_correction[camera]
```

In order to get filename from 'driving_log.csv', I add a new row of ['left', 'center', 'right', 'steering'] in the top of csv, each element of the row corresponded to the content of column.   

- **Horizontal flip**. For every batch I flip half of the frames horizontally and change the sign of the steering angle, thus yet increasing number of examples by a factor of 2.
```python
flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
x[flip_indices] = x[flip_indices, :, ::-1, :]
y[flip_indices] = -y[flip_indices]
```
- **Vertical shift**. I cut out insignificant top and bottom portions of the image during preprocessing, and choosing the amount of frame to crop at random should increase the ability of the model to generalise.
```python
top = int(random.uniform(.325, .425) * image.shape[0])
bottom = int(random.uniform(.075, .175) * image.shape[0])
image = image[top:-bottom, :]
```

I then preprocess each frame by cropping top and bottom of the image and resizing to a shape our model expects (`32×128×3`, RGB pixel intensities of a 32×128 image). The resizing operation also takes care of scaling pixel values to `[0, 1]`.

```python
image = skimage.transform.resize(image, (32, 128, 3))
```

The 3 cameras data as below:
<p align="center">
  <img src="images/frames_original.png" alt="Original"/>
</p>


Augmentation pipeline is applied using a Keras generator, which lets us do it in real-time on CPU while GPU is busy backpropagating!

## Model 

I started with the model described in [Nvidia paper](https://arxiv.org/abs/1604.07316) and kept simplifying and optimising it while making sure it performs well on both tracks. It was clear I wouldn't need that complicated model, as the data I am working with is way simpler and much more constrained than the one Nvidia team had to deal with when running their model. Eventually I settled on a fairly simple architecture with **3 convolutional layers and 3 fully connected layers**.

<p align="center">
  <img src="images/model.png" alt="Architecture"/>
</p>

This model can be very briefly encoded with Keras.

```python
from keras import models
from keras.layers import core, convolutional, pooling
from keras.layers import Flatten, Dense, Lambda, Dropout

model = models.Sequential()
model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
``` 

I tried to add dropout layer after dense layers to prevent overfitting, and the model proved to generalise quite well. The model was trained using **Adam optimiser with a learning rate = `1e-04` and mean squared error as a loss function**. I used 20% of the training data for validation (which means that I only used **3540 out of 4425 examples** for training), and the model seems to perform quite well after training for **~10 epochs**.

## Results

The car manages to drive fine on tracks one. It rarely goes off the middle of the road, the video named 'output_video.mp4' shown the record of vehicle autonomously driving around the track one over one lap.


