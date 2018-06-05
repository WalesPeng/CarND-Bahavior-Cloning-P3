import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os


# Cameras we will use
cameras = ['left', 'center', 'right']
cameras_steering_correction = [.25, 0., -.25]

def preprocess(image, top_offset=.375, bottom_offset=.125):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (38, 128, 3))
    return image

def generate_samples(data, root_path, augment=True):
    """
    Keras generator yielding batches of training/validation data.
    Applies data augmentation pipeline if `augment` is True.
    """
    while True:
        # Generate random batch of indices
        indices = np.random.permutation(data.count()[0])
        batch_size = 38
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, 38, 128, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for i in batch_indices:
                # Randomly select camera
                camera = np.random.randint(len(cameras)) if augment else 1
                # Read frame image and work out steering angle
                imgDataPath = os.path.join(data[cameras[camera]].values[i].strip())
                filename = imgDataPath.split('\\')[-1]
                imgpath = os.path.join(root_path, filename)
                # print('imgpath = ', imgpath)
                image = mpimg.imread(imgpath)
                angle = data['steering'].values[i] + cameras_steering_correction[camera]
                # Randomly shift up and down while preprocessing
                v_delta = .05 if augment else 0
                image = preprocess(
                    image,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )
                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
            # Randomly flip half of images in the batch
            flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)
