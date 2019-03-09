import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D


# Generate data batch-by-batch to train and validate the model
def generator(samples, batch_size=32, training=True):
    num_samples = len(samples)
    # The generator is expected to loop over its data indefinitely.
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Get path to images from center, left, and right cameras
                for i in range(3):
                    filename = batch_sample[i].split('/')[-1]
                    path = './data/IMG/' + filename
                    # Images in drive.py are loaded in RGB color space. Train and evaluate images in RGB color space.
                    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                    images.append(image)

                # Create adjusted steering angles for images from center, left, and right cameras
                correction = 0.2
                angle = float(batch_sample[3])
                # Steering angles for images from center, left, and right cameras
                angles.extend([angle, angle + correction, angle - correction])

            if training:
                # Augment images with their flipped images
                aug_images = []
                aug_angles = []
                for image, angle in zip(images, angles):
                    aug_images.append(image)
                    aug_angles.append(angle)
                    aug_images.append(cv2.flip(image, 1))
                    aug_angles.append(-1.0 * angle)

                X_train = np.array(aug_images)
                y_train = np.array(aug_angles)
                yield shuffle(X_train, y_train)
            else:
                X_valid = np.array(images)
                y_valid = np.array(angles)
                yield shuffle(X_valid, y_valid)


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)    # Skip the headers
    for sample in reader:
        samples.append(sample)

# Split data between Training and Validation set before data augmentation
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size = 32

# Compile and train model with generator function
train_generator = generator(train_samples, batch_size=batch_size)

# No image augmentation for validation of the model (simulate real-world usage)
valid_generator = generator(valid_samples, batch_size=batch_size, training=False)

# Model from 'End to End Learning for Self-Driving Cars' by Nvidia
model = Sequential()

# Cropping:
#   70 rows pixels from the top of the image
#   25 rows pixels from the bottom of the image
#   0 columns of pixels from the left of the image
#   0 columns of pixels from the right of the image
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))

# pixel_normalized = pixel / 255
# pixel_mean_centered = pixel_normalized - 0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(24, 5, strides=2, activation='relu'))    # Output = (31, 158, 24)
model.add(Conv2D(36, 5, strides=2, activation='relu'))    # Output = (14, 77, 36)
model.add(Conv2D(48, 5, strides=2, activation='relu'))    # Output = (5, 37, 48)
model.add(Conv2D(64, 3, activation='relu'))    # Output = (3, 35, 64)
model.add(Conv2D(64, 3, activation='relu'))    # Output = (1, 33, 64)
model.add(Flatten())    # Output = 2112
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile('adam', loss='mse')    # optimizer='adam'

history_object = model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(len(train_samples) / batch_size),
    validation_data=valid_generator,
    validation_steps=np.ceil(len(valid_samples) / batch_size),
    epochs=7,
    verbose=1)

model.save('model.h5')
