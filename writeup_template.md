# **Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on 'End to End Learning for Self-Driving Cars' by Nvidia [1].

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes, and depths between 24 and 64 (model.py lines 82-86)

The model includes RELU layers to introduce nonlinearity (code line 82-86), and the data is normalized in the model using a Keras lambda layer (code line 80).

[1] End to End Learning for Self-Driving Cars by Mariusz Bojarski et al., https://arxiv.org/abs/1604.07316


#### 2. Attempts to reduce overfitting in the model

The model does not contain any dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets (used images from center, left, and right cameras, and their flipped images) to ensure that the model was not overfitting (code line 24-44). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data provided by Udacity.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a convolution neural network model using only images from center camera only.

My first step was to use a convolution neural network model similar to the 'End to End Learning for Self-Driving Cars' by Nvidia [1].

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added images from left and right cameras to the training and validation sets.

Then I augmented the training and validation sets with their flipped images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I modified number of epochs between 5-10, and selected final epochs of 7 after monitoring training and validation loss (i.e. stop when validation loss starts to increase).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

[1] End to End Learning for Self-Driving Cars by Mariusz Bojarski et al., https://arxiv.org/abs/1604.07316


#### 2. Final Model Architecture

The final model architecture (model.py lines 69-91) consisted of a convolution neural network with the following layers and layer sizes.

**Model Summary**
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________


#### 3. Creation of the Training Set & Training Process

I used only data provided by Udacity.

I had 24,108 images (with 8,036 images from center, left, and right cameras each). I then augmented this data by flipping each image. I then preprocessed this data by cropping the images from top (70) and bottom (25).

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. I used an adam optimizer so that manually training the learning rate wasn't necessary.
