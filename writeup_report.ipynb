{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Traffic sign classifier project"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here is a writeup report where i will go trough all steps of the project to descride why and how I addressed each one.\n",
    "\n",
    "The goals / steps of this project are the following:\n",
    "* Load the data set \n",
    "* Explore, summarize and visualize the data set\n",
    "* Design, train and test a model architecture\n",
    "* Use the model to make predictions on new images\n",
    "* Analyze the softmax probabilities of the new images\n",
    "  \n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data set summary and exploration"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The data is downloaded from German Traffic Sign Dataset that contains traffic sign images and their labels in three files (train.p,valid.p and test.p).\n",
    "\n",
    "I used the pandas library to calculate summary statistics of the traffic\n",
    "signs data set:\n",
    "\n",
    "* The size of training set is 34799\n",
    "* The size of the validation set is 4410\n",
    "* The size of test set is 12630\n",
    "* The shape of a traffic sign image is (32,32,3)\n",
    "* The number of unique classes/labels in the data set is 43\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Exploratory visualization of the data set"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First, I used a function that I called signS_label to read the \"signnames.csv\" file, and I plot out some examples of signs traffic images with their labels as shown below:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"example_for_report/traffic_sign.png\",width=300,height=300>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Second, I plot a graph showing the distridution of the classes,it s is easy to note that some classes have a lower frequency,which means that the way the data are distributed can affect our model, so i tried to add more images to classes that have lower frequency to reach the mean of the images in each class."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"example_for_report/graph1.png\",width=300,height=300>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Peprocessing data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The way I did that is by generating additional data by rotating images using ndimage function from scipy libray and add those images to low frequency classes to reach the mean of number of images in each bin.and that s what I get."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"example_for_report/graph2.png\",width=300,height=300>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we have more data with most have same frequency,the number of training set now is 46714."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Processing data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this step, I chossed two techniques to process data:\n",
    "    - grayscale images to reduce the size of color channel dimension from 3 to 1.\n",
    "    - normalize images to change the range of the pixel intensity to be between 0 and 1\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Design and Test a Model Architecture"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model architecture"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For this step I used first LeNet architecture used in leNet lab as a point of start to see how the model is converging, but the accuracy did not reach 93%, so I tried to modify the architecture by adding three convolutional layers to make it more depth. \n",
    "Here is the pipelines of my chosen architecture:\n",
    "- Layer 1: Convolutional. Input = 32x32x1. Output = 30x30x32.followed by an activation layer\n",
    "- Layer 2: Convolutional. Input  = 30x30x32.Output = 28x28x32.followed by an activation layer\n",
    "- MaxPooling layer. Input =28x28x32. Output = 14x14x32.\n",
    "I added three more convolutional layers to make the network more deeper.\n",
    "- Layer 3: Convolutional. Iutput = 14x14x32. Output = 12x12x64\n",
    "- Layer 4: Convolutional. Iutput = 12x12x64. Output = 10x10x64\n",
    "- Layer 5: Convolutional. Iutput = 5x5x64. Output = 3x3x128\n",
    "then i followed the normal leNet architecture by adding:\n",
    "- Flatten layer. Input = 3x3x128. Output = 1152.\n",
    "- Layer 6: Fully Connected. Input = 1152. Output = 1024.followed by activation (relu).\n",
    "- Layer 7: Fully Connected. Input = 1024. Output = 1024.followed by activation (relu).\n",
    "I added a dropout layer \n",
    "- Dropout layer with keep_prob = 0.6\n",
    "then the last layer with a linear activation\n",
    "- Layer 8: Fully Connected. Input = 1024. Output = 43."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train the model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To train the model:\n",
    "- I first shuffle both X_train,y_train and X_valid,y_valid.\n",
    "here are the parameters tuned and the valid_accuracy that i got :\n",
    "- EPOCHS = 10, BATCH_SIZE = 256, learning rate = 0.005, valid_accuracy =0.93\n",
    "- EPOCHS = 10, BATCH_SIZE = 128, learning rate = 0.005, valid_accuracy =0.95\n",
    "- EPOCHS = 03, BATCH_SIZE = 150, learning rate = 0.001, valid_accuracy =0.96\n",
    "After I trained the model several times and make changes to the architecture of the network\n",
    "my final results were :\n",
    "* validation set accuracy of 0.963 \n",
    "* test set accuracy of 0.868\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "While training I started to use LeNet, but i seems that eventhough i changed the hyperparameters\n",
    "the model finish with a validation accuracy of 0.9.so i tried to make the network deeper and i added three more\n",
    "convolutional layers,finally the validation acuracy increased to reach at the end 0.96 with a test accuracy of 0.95\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test a Model on New Images\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here are 5 German traffic signs that I found on the web:\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " <img src=\"my_pics/sign_1.jpg\",width=100,height=100,title=\"Arterial\">\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"my_pics/Arterial.jpg\",width=100,height=100,title=\"Attention\">\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"my_pics/Do-Not-Enter.jpg\",width=100,height=100,title=\"speed limit(120km/h\">\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"my_pics/sign_2.jpg\",width=100,height=100,title=\"speed limit(130km/h)\">\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"my_pics/STOP_sign.jpg\",width=100,height=100,title=\"stop\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following step was to process those images by :\n",
    "- converting images to RGB.\n",
    "- resize images to a dim of 32*32*3.\n",
    "- grayscale images.\n",
    "- normalize images."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analyze Performance\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Your explanation can look something like: the accuracy on the captured images is 100% while it was 86% on the testing set thus It seems the model is not overfitting."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  },
  "widgets": {
   "state": {},
   "version": "1.1.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
