import tensorflow as tf
import os
import numpy as np
import time
from random import randint

# TRAIN_DIR = "./data/MontgomerySet/CXR_png/"
# TEST_DIR = "./data/ChinaSet_AllFiles/CXR_png/"

TRAIN_DIR = "./data/ChinaSet_AllFiles/CXR_png/"
TEST_DIR = "./data/MontgomerySet/CXR_png/"

# Dir with only a few images, to test general workings before training on full set
# TRAIN_DIR = "./data/ChinaSet_AllFiles/CXR_png/tmptest/"
# TEST_DIR = "./data/MontgomerySet/CXR_png/tmptest/"

#TODO: restructure input formats

imagesets = {}
resultsets = {}

# Read images in specified dir
def next_batch(directory):
    input_tensors = []
    sess = tf.Session()

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            print("decoding file:" + filename)
            image = tf.image.decode_png(tf.read_file(directory + filename), channels=1)

            resized = tf.image.resize_image_with_crop_or_pad(image, 128, 128)

            with sess.as_default():
                array_vals = resized.eval()

            input_tensors.append(array_vals)
    #return_array = np.array(input_tensors, dtype=np.int32)
    return input_tensors


# get the results for the files
def getresults(directory):
    expected_results = []
    for filename in os.listdir(directory):
        if filename.endswith("_0.png"):
            expected_results.append(0)
        elif filename.endswith("_1.png"):
            expected_results.append(1)

    #return_array = np.array(expected_results, dtype=np.int32)
    return expected_results


# These are the input dimensions
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=128*128)]

# Define the neural network
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[1024, 1024],
                                          n_classes=2,
                                          model_dir="./tmp/tbc_naive")


# Make an input function to provide data
def input_fn(directory, size):
    if directory in imagesets:
        input_tensors = imagesets[directory]
    else:
        input_tensors = next_batch(directory)
        imagesets[directory] = input_tensors

    if directory in resultsets:
        results = resultsets[directory]
    else:
        results = getresults(directory)
        resultsets[directory] = results


    if len(input_tensors) != len(results):
        print("input and result length do not match")

    resultsize = len(input_tensors)
    print("retrieved " + str(resultsize) + " images")
    tensor_set = []
    result_set = []

    for i in range(size):
        rand = randint(0, resultsize - 1)
        tensor_set.append(input_tensors[rand])
        result_set.append(results[rand])

    #Wait, this should already have happened...
    numtpytensors = np.array(tensor_set, dtype=np.int32)
    numtpyresults = np.array(result_set, dtype=np.int32)
    return tf.convert_to_tensor(numtpytensors), tf.convert_to_tensor(numtpyresults)


# Fit model.
#classifier.fit(input_fn=lambda: input_fn(TRAIN_DIR, 600), steps=10000)

#TODO: remove model dir and train with logging. Is my current loss any good or no??

# this eval function performs the random again. Does not give a good trianing fit
ev_train = classifier.evaluate(input_fn=lambda: input_fn(TRAIN_DIR, 600), steps=1)
loss_score_train = ev_train["loss"]
print("Training loss: {0:f}".format(loss_score_train))
accuracy_score_train = ev_train["accuracy"]
print("Training accuracy: {0:f}".format(accuracy_score_train))

ev_test = classifier.evaluate(input_fn=lambda: input_fn(TEST_DIR, 100), steps=1)
loss_score = ev_test["loss"]
print("Testing loss: {0:f}".format(loss_score))
accuracy_score = ev_test["accuracy"]
print("Testing accuracy: {0:f}".format(accuracy_score))
