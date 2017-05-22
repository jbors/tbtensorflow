import tensorflow as tf
import os
import csv
import numpy as np

DIR = "./data/MontgomerySet/CXR_png/"
RESULTS_DIR = "./data/MontgomerySet/ClinicalReadings/"
TEST_DIR = ""


def next_batch():
    input_tensors = []
    for filename in os.listdir(DIR):
        if filename.endswith(".png"):
            print("deconding file:" + filename)
            image = tf.image.decode_png(tf.read_file(DIR + filename))

            resized = tf.image.resize_image_with_crop_or_pad(image, 128, 128)

            array_vals = resized.eval()
            input_tensors.append(array_vals)

    return input_tensors


def results():
    expected_results = []
    csv_file = open(RESULTS_DIR + 'results.csv', "r")
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:

        if row[3] == '0':
            expected_results.append([])
            expected_results[len(expected_results) - 1].append(1)
            expected_results[len(expected_results) - 1].append(0)
        elif row[3] == '1':
            expected_results.append([])
            expected_results[len(expected_results) - 1].append(0)
            expected_results[len(expected_results) - 1].append(1)
    print("Expected: " + str(len(expected_results)))
    return expected_results


# Start by defining some functions to get randomly initialized variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution function
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling function
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Convolution layer 1. Connect 5*5 grids to 32 nodes
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# Define the input placeholder of the correct size
x = tf.placeholder(tf.float32, [None,128,128,1])

# Reshape the input to a 128x128 block
x_image = tf.reshape(x, [-1,128,128,1])

# Apply ReLU nodes to the image inputs
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Pool the outputs in 2x2 blocks. This should make it a 64x64 image.
# TODO: Not sure why this is being applied AFTER the convolution layer...
h_pool1 = max_pool_2x2(h_conv1)


# Create a second conversion layer (64 nodes)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Pool the results of this again
h_pool2 = max_pool_2x2(h_conv2)


# Transform into a 32*32 matrix and connect to a fully connected layer of 1024 nodes
W_fc1 = weight_variable([32 * 32 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Add a dropout on the results of the fully connected layer to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax into 2 variables
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Expected outcomes. This should be an array of one zero and a one
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start your session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train
sess.run(tf.global_variables_initializer())
batch = np.array(next_batch())
results = np.array(results())
for i in range(1500):
    if i%20 == 18:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch, y_:results, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch, y_: results, keep_prob: 1.0})

print("Done training")


