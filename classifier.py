import dicom
import numpy as np
import tensorflow as tf
from random import shuffle, sample

sess = tf.InteractiveSession()

data = []
trumax = 4095

counter = 1
while True:
    try:
        ds = dicom.read_file("C:/Users/jerem/Desktop/CT SCAN DATA/case_100/100_20170225_093021/DICOM/" + str(counter) + ".DCM")
    except:
        break
    if ds.SeriesDescription == "PE":
        pix = ds.pixel_array
        if counter >= 85 and counter <= 110:
            for i in range(0, 10):
                data.append((pix/trumax, np.array([0, 1])))
        else:
            data.append((pix/trumax, np.array([1, 0])))
    counter += 1

counter = 1
while True:
    try:
        ds = dicom.read_file("C:/Users/jerem/Desktop/CT SCAN DATA/case_101/PATIENT12381_20170302_073751/DICOM/" + str(counter) + ".DCM")
    except:
        break
    if ds.SeriesDescription == "PE":
        pix = ds.pixel_array
        data.append((pix/trumax, np.array([1, 0])))
    counter += 1

print(len(data))
shuffle(data)
training_data = data[0:int(len(data)*2/3)]
testing_data = data[int(len(data)*2/3):]


# Build model.

x = tf.placeholder(tf.float32, shape=[None, 512 * 512])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 512, 512, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([64 * 64 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 64 * 64 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

print(y_conv.shape)
print(y_.shape)


# Run model?

def frames_to_batch(frames):
    batch_x = np.zeros([len(frames), 512 * 512])
    batch_y = np.zeros([len(frames), 2])

    for i in range(0, len(frames)):
        batch_x[i] = frames[i][0].reshape((512 * 512))
        batch_y[i] = np.array(frames[i][1])

    return batch_x, batch_y

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20):
  frames = sample(training_data, 50)
  batch = frames_to_batch(frames)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  print("About to start " + str(i) + "th train step!")
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

eval_batch = frames_to_batch(testing_data)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: eval_batch[0], y_: eval_batch[1], keep_prob: 1.0}))

