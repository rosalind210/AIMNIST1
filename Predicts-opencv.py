# Rosalind Ellis (rnellis@brandeis.edu)
# Alex Feldman (felday@brandeis.edu)
# Sofia Lavrentyeva (slavren@brandeis.edu)
# COSI 101A
# Makes predictions with saved neural network from MNIST_CNN.py
# 3/26/17

import sys
import glob
import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
session = tf.InteractiveSession();

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # small amount of noise to prevent 0 gradients
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape) # positive to avoid dead neurons
  return tf.Variable(initial)

#the convolution
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#reduce 2X2 and take largest of the pool
def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784]) #input
y_ = tf.placeholder(tf.float32, [None, 10])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = avg_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_2x2(h_conv2)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #output
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./neuralnetwork-10k-max/model.ckpt")

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
	parts = numbers.split(value)
	parts[1::2] = map(int, parts[1::2])
	return parts

with session.as_default():
	output = open('predictions-max.txt', 'w')
	with sess.as_default():
	  	for image_path in sorted(glob.glob("./examples/*.png"), key=numericalSort):
	  		image = cv2.imread(image_path,0) #black and white
	  		cv2.imshow("f",image)
	  		print(image.shape)
	  		cv2.waitKey(22)
			image = misc.imread(image_path)
			image = misc.imresize(image, (28, 28), interp="bicubic").astype(np.float32, casting='unsafe')
			#checked shape is 28x28, checked type is float32
			newX = np.reshape(image, (1,784))
			prediction = tf.argmax(y_conv, 1)
			p = prediction.eval(session = sess, feed_dict={x: newX, keep_prob: 1.0})
			img_path = image_path.split("/")
			p = str(p)
			string = '{}\t{}\n'.format(img_path[2], p[1:-1])
			output.write(string)
