import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from load_create_data import *

class Word2Vec():
	def __init__(self, questions, indices, input_dim, output_dim=1000, hidden_layer_units, activation='softmax', loss, name="word2vec_net"):
		"""
		Args:
			input_dim (int): Dimensionality of input (not including batch)
			output_dim (int): Dimensionality of input (not including batch)
			hidden_layer_units (array of ints): Array number of units for each
					hidden layer. 
					Length of the array is the number of hidden layers.
					0th index comes first after input tensor
			activations (array): List of tf activation functions.
					Must be the same length as hidden_layer_units
			loss function (tf fnctn): Loss function (after linear output)
			name (str): Name of neural net.
		"""

        self.input_dim = input_dim
		self.output_dim = output_dim
        # only using one hidden layer
        self.hidden_layer_units = hidden_layer_units
       
	    self.input = tf.placeholder(dtype=tf.float64, 
										shape=(None, input_dim),
										name="input")
        self.labels = tf.placeholder(dtype=tf.float64, 
										shape=(None, output_dim),
										name="labels")
        
        self.weights = tf.Variable(np.random.normal(size=(hidden_layer_units, output_dim)), dtype=tf.float64)
        self.biases = tf.Variable(tf.zeros([vocabulary_size]))

        self.hidden_layer = activation(tf.matmul(self.input, self.weights)) + self.biases
        
		self.one_hot = tf.one_hot(indices, self.input_dim)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, 
            labels=self.one_hot))

        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_op = self.optimizer.minimize(self.loss)
'''
		assert len(hidden_layer_units) == len(activations)
		with tf.name_scope(name):
			self.input_dim = input_dim
			self.output_dim = output_dim
			self.weights = []
			self.biases = []

			self.input = tf.placeholder(dtype=tf.float64, 
										shape=(None, input_dim),
										name="input")
			self.labels = tf.placeholder(dtype=tf.float64, 
										shape=(None, output_dim),
										name="labels")
			self.neurons = [self.input]
			cur_layer = self.input
			prev_units = input_dim
			for num_units, activation in zip(hidden_layer_units, activations):
				cur_W = tf.Variable(np.random.normal(size=(prev_units, num_units)), dtype=tf.float64)
				cur_b = tf.Variable(np.random.normal(num_units), dtype=tf.float64)
				cur_layer = activation(tf.matmul(cur_layer, cur_W)) + cur_b
				self.weights.append(cur_W)
				self.biases.append(cur_b)
				self.neurons.append(cur_layer)
				prev_units = num_units
			cur_W = tf.Variable(np.random.normal(size=(hidden_layer_units[-1], output_dim)), 
								dtype=tf.float64)
			self.output = tf.matmul(cur_layer, cur_W)
			self.weights.append(cur_W)
			self.neurons.append(self.output)
			self.loss = tf.reduce_mean(loss(self.labels, self.output))
			self.optimizer = tf.train.AdamOptimizer(1e-4)
			self.train_op = self.optimizer.minimize(self.loss)

		return
'''

	def train_step(self, features, labels, sess):
		loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.train_op], 
											  feed_dict={self.input: features, self.labels: labels.astype(np.float64)})
		return loss, accuracy
	
	def evaluate(self, features, labels, sess):
		loss, accuracy = sess.run([self.loss, self.accuracy], 
										   feed_dict={self.input: features, self.labels: labels})
		return loss, accuracy

	def predict(self, features, sess):
		pred = sess.run(self.output, feed_dict={self.input: features})
		return pred