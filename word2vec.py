import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from load_create_data import *

class Word2Vec():
	def __init__(self, indices, vocab_size, output_dim=1000, embedding_size, activation='softmax', loss, name="word2vec_net"):
		"""
		Args:
			indices (arr): indices for tf one hot encoder
			vocab_size (int): Dimensionality of input (not including batch)
			output_dim (int): Dimensionality of output (not including batch)
			embedding_size (int): Number of units for the hidden layer. 
			activation (func): activation function
			loss function (tf fnctn): Loss function (after linear output)
			name (str): Name of neural net.
		"""

		self.vocab_size = vocab_size
		self.output_dim = output_dim
		self.indices = indices
		
        # only using one hidden layer
        self.embedding_size = embedding_size
       
	    self.input = tf.placeholder(dtype=tf.float64, 
										shape=(None, vocab_size),
										name="input")
        self.labels = tf.placeholder(dtype=tf.float64, 
										shape=(None, output_dim),
										name="labels")
        
		# begin building network
		self.embeddings = tf.Variable(np.random.normal(size=(self.vocab_size, self.embedding_size)), 
														dtype=tf.float64)
		self.input = tf.nn.embedding_lookup(self.embeddings, self.input)

		self.weights = tf.Variable(np.random.normal(size=(self.vocab_size, self.embedding_size)), dtype=tf.float64)
		self.biases = tf.Variable(tf.zeros([self.vocab_size]))

        self.hidden_layer = activation(tf.matmul(self.input, self.weights)) + self.biases
        
		self.one_hot = tf.one_hot(indices, self.vocab_size)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hidden_layer, 
            labels=self.one_hot))

        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_op = self.optimizer.minimize(self.loss)
'''
		batch_size = 128
		embedding_size = 128  # Dimension of the embedding vector.

		embeddings = tf.Variable(
			tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)
		
		weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
		biases = tf.Variable(tf.zeros([vocabulary_size]))
		hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

		train_one_hot = tf.one_hot(train_context, vocabulary_size)
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, 
			labels=train_one_hot))
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

		assert len(embedding_size) == len(activations)
		with tf.name_scope(name):
			self.vocab_size = vocab_size
			self.output_dim = output_dim
			self.weights = []
			self.biases = []

			self.input = tf.placeholder(dtype=tf.float64, 
										shape=(None, vocab_size),
										name="input")
			self.labels = tf.placeholder(dtype=tf.float64, 
										shape=(None, output_dim),
										name="labels")
			self.neurons = [self.input]
			cur_layer = self.input
			prev_units = vocab_size
			for num_units, activation in zip(embedding_size, activations):
				cur_W = tf.Variable(np.random.normal(size=(prev_units, num_units)), dtype=tf.float64)
				cur_b = tf.Variable(np.random.normal(num_units), dtype=tf.float64)
				cur_layer = activation(tf.matmul(cur_layer, cur_W)) + cur_b
				self.weights.append(cur_W)
				self.biases.append(cur_b)
				self.neurons.append(cur_layer)
				prev_units = num_units
			cur_W = tf.Variable(np.random.normal(size=(embedding_size[-1], output_dim)), 
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