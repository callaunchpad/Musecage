import tensorflow as tf
import pandas as pd
import numpy as np
import collections


class RNNModel():

	def __init__ RNNModel(self, num_hidden=512, num_input=10, timesteps=3, vocab_size): 

		# Training Parameters
		self.learning_rate = 0.001
		self.training_steps = 100
		self.batch_size = 10
		self.display_step = 200

		# Network Parameters
		self.num_input = num_input
		self.timesteps = timesteps
		self.num_hidden = num_hidden # hidden layer num of features
		self.num_classes = vocab_size #num of unique words

		self.weights = {
			'out': tf.Variable(tf.random_normal([num_hidden, vocab_size]))
		}
		self.biases = {
			'out': tf.Variable(tf.random_normal([vocab_size]))
		}

		# tf Graph input
		X = tf.placeholder("float", [None, timesteps, num_input])
		Y = tf.placeholder("float", [None, num_classes])

	def RNN(self, x, weights, biases):
		x = tf.reshape(x, [-1, self.num_input])
		x = tf.split(x, self.num_input,1)
		rnn_cell = rnn.BasicLSTMCell(self.num_hidden)
		outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
		return tf.matmul(outputs[-1], weights['out']) + biases['out']
	