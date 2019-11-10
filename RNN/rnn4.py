import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import collections


class RNNModel():

	def __init__ (self, question, n_hidden=512, n_input=3): 

		# Network Parameters
		self.n_input = n_input
		self.n_hidden = n_hidden # hidden layer num of features

		question_list = question.split()

		def build_dataset(words):
			count = collections.Counter(words).most_common()
			dictionary = dict()
			for word, _ in count:
				dictionary[word] = len(dictionary)
			reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
			return dictionary, reverse_dictionary

		dictionary, reverse_dictionary = build_dataset(question_list)
		vocab_size = len(dictionary)

		self.weights = {
			'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
		}
		self.biases = {
			'out': tf.Variable(tf.random_normal([vocab_size]))
		}

		# tf Graph input
		self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input, 1], name="X")
		self.Y = tf.placeholder(dtype=tf.float32, shape=[None, vocab_size], name="Y")

	def RNN(self, x):
	    # reshape to [1, n_input]
	    x = tf.reshape(x, [-1, self.n_input])

	    # Generate a n_input-element sequence of inputs
	    # (eg. [had] [a] [general] -> [20] [6] [33])
	    x = tf.split(x,self.n_input,1)

	    # 2-layer LSTM, each layer has n_hidden units.
	    # Average Accuracy= 95.20% at 50k iter
	    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])

	    # 1-layer LSTM with n_hidden units but with lower accuracy.
	    # Average Accuracy= 90.60% 50k iter
	    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
	    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

	    # generate prediction
	    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

	    # there are n_input outputs but
	    # we only want the last output
	    return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
	
	