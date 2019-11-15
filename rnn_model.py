import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class RNNModel():

	def __init__ (self, rnn_input, n_hidden=512): 

		# Network Parameters
		# num_words = len(indices)
		# question_embedding = tf.one_hot(indices, depth)
		# tf Graph input
		# self.x = tf.placeholder(dtype=tf.float32, shape=[None, 1, embedding_len], name="inputs")
		self.n_hidden = n_hidden
		self.rnn_input = rnn_input
		self.output = self._build_graph()

	def _build_graph(self):
		rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])
		embed_input = tf.layers.dense(self.rnn_input, 300, activation=tf.tanh)
		outputs, states = tf.nn.dynamic_rnn(rnn_cell, embed_input, dtype=tf.float64)

		out = tf.concat([states[1][0], states[1][1]], axis=1)

		return out
	
	