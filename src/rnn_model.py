import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class RNNModel():

	def __init__ (self, rnn_input, n_hidden=512, embed_size=300, dense=True): 
		self.n_hidden = n_hidden
		self.embed_size = embed_size
		self.rnn_input = rnn_input
		self.dense = dense
		self.output = self.build_graph()

	def build_graph(self):
		rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden), rnn.BasicLSTMCell(self.n_hidden)])
		if self.dense:
			self.rnn_input = tf.layers.dense(self.rnn_input, self.embed_size, activation=tf.tanh)
		outputs, states = tf.nn.dynamic_rnn(rnn_cell, self.rnn_input, dtype=tf.float64)

		out = tf.concat([states[1][0], states[1][1]], axis=1)
		return out
	
	