import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class RNNModel():

	def __init__ (self, question_embedding, n_hidden=512): 
		"""
		params: question_embedding is a matrix with shape (num_words, embedding_len)
		"""
		# Network Parameters
		self.n_hidden = n_hidden # hidden layer num of features
		num_words, embedding_len = question_embedding.shape
		# tf Graph input
		self.x = tf.placeholder(dtype=tf.float32, shape=[None, num_words, embedding_len], name="inputs")
		self.outputs, self.states = self._build_graph()

	def _build_graph(self):
		rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])
		outputs, states = tf.nn.dynamic_rnn(rnn_cell, self.x, dtype=tf.float32)

		print("Output shape:", tf.shape(outputs))
		print("States shape:", tf.shape(states))
		return outputs, states

pred=RNNModel(np.random.rand(10,1024))
	
	