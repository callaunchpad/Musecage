import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from load_create_data import *

class Word2Vec():
	def __init__(self, vocab_size, embedding_size, name="word2vec"):
		"""
		Args:
			vocab_size (int): Range of input (not including batch)
			embedding_size (int): Equal to vocabulary size 
			name (str): Name of neural net.
		"""
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size

		self.input = tf.placeholder(dtype=tf.int32, 
										shape=(None, 1),
										name="input")
		self.labels = tf.placeholder(dtype=tf.int32, 
										shape=(None, 1),
										name="labels")
		
		self.one_hot = tf.one_hot(self.input, self.vocab_size, dtype=tf.float64)
		self.labels = tf.stop_gradient(self.labels)

		self.embed_layer = tf.layers.dense(self.one_hot, self.embedding_size, use_bias=False)
		self.output = tf.layers.dense(self.embed_layer, self.vocab_size, use_bias=False)
		
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels))

		self.optimizer = tf.train.AdamOptimizer(1e-4)
		self.train_op = self.optimizer.minimize(self.loss)

	def train_step(self, features, labels, sess):
		loss, _ = sess.run([self.loss, self.train_op], 
												feed_dict={self.input: features, self.labels: labels.astype(np.float64)})
		return loss
	
	def evaluate(self, features, labels, sess):
		loss = sess.run(self.loss, feed_dict={self.input: features, self.labels: labels})
		return loss

	def predict(self, features, sess):
		pred = sess.run(self.output, feed_dict={self.input: features})
		return pred

	def get_embed(self):
		weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.embed_layer.name)[0] + '/kernel:0')
		return weights