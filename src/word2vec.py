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
        Creates Word2Vec neural net based on:
            - vocab_size (int): number of top words the model will choose a solution from
            - embedding_size (int): dimension of word embedding
            - name (str): name of neural net
        """

        # initialize variables (vocab_size, embedding_size, input, labels)
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

		# pass one hot encoding through embedding layer
		self.embed_layer = tf.layers.dense(self.one_hot, self.embedding_size, use_bias=False)
		self.output = tf.layers.dense(self.embed_layer, self.vocab_size, use_bias=False)
		
		# calculate loss, optimize, and train
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels))
		self.optimizer = tf.train.AdamOptimizer(1e-4)
		self.train_op = self.optimizer.minimize(self.loss)

	def evaluate(self, features, labels, sess):
		"""
        Returns the loss of the word2vec model with params:
        	- features: array of features as input
        	- labels: array of labels as output
        	- sess: a tensorflow session
        """
		loss = sess.run(self.loss, feed_dict={self.input: features, self.labels: labels})
		return loss

	def train_step(self, features, labels, sess):
		"""
        Trains one step of the model with params:
        	- features: array of features as input
        	- labels: array of labels as output
        	- sess: a tensorflow session
        """
		loss, _ = sess.run([self.loss, self.train_op], 
												feed_dict={self.input: features, self.labels: labels.astype(np.float64)})
		return loss

	def get_output(self, features, sess):
		"""
        Returns the output of the model
        	- features: array of features as input
        	- labels: array of labels as output
        	- sess: a tensorflow session
        """
		output = sess.run(self.output, feed_dict={self.input: features})
		return output

	def get_embed(self):
		"""
        Returns the weights of the first fully connected layer as the word embedding
        """
		weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.embed_layer.name)[0] + '/kernel:0')
		return weights