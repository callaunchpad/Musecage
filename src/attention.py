import tensorflow as tf 
import numpy as np 
import random
from rnn_model import RNNModel

class AttentionModel:
	def __init__(self, cnn_input_size, word2vec_embed_size=300):
		"""
        Creates AttentionModel based on:
            - cnn_input_size: dimension of image embedding
			- word2vec_embed_size: dimension of word2vec embedding
        """
		
		self.cnn_input_size = cnn_input_size
		self.word2vec_embed_size = word2vec_embed_size

		self.build_graph()

	def build_graph(self):
		"""
        Builds graph of AttentionModel, called only during __init__. 
        """

		self.cnn_in = tf.placeholder(tf.float64, [None, self.cnn_input_size], name="cnn_input")

		self.q_batch = tf.placeholder(tf.float64, [None, None, self.word2vec_embed_size])
		self.embed_output = tf.stop_gradient(self.q_batch)

		self.cnn_l2_reg = tf.nn.l2_normalize(tf.stop_gradient(self.cnn_in))

		self.cnn_dense = tf.layers.dense(self.cnn_l2_reg, self.word2vec_embed_size, activation=self.activation_fn, name='cnn_in_layer')

		self.pointwise_layer = tf.multiply(self.cnn_dense, self.embed_output, name="pointwise_layer")

		self.attention_mask = tf.nn.softmax(self.pointwise_layer, axis=2)

		self.attention_image = tf.multiply(self.attention_mask, self.cnn_in)
		self.image_word_vec = tf.concat(self.attention_image, self.q_batch)

		return image_word_vec














