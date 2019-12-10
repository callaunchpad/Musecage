import tensorflow as tf 
import numpy as np 
import random
from rnn_model import RNNModel

class AttentionModel:
	def __init__(self, cnn_input_size, cnn_in, q_batch, embed_size=300, activation_fn=tf.nn.relu):
		"""
        Creates AttentionModel based on:
            - cnn_input_size: dimension of image embedding
			- embed_size: dimension of word2vec embedding
        """
		
		self.cnn_input_size = cnn_input_size
		self.cnn_in = cnn_in
		self.q_batch = q_batch
		self.embed_size = embed_size
		self.activation_fn = activation_fn
		self.output = self.build_graph()

	def build_graph(self):
		"""
        Builds graph of AttentionModel, called only during __init__. 
        """
		self.embed_output = tf.stop_gradient(self.q_batch)

		self.cnn_l2_reg = tf.nn.l2_normalize(tf.stop_gradient(self.cnn_in))

		self.cnn_dense = tf.layers.dense(self.cnn_l2_reg, self.embed_size, activation=self.activation_fn, name='cnn_in_layer')

		self.pointwise_layer = tf.multiply(self.embed_output, tf.expand_dims(self.cnn_dense, axis=1), name="pointwise_layer")

		self.attention_mask = tf.nn.softmax(self.pointwise_layer, axis=2)

		self.attention_image = tf.multiply(self.attention_mask, tf.expand_dims(self.cnn_dense, axis=1))
		self.image_word_vec = tf.concat([self.attention_image, self.q_batch], axis=1)

		return self.image_word_vec














