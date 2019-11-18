import tensorflow as tf 
import numpy as np 
import random
from rnn_model import RNNModel

class FCNN:
	def __init__(self, cnn_input_size, rnn_input_size, pointwise_layer_size, output_size, vocab_size, 
					glove_embed_size=300, word2vec_embed_size=300, dropout=0.2, net_struct={'h1': 1000}, 
			        activation_fn=tf.nn.relu, embed_type="RNN",
			        loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits, lr=1e-3):
		
		self.cnn_input_size = cnn_input_size
		self.rnn_input_size = rnn_input_size
		self.pointwise_layer_size = pointwise_layer_size
		self.output_size = output_size
		self.vocab_size = vocab_size
		self.glove_embed_size = glove_embed_size 
		self.word2vec_embed_size = word2vec_embed_size
		self.dropout = dropout
		self.net_struct = net_struct
		self.activation_fn = activation_fn
		self.embed_type = embed_type
		self.loss_fn = loss_fn
		self.lr = lr 

		self.build_model()

	def build_model(self):
		self.cnn_in = tf.placeholder(tf.float64, [None, self.cnn_input_size], name="cnn_input")

		if self.embed_type == "RNN":
			self.q_batch = tf.placeholder(tf.int32, [None, None], name="q_batch")
		elif self.embed_type == "GloVe":
			self.q_batch = tf.placeholder(tf.float64, [None, self.glove_embed_size])
		elif self.embed_type == "Word2Vec":
			self.q_batch = tf.placeholder(tf.float64, [None, self.word2vec_embed_size])
			
		self.labels = tf.placeholder(tf.int32, [None], name="labels")

		if self.embed_type == "RNN":
			self.q_batch = tf.stop_gradient(self.q_batch)
			self.one_hot = tf.one_hot(self.q_batch, self.vocab_size, dtype=tf.float64)

			rnn = RNNModel(self.one_hot)
			self.embed_output = rnn.output
			self.embed_output = tf.nn.l2_normalize(self.embed_output)
		elif self.embed_type == "GloVe":
			self.embed_output = tf.stop_gradient(self.q_batch)
		elif self.embed_type == "Word2Vec":
			self.embed_output = tf.stop_gradient(self.q_batch)

		self.cnn_l2_reg = tf.nn.l2_normalize(tf.stop_gradient(self.cnn_in))
		self.cnn_dense = tf.layers.dense(self.cnn_l2_reg, self.pointwise_layer_size, activation=self.activation_fn, name='cnn_in_layer')
		self.q_dense = tf.layers.dense(self.embed_output, self.pointwise_layer_size, activation=self.activation_fn, name='rnn_in_layer')
		self.pointwise_layer = tf.multiply(self.cnn_dense, self.q_dense, name="pointwise_layer")

		self.pointwise_layer = tf.nn.dropout(self.pointwise_layer, self.dropout)
		self.prev_layer = self.pointwise_layer
		for layer_name, layer_nodes in self.net_struct.items():
			self.prev_layer = tf.layers.dense(self.prev_layer, layer_nodes, 
										activation=self.activation_fn, 
										name=layer_name)
			self.prev_layer = tf.nn.dropout(self.prev_layer, self.dropout)

		self.output = tf.layers.dense(self.prev_layer, self.output_size, 
										activation=self.activation_fn,
										name="output")

		self.labels = tf.stop_gradient(self.labels)
		self.loss = tf.reduce_mean(self.loss_fn(labels=self.labels, logits=self.output))
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

	def evaluate(self, sess, cnn_batch, q_batch, label_batch):
		loss = sess.run(self.loss, feed_dict={self.cnn_in: cnn_batch, self.q_batch: q_batch, self.labels: label_batch})
		return loss

	def train_step(self, sess, cnn_batch, q_batch, label_batch):
		_, loss = sess.run([self.train_op, self.loss], feed_dict={self.cnn_in: cnn_batch, self.q_batch: q_batch, self.labels: label_batch})
		return loss

	def get_output(self, sess, cnn_in, q_in, label_in):
		output = sess.run(self.output, feed_dict={self.cnn_in: cnn_in, self.q_batch: q_in, self.labels: label_in})
		return output














