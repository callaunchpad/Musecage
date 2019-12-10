import tensorflow as tf 
import numpy as np 
import random
from attention import AttentionModel
from rnn_model import RNNModel

class AttentionRNN:
	def __init__(self, cnn_input_size, output_size, embed_size=300, n_hidden=512, dropout=0.3, net_struct={'h1': 1000}, 
					activation_fn=tf.nn.relu, loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits, 
					start_lr=1e-3, boundaries=[200, 400, 600, 2000, 4000], factors=[1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]):
		"""
		Creates AttentionModel based on:
			- cnn_input_size: dimension of image embedding
			- n_hidden
			- output_size: dimension of the output layer
			- word2vec_embed_size: dimension of word2vec embedding
			- dropout: rate of dropout layer
			- net_struct: dictionary of {name: layer_size} for each layer in the fully connected layers
			- activation_fn: activation function
			- embed_type: type of question embedding used; can be "Glove", "Word2Vec", or "RNN"
			- loss_fn: loss function
			- lr: learning rate
		"""
		
		self.cnn_input_size = cnn_input_size
		self.n_hidden = n_hidden
		self.output_size = output_size
		self.embed_size = embed_size
		self.dropout = dropout
		self.net_struct = net_struct
		self.activation_fn = activation_fn
		self.loss_fn = loss_fn
		self.start_lr = start_lr
		self.boundaries = boundaries
		self.values = list(self.start_lr * np.array(factors))

		self.build_model()

	def build_model(self):
		"""
		Builds graph of FCNN, called only during __init__.
		"""
		self.cnn_in = tf.placeholder(tf.float64, [None, self.cnn_input_size], name="cnn_input")
		self.q_batch = tf.placeholder(tf.float64, [None, None, self.embed_size])
		self.labels = tf.placeholder(tf.int32, [None], name="labels")

		self.attention_vec = AttentionModel(self.cnn_input_size, self.cnn_in, self.q_batch, embed_size=self.embed_size).output

		self.rnn_out = RNNModel(self.attention_vec, dense=False).output

		# self.rnn_out = tf.layers.dropout(self.rnn_out, self.dropout)
		self.prev_layer = self.rnn_out
		for layer_name, layer_nodes in self.net_struct.items():
			self.prev_layer = tf.layers.dense(self.prev_layer, layer_nodes, 
										activation=self.activation_fn, 
										name=layer_name)
			# self.prev_layer = tf.layers.dropout(self.prev_layer, self.dropout)

		self.output = tf.layers.dense(self.prev_layer, self.output_size, 
										activation=self.activation_fn,
										name="output")

		self.labels = tf.stop_gradient(self.labels)
		self.loss = tf.reduce_mean(self.loss_fn(labels=self.labels, logits=self.output))
		# self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.train.piecewise_constant(self.global_step, self.boundaries, self.values)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)


	def evaluate(self, sess, cnn_batch, q_batch, label_batch):
		"""
		Evaluates loss of the FCNN model with parameters:
			- sess: a tensorflow session
			- cnn_batch: a batch of images to evaluate on
			- q_batch: a batch of questions to evaluate on
			- label_batch: a batch of labels to evaluate on
		"""
		loss = sess.run(self.loss, feed_dict={self.cnn_in: cnn_batch, self.q_batch: q_batch, self.labels: label_batch})
		return loss

	def train_step(self, sess, cnn_batch, q_batch, label_batch):
		"""
		Trains the FCNN model with parameters:
			- sess: a tensorflow session
			- cnn_batch: a batch of training images
			- q_batch: a batch of training questions
			- label_batch: a batch of trainin labels
		"""
		_, loss = sess.run([self.train_op, self.loss], feed_dict={self.cnn_in: cnn_batch, self.q_batch: q_batch, self.labels: label_batch})
		return loss

	def get_output(self, sess, cnn_in, q_in, label_in):
		"""
		Tests and returns the output of the FCNN model with:
			- sess: a tensorflow session
			- cnn_in: a test image
			- q_in: a test question
			- label_in: a test label
		"""
		output = sess.run(self.output, feed_dict={self.cnn_in: cnn_in, self.q_batch: q_in, self.labels: label_in})
		return output















