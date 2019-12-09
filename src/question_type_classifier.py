import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class QTypeClassifier():

	def __init__ (self, n_hidden=512, embed_size=300, dense_size = 3, lr=1e-3, loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits): 
		self.n_hidden = n_hidden
		self.embed_size = embed_size
		self.dense_size = dense_size
		self.loss_fn = loss_fn
		self.lr = lr
		self._build_graph()

	def _build_graph(self):
		self.q_batch = tf.placeholder(tf.float64, [None, None, self.embed_size])
		self.q_batch = tf.stop_gradient(self.q_batch)
		self.rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden), rnn.BasicLSTMCell(self.n_hidden)])
		self.embed_input = tf.layers.dense(self.q_batch, self.embed_size, activation=tf.tanh, name="embed_input")
		_, states = tf.nn.dynamic_rnn(self.rnn_cell, self.embed_input, dtype=tf.float64)

		self.q_embedding = tf.concat([states[1][0], states[1][1]], axis=1)
		self.output = tf.layers.dense(self.q_embedding, self.dense_size, name="output")
		print("Output.shape: ",self.output.shape)
		self.labels = tf.placeholder(tf.int32, [None], name="labels")
		self.labels = tf.stop_gradient(self.labels)
		print("Label.shape: ",self.labels.shape)
		self.loss = tf.reduce_mean(self.loss_fn(labels=self.labels, logits=self.output))
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

	def train_step(self, sess, q_batch, label_batch):
		"""
        Trains the FCNN model with parameters:
            - sess: a tensorflow session
			- cnn_batch: a batch of training images
			- q_batch: a batch of training questions
			- label_batch: a batch of trainin labels
        """
		_, loss = sess.run([self.train_op, self.loss], feed_dict={self.q_batch: q_batch, self.labels: label_batch})
		return loss

	def get_output(self, sess, q_in, label_in):
		"""
        Tests and returns the output of the FCNN model with:
            - sess: a tensorflow session
			- cnn_in: a test image
			- q_batch: a test question
			- label_in: a test label
        """
		output = sess.run(self.output, feed_dict={self.q_batch: q_in, self.labels: label_in})
		return output

	def evaluate(self, sess, q_batch, label_batch):
		"""
        Evaluates loss of the FCNN model with parameters:
            - sess: a tensorflow session
			- cnn_batch: a batch of images to evaluate on
			- q_batch: a batch of questions to evaluate on
			- label_batch: a batch of labels to evaluate on
        """
		loss = sess.run(self.loss, feed_dict={self.q_batch: q_batch, self.labels: label_batch})
		return loss


