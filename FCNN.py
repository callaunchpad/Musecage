import tensorflow as tf 
import numpy as np 
import random
from rnn_model import RNNModel

class FCNN:
	def __init__(self, cnn_input_size, rnn_input_size, pointwise_layer_size, output_size, vocab_size, net_struct={'h1': 1000}, 
			           initializer=tf.random_normal_initializer, activation_fn=tf.nn.relu, embed_type="RNN",
			           loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits, lr=1e-1):

		self.cnn_input_size = cnn_input_size
		self.rnn_input_size = rnn_input_size
		self.pointwise_layer_size = pointwise_layer_size
		self.output_size = output_size
		self.vocab_size = vocab_size
		self.net_struct = net_struct
		self.initializer = initializer
		self.activation_fn = activation_fn
		self.embed_type = embed_type
		self.loss_fn = loss_fn
		self.lr = lr

		tf.reset_default_graph() 
		self._build_model()

	def train(self, sess, batched_cnn_inputs, batched_rnn_inputs, batched_outputs, 
			        save_model_loc="", checkpoint_freq=100, epochs=10000, verbose=True):

		num_batches = len(batched_outputs)

		for epoch in range(epochs):
			# get random batch
			i = random.randint(0, num_batches-1)
			cnn_batch, rnn_batch, y_batch = batched_cnn_inputs[i], batched_rnn_inputs[i], batched_outputs[i]
			step_loss = self._train_step(sess, cnn_batch, rnn_batch, y_batch)
			if verbose:
				if epoch % checkpoint_freq:
					print("Epoch: ", epoch, "Loss:", step_loss)
					if save_model_loc:
						print("Saving Model...")
						print("================================")
						self._save_model(save_model_loc)


	def predict(self, sess, cnn_in, q_batch):
		return sess.run(self.output, feed_dict={self.cnn_in: cnn_in, self.q_batch: q_batch})

	# TODO: implement
	# def _save_model(self, loc):


	def _train_step(self, sess, cnn_batch, q_batch, label_batch):
		_, step_loss, output, grads = sess.run([self.train_op, self.loss, self.output, self.grads], feed_dict={self.cnn_in: cnn_batch, self.q_batch: q_batch, self.labels: label_batch})
		return step_loss, output, grads
	
	def _build_model(self):
		self.cnn_in = tf.placeholder(tf.float64, [None, self.cnn_input_size], name="cnn_input")
		if self.embed_type == "RNN":
			self.q_batch = tf.placeholder(tf.int32, [None, None], name="q_batch")
		elif self.embed_type == "GloVe":
			self.q_batch = tf.placeholder(tf.float64, [None, 300])
		self.labels = tf.placeholder(tf.int32, [None], name="labels")

		if self.embed_type == "RNN":
			self.q_batch = tf.stop_gradient(self.q_batch)
			self.one_hot = tf.one_hot(self.q_batch, self.vocab_size, dtype=tf.float64)

			rnn = RNNModel(self.one_hot)
			self.embed_output = rnn.output
		elif self.embed_type == "GloVe":
			self.embed_output = tf.stop_gradient(self.q_batch)

		self.cnn_l2_reg = tf.nn.l2_normalize(tf.stop_gradient(self.cnn_in))
		self.cnn_dense = tf.layers.dense(self.cnn_l2_reg, self.pointwise_layer_size, activation=self.activation_fn, name='cnn_in_layer')
		self.q_dense = tf.layers.dense(self.embed_output, self.pointwise_layer_size, activation=self.activation_fn, name='rnn_in_layer')
		self.pointwise_layer = tf.multiply(self.cnn_dense, self.q_dense, name="pointwise_layer")

		self.prev_layer = self.pointwise_layer
		for layer_name, layer_nodes in self.net_struct.items():
			prev_layer = tf.layers.dense(self.prev_layer, layer_nodes, 
										activation=self.activation_fn, 
										name=layer_name)

		self.output = tf.layers.dense(self.prev_layer, self.output_size, 
										activation=self.activation_fn,
										name="output")

		self.labels = tf.stop_gradient(self.labels)
		self.loss = tf.reduce_mean(self.loss_fn(labels=self.labels, logits=self.output))
		self.grads = tf.gradients(self.loss, self.output)
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)















