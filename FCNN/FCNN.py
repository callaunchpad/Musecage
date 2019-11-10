import tensorflow as tf 
import numpy as np 
import random

class FCNN:
	def __init__(self, cnn_input_size, rnn_input_size, pointwise_layer_size, output_size, net_struct={'h1': 1000}, 
			           initializer=tf.random_normal_initializer, activation_fn=tf.nn.relu, 
			           loss_fn=tf.nn.softmax_cross_entropy_with_logits_v2, lr=1e-4, scope='FCNN'):

		self.cnn_input_size = cnn_input_size
		self.rnn_input_size = rnn_input_size
		self.pointwise_layer_size = pointwise_layer_size
		self.output_size = output_size
		self.net_struct = net_struct
		self.initializer = initializer
		self.activation_fn = activation_fn
		self.loss_fn = loss_fn
		self.lr = lr
		self.scope = scope

		self.cnn_in, self.rnn_in, self.y, self.output, self.loss, self.train_op = self._build_model()

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


	def predict(self, sess, cnn_in, rnn_in):
		return sess.run(self.output, feed_dict={self.cnn_in: cnn_in, self.rnn_in: rnn_in})

	# TODO: implement
	# def _save_model(self, loc):


	def _train_step(self, sess, cnn_batch, rnn_batch, y_batch):
		_, step_loss = sess.run([self.train_op, self.loss], feed_dict={self.cnn_in: cnn_batch, self.rnn_in: rnn_batch, self.y: y_batch})
		return step_loss
	
	def _build_model(self):

		with tf.variable_scope(self.scope):

			cnn_in = tf.placeholder(tf.float32, [None, self.cnn_input_size], name="cnn_input")
			rnn_in = tf.placeholder(tf.float32, [None, self.rnn_input_size], name="rnn_input")
			y = tf.placeholder(tf.float32, [None, self.output_size], name="y")

			cnn_dense = tf.layers.dense(cnn_in, self.pointwise_layer_size, activation=self.activation_fn, kernel_initializer=self.initializer, name='cnn_in_layer')
			rnn_dense = tf.layers.dense(rnn_in, self.pointwise_layer_size, activation=self.activation_fn, kernel_initializer=self.initializer, name='rnn_in_layer')
			pointwise_layer = tf.math.multiply(cnn_dense, rnn_dense, name="pointwise_layer")

			prev_layer = pointwise_layer
			for layer_name, layer_nodes in self.net_struct.items():
				prev_layer = tf.layers.dense(prev_layer, layer_nodes, 
											 activation=self.activation_fn, 
											 kernel_initializer=self.initializer, name=layer_name)

			output = tf.layers.dense(prev_layer, self.output_size, activation=tf.nn.softmax, kernel_initializer=self.initializer, name="output")


			loss = self.loss_fn(labels=y, logits=output)
			train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

		return cnn_in, rnn_in, y, output, loss, train_op


