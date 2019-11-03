import tensorflow as tf 
import numpy as np 
import random





class FCNN:
	def __init__(self, cnn_input, rnn_input, pointwise_layer_size=1024, net_struct={'h1': 1000}, 
			     output_size=1000, initializer=tf.random_normal_initializer, activation_fn=tf.nn.relu, 
			     loss_fn=tf.nn.softmax_cross_entropy_with_logits, lr=1e-4, scope='FCNN'):

		self.cnn_input = cnn_input
		self.rnn_input = rnn_input
		self.cnn_input_size = tf.size(cnn_input)
		self.rnn_input_size = tf.size(rnn_input)
		self.pointwise_layer_size = pointwise_layer_size
		self.net_struct = net_struct
		self.output_size = output_size
		self.initializer = initializer
		self.activation_fn = activation_fn
		self.loss_fn = loss_fn
		self.lr = lr
		self.scope = scope

		self.cnn_in, self.rnn_in, self.y, self.output, self.loss, self.train_op = self._build_model()

	def train(self, sess, batched_cnn_inputs, batched_rnn_inputs, batched_outputs, epochs=100000, verbose=True):
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		num_batches = len(batched_outputs)

		for epoch in range(epochs):
			# get random batch
			i = random.randint(0, num_batches)
			cnn_batch, rnn_batch, y_batch = batched_cnn_inputs[i], batched_rnn_inputs[i], batched_outputs[i]
			step_loss = self._train_step(cnn_batch, rnn_batch, y_batch)
			if verbose:
				if epoch % 100:
					print("Loss:", step_loss)


	def predict(self, cnn_in, rnn_in):
		return self.sess.run(self.output, feed_dict={self.cnn_in: cnn_in, self.rnn_in: rnn_in})


	def _train_step(self, cnn_batch, rnn_batch, y_batch):
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




	# def _make_dense_nn(scope, n_inputs, n_outputs, h_layer_node_dict, initializer, activation_fn, loss_fn, lr=1e-4):
	# """
	# Construct dense neural network using tf.layers.
	# 	Args:
	# 		scope - scope under which to define tf variables 
	# 		n_inputs - number of inputs to network
	# 		n_outputs - number of outputs of network
	# 		h_layer_node_dict - dictionary of form: {"<layer_name>": <nodes_in_layer>, ...}
	# 		initializer: function to initialize weights
	# 		activation_fn: activation functions for layers
	# 		loss_fn: loss function for network
	# 		lr: learning rate of layers
	# 	Returns: 
	# 		x - input placeholder
	# 		y - output placeholder
	# 		output - output of network
	# 		loss - loss of network
	# 		train_op - training operation of network
	# """
	# 	with tf.variable_scope(scope):
	# 		x = tf.placeholder(tf.float32, [None, n_inputs], name="x")
	# 		y = tf.placeholder(tf.float32, [None, n_outputs], name="y")
	        
	# 		prev_layer = x
	# 		for layer_name, layer_nodes in h_layer_node_dict.items():
	# 			prev_layer = tf.layers.dense(prev_layer, 	
	#             							 layer_nodes, 
	#             							 activation=activation_fn, 
	#             							 kernel_initializer=initializer, 
	#             							 name=layer_name)
	        
	# 		output = tf.layers.dense(prev_layer, n_outputs, name="outputs")
	        
	# 		return x, y, output, loss, train_op




