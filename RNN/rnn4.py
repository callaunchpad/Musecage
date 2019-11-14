import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import collections


class RNNModel():

	def __init__ (self, question, n_hidden=512, n_input=3): 

		# Network Parameters
		self.n_input = n_input
		self.n_hidden = n_hidden # hidden layer num of features

		question_list = question.split()

		def build_dataset(words):
			count = collections.Counter(words).most_common()
			dictionary = dict()
			for word, _ in count:
				dictionary[word] = len(dictionary)
			reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
			return dictionary, reverse_dictionary

		self.dictionary, self.reverse_dictionary = build_dataset(question_list)
		self.vocab_size = len(self.dictionary)

		self.weights = {
			'out': tf.Variable(tf.random_normal([self.n_hidden, self.vocab_size]))
		}
		self.biases = {
			'out': tf.Variable(tf.random_normal([self.vocab_size]))
		}

		# tf Graph input
		self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input, 1], name="X")
		self.y = tf.placeholder(dtype=tf.float32, shape=[None, vocab_size], name="Y")
		self.pred, self.cost, self.optimizer = self._build_graph()

	def _build_graph(self):
		rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])
		outputs, states = tf.nn.dynamic_rnn(rnn_cell, self.x, dtype=tf.float32)
		pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
		print("Output shape:", tf.shape(outputs))
		print("States shape:", tf.shape(states))
		
		# Loss and optimizer
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
		optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)

		return pred, cost, optimizer

	def train(self):
		# Initializing the variables
		init = tf.global_variables_initializer()

		# Launch the graph
		with tf.Session() as session:
			session.run(init)
			step = 0
			offset = random.randint(0,n_input+1)
			end_offset = n_input + 1
			acc_total = 0
			loss_total = 0

			#writer.add_graph(session.graph)

			while step < training_iters:
				# Generate a minibatch. Add some randomness on selection process.
				if offset > (len(training_data)-end_offset):
					offset = random.randint(0, n_input+1)

				symbols_in_keys = [ [self.dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
				symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, self.n_input, 1])

				symbols_out_onehot = np.zeros([self.vocab_size], dtype=float)
				symbols_out_onehot[self.dictionary[str(training_data[offset+self.n_input])]] = 1.0
				symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

				_, acc, loss, onehot_pred = session.run([self.optimizer, self.cost, self.pred], \
														feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
				loss_total += loss
				if (step+1) % display_step == 0:
					print("Iter= " + str(step+1) + ", Average Loss= " + \
						"{:.6f}".format(loss_total/display_step))
					loss_total = 0
					symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
					symbols_out = training_data[offset + n_input]
					# print(onehot_pred)
					symbols_out_pred = self.reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval()[0])]
					print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
				step += 1
				offset += (n_input+1)


rnn = RNNModel("did the mice fail the midterm")
pred = rnn.build_graph()








	# def RNN(self, x):
	#     # reshape to [1, n_input]
	#     #x = tf.reshape(x, [-1, self.n_input])

	#     # Generate a n_input-element sequence of inputs
	#     # (eg. [had] [a] [general] -> [20] [6] [33])
	#     #x = tf.split(x,self.n_input,1)

	#     # 2-layer LSTM, each layer has n_hidden units.
	#     # Average Accuracy= 95.20% at 50k iter
	#     rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])

	#     # 1-layer LSTM with n_hidden units but with lower accuracy.
	#     # Average Accuracy= 90.60% 50k iter
	#     # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
	#     # rnn_cell = rnn.BasicLSTMCell(n_hidden)

	#     # generate prediction
	#     outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

	#     # there are n_input outputs but
	#     # we only want the last output
	#     return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
	
	