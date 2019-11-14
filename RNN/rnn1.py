import tensorflow as tf
import pandas as pd
import numpy as np
import collections


class RNNModel():

	#Creating dataset
	words = "long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies ."

	def build_dataset(words = words):
		count = collections.Counter(words).most_common()
		dictionary = dict()
		for word, _ in count:
			dictionary[word] = len(dictionary)
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return dictionary, reverse_dictionary

	dictionary = build_dataset(words)
	vocab_size = len(dictionary)

	# Training Parameters
	learning_rate = 0.001
	training_steps = 100
	batch_size = 10
	display_step = 200

	# Network Parameters
	num_input = 3
	timesteps = 3 # timesteps CHANGE?
	num_hidden = 512 # hidden layer num of features
	num_classes = vocab_size #num of unique words

	# tf Graph input
	X = tf.placeholder("float", [None, timesteps, num_input])
	Y = tf.placeholder("float", [None, num_classes])

	# RNN output node weights and biases
	weights = {
		'out': tf.Variable(tf.random_normal([num_hidden, vocab_size]))
	}
	biases = {
		'out': tf.Variable(tf.random_normal([vocab_size]))
	}

	def RNN(x, weights, biases):

		# reshape to [1, num_input]
		x = tf.reshape(x, [-1, num_input])

		# Generate a num_input-element sequence of inputs
		# (eg. [had] [a] [general] -> [20] [6] [33])
		x = tf.split(x,num_input,1)

		# 1-layer LSTM with num_hidden units.
		rnn_cell = rnn.BasicLSTMCell(num_hidden)

		# generate prediction
		outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

		# there are num_input outputs but
		# we only want the last output
		return tf.matmul(outputs[-1], weights['out']) + biases['out']
	
	symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+num_input) ]

	symbols_out_onehot = np.zeros([vocab_size], dtype=float)
	symbols_out_onehot[dictionary[str(training_data[offset+num_input])]] = 1.0

	pred = RNN(x, weights, biases)

	# Loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=symbols_out_onehot))
	optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

	_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})