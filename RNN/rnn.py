import tensorflow as tf
import pandas as pd
import numpy as np

class RNNModel():

	def __init__(self, r_neuron=10, n_windows=20, n_input=1, n_output=1):
	
		X = tf.placeholder(tf.float32, [None, n_windows, n_input])   
		y = tf.placeholder(tf.float32, [None, n_windows, n_output])

		basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=r_neuron, activation=tf.nn.relu)   
		rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)              

		stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])          
		stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)       
		outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])


		#TODO: GET RID OF 

		def create_ts(start = '2001', n = 201, freq = 'M'):
			rng = pd.date_range(start=start, periods=n, freq=freq)
			ts = pd.Series(np.random.uniform(-18, 18, size=len(rng)), rng).cumsum()
			return ts
		ts = create_ts(start = '2001', n = 192, freq = 'M')
		ts.tail(5)

		series = np.array(ts)

		print(series)
		print(len(series))
		
		size_train = 150
		train = series[:size_train]
		test = series[size_train:]
		
		def create_batches(df, windows, input, output):
			## Create X         
			x_data = train[:size_train-1] # Select the data
			X_batches = x_data.reshape(-1, windows, input)  # Reshape the data 
			## Create y
			y_data = train[n_output:size_train]
			y_batches = y_data.reshape(-1, windows, output)
			return X_batches, y_batches

		X_batches, y_batches = create_batches(df = train,
									  windows = n_windows,
									  input = n_input,
									  output = n_output)

		x_data = train[:size_train-1]
		X_batches = x_data.reshape(-1, windows, input)
	

	def train(self):

		learning_rate = 0.001  
		loss = tf.reduce_sum(tf.square(outputs - y))    
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)         
		training_op = optimizer.minimize(loss)  

		init = tf.global_variables_initializer() 
		iteration = 1500 

		with tf.Session() as sess:
			init.run()
			for iters in range(iteration):
				sess.run(training_op, feed_dict={X: X_batches, y: y_batches})
				if iters % 150 == 0:
					mse = loss.eval(feed_dict={X: X_batches, y: y_batches})
					print(iters, "\tMSE:", mse)
			
			y_pred = sess.run(outputs, feed_dict={X: X_test})

	def predict(self, test_q_arr, test_q_ids):
		return preds

model = RNNModel()