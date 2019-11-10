from FCNN import FCNN
import numpy as np
import tensorflow as tf


if __name__ == "__main__":
	fcnn = FCNN(cnn_input_size=1024, rnn_input_size=1024, pointwise_layer_size=1024, output_size=1000)
	cnn_in = np.zeros((1,1,1024))
	rnn_in = np.zeros((1,1,1024))
	outputs = np.zeros((1,1,1000))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		fcnn.train(sess, batched_cnn_inputs=cnn_in, batched_rnn_inputs=rnn_in, batched_outputs=outputs, save_model_loc="", checkpoint_freq=10, epochs=100, verbose=True)
		print(fcnn.predict(sess, cnn_in[0], rnn_in[0]))