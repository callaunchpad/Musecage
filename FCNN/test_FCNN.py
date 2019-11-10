from FCNN import FCNN

import tensorflow as tf


if __name__ == "__main__":
	fcnn = FCNN(cnn_input_size=1024, rnn_input_size=1024, pointwise_layer_size=1024, output_size=1000)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())