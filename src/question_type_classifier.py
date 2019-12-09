import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from pipeline import Pipeline
from load_create_data import *
import time

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
		print(self.output.shape)
		self.labels = tf.placeholder(tf.int32, [None], name="labels")
		self.labels = tf.stop_gradient(self.labels)
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

num_epochs = 10
save=True
save_freq=100
savedir="../model_/"
verbose=True
verbose_freq=10
data_len = 90000


data_arr = (get_by_ques_type([], train=True) + get_by_ques_type([], train=False))[:data_len]

p = Pipeline(data_arr, embed_type = "Word2Vec")
p.create_split()


train_step = 0
curr_samples = 0

train_losses = []
test_losses = []

classifier = QTypeClassifier(n_hidden=512, embed_size=300, dense_size = 3)


sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

for epoch in range(num_epochs):
	while p.next_batch(train=True, replace=False):
		start_time = time.time()
		train_qs, train_ims, train_ans, ans_types, all_ans = p.batch_attention()
		ans_type_dict = {"yes/no": 0, "number": 1, "other": 2}
		ans_types = [ans_type_dict[i] for i in ans_types]

		# print("shapes: ", np.array(train_qs).shape, np.array(ans_types).shape)

		train_step += 1
		batch_samples = len(train_qs)
		curr_samples += batch_samples

		if len(train_qs) > 0:
			train_loss = classifier.train_step(sess, np.array(train_qs), np.array(ans_types))
			train_losses.append(train_loss)
			
		p.next_batch(train=False, replace=True)
		test_qs, test_ims, test_ans, ans_types, all_ans = p.batch_attention()
		
		ans_type_dict = {"yes/no": 0, "number": 1, "other": 2}
		ans_types = [ans_type_dict[i] for i in ans_types]

		if len(test_qs) > 0:
			test_loss = classifier.evaluate(sess,np.array(test_qs), np.array(ans_types))
			test_losses.append(test_loss)
	
		if train_step % save_freq == 0 and save:
			tf.train.Saver().save(sess, savedir+"%d"%(train_step), global_step=train_step)
			np.savez(savedir+"train_losses_%d.npz"%(train_step), np.array(train_losses))
			np.savez(savedir+"test_losses_%d.npz"%(train_step), np.array(test_losses))

		end_time = time.time()
		if train_step % verbose_freq == 0 and verbose:
			print("TRAIN STEP: %d | SAMPLES IN TRAIN BATCH: %d | TRAIN SAMPLES SO FAR: %d | TRAIN LOSS: %f | TEST LOSS: %f" %(train_step, batch_samples, curr_samples, train_loss, test_loss))
			print("Time elapsed: ", end_time - start_time, " seconds")
	if verbose:
		print("********************FINISHED EPOCH %d********************"%(epoch))
	p.reset_batch(train=True)

if save:
	tf.train.Saver().save(sess, savedir+"%d"%(train_step), global_step=train_step)
	np.savez(savedir+"train_losses_%d.npz"%(train_step), np.array(train_losses))
	np.savez(savedir+"test_losses_%d.npz"%(train_step), np.array(test_losses))

