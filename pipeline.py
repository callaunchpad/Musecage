from __future__ import print_function

import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.backend import set_session

from load_create_data import *
from knn import KNNModel
from word2vec import Word2Vec
from rnn_model import RNNModel
from FCNN import FCNN

class Pipeline():
    def __init__(self, data_arr, metric="min_k", embed_type="RNN", batch_size=10):
        self.data_arr = data_arr
        self.metric = metric
        self.batch_size = batch_size
        self.train_curr_index = 0
        self.test_curr_index = 0
        self.embed_type = embed_type
        if self.embed_type == "RNN":
            self.embed_index = load_glove()
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.vision_model = VGG16(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000)
        
        
    def create_split(self, split_val=.8, custom_split=False, custom_train=[], custom_test=[], build_top_vocab=True, top_k=1000):
        if custom_split:
            self.train_data = custom_train
            self.test_data = custom_test
        else:
            self.train_data = self.data_arr[:int(len(self.data_arr) * split_val)]
            self.test_data = self.data_arr[int(len(self.data_arr) * split_val):]

        self.train_q_arr = [format_q_for_embed(val["question"]) for val in self.train_data]
        self.test_q_arr = [format_q_for_embed(val["question"]) for val in self.test_data]

        if build_top_vocab:
            allwords = []
            for q in self.train_q_arr + self.test_q_arr:
                allwords.extend(q.split(" "))
            c = Counter(allwords)
            top_k_words = c.most_common(top_k)
            self.top_k_q_dict = {}
            for ind in range(len(top_k_words)):
                self.top_k_q_dict[top_k_words[ind][0]] = ind
            self.top_k_q_dict["__END__"] = len(top_k_words)

        self.train_q_id_arr = [val["question_id"] for val in self.train_data]
        self.test_q_id_arr = [val["question_id"] for val in self.test_data]

        self.train_im_id_arr = [val["image_id"] for val in self.train_data]
        self.test_im_id_arr = [val["image_id"] for val in self.test_data]
        
        self.train_ans_arr = [val["answers"] for val in self.train_data]
        self.test_ans_arr = [val["answers"] for val in self.test_data]

        self.train_ans_type = [val["answer_type"] for val in self.train_data]
        self.test_ans_type = [val["answer_type"] for val in self.test_data]

        if build_top_vocab:
            allwords = []
            for q in list(np.array(self.train_ans_arr).flatten()) + list(np.array(self.test_ans_arr).flatten()):
                allwords.extend(q.split(" "))
            c = Counter(allwords)
            top_k_words = c.most_common(top_k)
            self.top_k_ans_dict = {}
            for ind in range(len(top_k_words)):
                self.top_k_ans_dict[top_k_words[ind][0]] = ind

    def next_batch(self, train=True, replace=False):
        next_batch_avail = True
        if train:
            if not replace:
                self.q_batch = self.train_q_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.q_id_batch = self.train_q_id_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.im_id_batch = self.train_im_id_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.ans_batch = self.train_ans_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.train_curr_index += self.batch_size
                if self.train_curr_index == len(self.train_q_arr):
                    next_batch_avail = False
            else:
                self.q_batch = random.sample(self.train_q_arr, self.batch_size)
                self.q_id_batch = random.sample(self.train_q_id_arr, self.batch_size)
                self.im_id_batch = random.sample(self.train_im_id_arr, self.batch_size)
                self.ans_batch = random.sample(self.train_ans_arr, self.batch_size)
        else:
            if not replace:
                self.q_batch = self.test_q_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.q_id_batch = self.test_q_id_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.im_id_batch = self.test_im_id_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.ans_batch = self.test_ans_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.test_curr_index += self.batch_size
                if self.train_curr_index == len(self.test_q_arr):
                    next_batch_avail = False
            else:
                self.q_batch = random.sample(self.test_q_arr, self.batch_size)
                self.q_id_batch = random.sample(self.test_q_id_arr, self.batch_size)
                self.im_id_batch = random.sample(self.test_im_id_arr, self.batch_size)
                self.ans_batch = random.sample(self.test_ans_arr, self.batch_size)
        return next_batch_avail

    def batch_word2vec(self, discard=True):
        """
        if discard is True, throw away questions in which all words are not in top_k_q_dict (51474 out of 60k questions have all words within top 1k)
        """
        inp_inds = []
        out_inds = []
        if discard:
            for q in self.q_batch:
                words = q.split(" ")
                curr_inds = []
                all_found = True
                for word in words:
                    if word in self.top_k_q_dict:
                        curr_inds.append(self.top_k_q_dict[word])
                    else:
                        all_found = False
                        break
                if all_found:
                    inp_inds.extend([[ind] for ind in curr_inds])
                    out_inds.extend([[ind] for ind in (curr_inds[1:] + [self.top_k_q_dict["__END__"]])])
        else:
            # to be implemented
            pass
        return inp_inds, out_inds

    def batch_fcnn(self, discard=True):
        """
        if discard is True, throw away questions in which all words are not in top_k_q_dict (51474 out of 60k questions have all words within top 1k)
        """
        inp_inds = []
        im_embeds = []
        ans_inds = []

        max_len = max([len(q) for q in self.q_batch])

        if discard:
            for ind in range(len(self.q_batch)):
                q = self.q_batch[ind]
                curr_inds = []
                if embed_type == "RNN":
                    words = q.split(" ")
                    all_found = True
                    for word in words:
                        if word in self.top_k_q_dict:
                            curr_inds.append(self.top_k_q_dict[word])
                        else:
                            all_found = False
                            break
                elif embed_type == "GloVe":
                    curr_inds.append(embed_question([q], self.embed_index, 300))
            
                im_id = self.im_id_batch[ind]
                img_path = img_id_to_path(im_id)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                with self.graph.as_default():
                    set_session(self.sess)
                    self.vision_model.predict(x)
                    fc2_features_extractor_model = Model(inputs=self.vision_model.input, outputs=self.vision_model.get_layer('fc2').output)
                    
                    fc2_features = fc2_features_extractor_model.predict(x)
                    fc2_features = fc2_features.reshape((4096,))
                K.clear_session()

                ans = self.ans_batch[ind]
                c = Counter(ans)
                most_common_ans = c.most_common(1)[0][0]
                found_ans = True
                if most_common_ans not in self.top_k_ans_dict:
                    found_ans = False

                if all_found and found_ans:
                    inp_inds.append(np.array(curr_inds + [-1]*(max_len-len(curr_inds))))
                    im_embeds.append(np.array(fc2_features))
                    ans_inds.append(self.top_k_ans_dict[most_common_ans])
        else:
            # to be implemented
            pass

        return inp_inds, im_embeds, ans_inds


    def get_preds(self, model_class=KNNModel, k=4):
        model = model_class(k)
        model.train(self.train_q_arr, self.train_q_id_arr, self.train_im_id_arr, self.train_ans_arr)
        
        self.preds = model.predict(self.test_q_arr, self.test_q_id_arr, self.test_im_id_arr)

    def get_accuracy(self):
        acc = 0
        for i in range(len(self.preds)):
            count = self.test_ans_arr[i].count(self.preds[i][0])
            print(self.preds[i], self.test_ans_arr[i])
            if self.metric == "min_k":
                acc += min(count/3, 1)    
        acc /= len(self.preds)
        return acc

data_arr = get_by_ques_type([])
vocab_size = 1000
embed_size = 300
output_size = 1000
pointwise_layer_size = 1024
rnn_input_size = 1000
cnn_input_size = 4096

p = Pipeline(data_arr)
w2v = Word2Vec(vocab_size + 1, embed_size)
p.create_split()

# sess = tf.Session()
# tf.global_variables_initializer().run(session=sess)

train_step = 0
curr_samples = 0

train_losses = []
test_losses = []
# while p.next_batch(train=True, replace=False):
#     train_inp, train_out = p.batch_word2vec()

#     batch_samples = len(train_inp)
#     curr_samples += batch_samples
#     train_step += 1
    
#     train_loss = w2v.train_step(np.array(train_inp), np.array(train_out), sess)

#     p.next_batch(train=False, replace=True)
#     test_inp, test_out = p.batch_word2vec()
#     test_samples = len(test_inp)

#     test_loss = w2v.evaluate(np.array(test_inp), np.array(test_out), sess)

#     train_losses.append(train_loss)
#     test_losses.append(test_loss)
    
#     print("TRAIN STEP: %d | SAMPLES IN TRAIN BATCH: %d | TRAIN SAMPLES SO FAR: %d | TRAIN LOSS: %f | TEST LOSS: %f" %(train_step, batch_samples, curr_samples, train_loss, test_loss))

# train_steps = list(range(train_step))
# plt.plot(train_steps, train_losses)
# plt.plot(train_steps, test_losses)
# plt.show()

saver = tf.train.Saver()

while p.next_batch(train=True, replace=False):
    train_qs, train_ims, train_ans = p.batch_fcnn()
    fcnn = FCNN(cnn_input_size, rnn_input_size, pointwise_layer_size, output_size, vocab_size)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    if len(train_qs) > 0:
        train_loss, output, cnn, rnn = fcnn._train_step(sess, np.array(train_ims), np.array(train_qs), np.array(train_ans))
        print("******************************************************************")
        print("************************* TRAIN LOSS *************************")
        print(train_loss)
        # print("************************* CNN LOSS *************************")
        # print(cnn)
        # print("************************* RNN LOSS *************************")
        # print(rnn)
        # print("************************* OUTPUT LOSS *************************")
        # print(output)
        print("******************************************************************")
        train_losses.append(train_loss)
        train_step += 1
        if train_step % 400 == 0:
            np.savez("losses%d.npz"%train_step, np.array(train_losses))
            # save_path = saver.save(sess, "model%s.ckpt"%train_step)
            tf.train.write_graph(sess.graph_def, '', 'train%s.pbtxt'%train_step)

