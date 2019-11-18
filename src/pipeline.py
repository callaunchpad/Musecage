from __future__ import print_function

import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
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

#comment what different lists mean in create_split and next_batch

class Pipeline():
    def __init__(self, data_arr, metric="min_k", embed_type="RNN", batch_size=64):
        self.data_arr = data_arr
        self.metric = metric
        self.batch_size = batch_size
        self.embed_type = embed_type

        if self.embed_type == "GloVe":
            self.embed_index = load_glove()
        elif self.embed_type == "Word2Vec":
            self.embed_mat = np.load("saved_models/word2vec_model/embed_mat_2000.npz")["arr_0"]

        self.train_curr_index = 0
        self.test_curr_index = 0
        
    def create_split(self, split_val=.8, custom_split=False, custom_train=[], custom_test=[], build_top_q_vocab=True, top_q_k=1000, build_top_ans_vocab=True, top_ans_k=1000):
        if custom_split:
            self.train_data = custom_train
            self.test_data = custom_test
        else:
            self.train_data = self.data_arr[:int(len(self.data_arr) * split_val)]
            self.test_data = self.data_arr[int(len(self.data_arr) * split_val):]

        self.train_q_arr = [format_q_for_embed(val["question"]) for val in self.train_data]
        self.test_q_arr = [format_q_for_embed(val["question"]) for val in self.test_data]

        if build_top_q_vocab:
            allwords = []
            for q in self.train_q_arr + self.test_q_arr:
                allwords.extend(q.split(" "))
            c = Counter(allwords)
            top_k_words = c.most_common(top_q_k)
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

        self.train_ans_type_arr = [val["answer_type"] for val in self.train_data]
        self.test_ans_type_arr = [val["answer_type"] for val in self.test_data]

        if build_top_ans_vocab:
            allwords = []
            for q in list(np.array(self.train_ans_arr).flatten()) + list(np.array(self.test_ans_arr).flatten()):
                allwords.extend(q.split(" "))
            c = Counter(allwords)
            top_k_words = c.most_common(top_ans_k)
            self.top_k_ans_dict = {}
            self.top_k_ans_dict_reverse = {}
            for ind in range(len(top_k_words)):
                self.top_k_ans_dict[top_k_words[ind][0]] = ind
                self.top_k_ans_dict_reverse[ind] = top_k_words[ind][0]

    def next_batch(self, train=True, replace=False):
        next_batch_avail = True
        if train:
            if not replace:
                self.q_batch = self.train_q_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.q_id_batch = self.train_q_id_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.im_id_batch = self.train_im_id_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.ans_batch = self.train_ans_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.ans_type_batch = self.train_ans_type_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.train_curr_index += self.batch_size
                if self.train_curr_index >= len(self.train_q_arr):
                    next_batch_avail = False
            else:
                #someone abstract this further? or not
                ind_arr = range(len(self.train_q_arr))
                ind_batch = random.sample(ind_arr, self.batch_size)
                self.q_batch = [self.train_q_arr[ind] for ind in ind_batch]
                self.q_id_batch = [self.train_q_id_arr[ind] for ind in ind_batch]
                self.im_id_batch = [self.train_im_id_arr[ind] for ind in ind_batch]
                self.ans_batch = [self.train_ans_arr[ind] for ind in ind_batch]
                self.ans_type_batch = [self.train_ans_type_arr[ind] for ind in ind_batch]
        else:
            if not replace:
                self.q_batch = self.test_q_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.q_id_batch = self.test_q_id_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.im_id_batch = self.test_im_id_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.ans_batch = self.test_ans_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
                self.ans_type_batch = self.test_ans_type_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.test_curr_index += self.batch_size
                if self.test_curr_index >= len(self.test_q_arr):
                    next_batch_avail = False
            else:
                ind_arr = range(len(self.test_q_arr))
                ind_batch = random.sample(ind_arr, self.batch_size)
                self.q_batch = [self.test_q_arr[ind] for ind in ind_batch]
                self.q_id_batch = [self.test_q_id_arr[ind] for ind in ind_batch]
                self.im_id_batch = [self.test_im_id_arr[ind] for ind in ind_batch]
                self.ans_batch = [self.test_ans_arr[ind] for ind in ind_batch]
                self.ans_type_batch = [self.test_ans_type_arr[ind] for ind in ind_batch]
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
        ans_types = []
        all_ans = []

        max_len = max([len(q) for q in self.q_batch])

        if discard:
            for ind in range(len(self.q_batch)):
                q = self.q_batch[ind]
                curr_inds = []
                if self.embed_type == "RNN":
                    words = q.split(" ")
                    all_found = True
                    for word in words:
                        if word in self.top_k_q_dict:
                            curr_inds.append(self.top_k_q_dict[word])
                        else:
                            all_found = False
                            break
                elif self.embed_type == "GloVe":
                    curr_inds = embed_question([q], self.embed_index, 300)[0]
                    all_found = True
                elif self.embed_type == "Word2Vec":
                    words = q.split(" ")
                    all_found = True
                    curr_inds = np.zeros((300))
                    for word in words:
                        if word in self.top_k_q_dict:
                            word_embed = self.embed_mat[self.top_k_q_dict[word]]
                            curr_inds = np.add(curr_inds, word_embed)
                        else:
                            all_found = False
                            break
            
                found_im = True
                try:
                    fc2_features = np.load("../data/vqa/im_embed_data/"+str(self.im_id_batch[ind])+".npz")["arr_0"]
                except:
                    found_im = False    

                ans = self.ans_batch[ind]
                c = Counter(ans)
                most_common_ans = c.most_common(1)[0][0]
                found_ans = True
                if most_common_ans not in self.top_k_ans_dict:
                    found_ans = False

                if all_found and found_ans and found_im:
                    if self.embed_type == "RNN":
                        inp_inds.append(np.array(curr_inds + [-1]*(max_len-len(curr_inds))))
                    elif self.embed_type == "GloVe":
                        inp_inds.append(curr_inds)
                    elif self.embed_type == "Word2Vec":
                        inp_inds.append(curr_inds)
                    im_embeds.append(np.array(fc2_features))
                    ans_inds.append(self.top_k_ans_dict[most_common_ans])
                    ans_types.append(self.ans_type_batch[ind])
                    all_ans.append(ans)
        else:
            # to be implemented
            pass

        return inp_inds, im_embeds, ans_inds, ans_types, all_ans


    def get_accuracy_dict(self, model, sess=None, k=4):
        if type(model) is KNNModel:
            model.train(self.train_q_arr, self.train_q_id_arr, self.train_im_id_arr, self.train_ans_arr)
            self.preds = model.predict(self.test_q_arr, self.test_q_id_arr, self.test_im_id_arr)
        elif type(model) is FCNN:
            if not sess:
                print("No session inputed")
                return None
            else:
                ans_type_dict = {"yes/no": [0, 0], "number": [0, 0], "other": [0, 0]}
                test_step = 0
                while self.next_batch(train=False):
                    print("TEST STEP: %d"%test_step)
                    inp_inds, im_embeds, ans_inds, ans_types, all_ans = self.batch_fcnn()
                    for i in range(len(inp_inds)):
                        pred_output = sess.run(model.output, feed_dict={model.cnn_in: [im_embeds[i]], model.q_batch: [inp_inds[i]]})
                        ans_type_dict[ans_types[i]][1] += 1
                        pred_value = np.argmax(pred_output)
                        c = Counter(all_ans[i])
                        if c[self.top_k_ans_dict_reverse[pred_value]] >= 3:
                            ans_type_dict[ans_types[i]][0] += 1
                    test_step += 1
                return ans_type_dict

    def get_accuracy(self, ans_type_dict):
        print("yes/no accuracy: ", ans_type_dict["yes/no"][0]/ans_type_dict["yes/no"][1])
        print("number accuracy: ", ans_type_dict["number"][0]/ans_type_dict["number"][1])
        print("other accuracy: ", ans_type_dict["other"][0]/ans_type_dict["other"][1])
        total_correct = ans_type_dict["yes/no"][0] + ans_type_dict["number"][0] + ans_type_dict["other"][0]
        total = ans_type_dict["yes/no"][1] + ans_type_dict["number"][1] + ans_type_dict["other"][1]
        print("total accuracy: ", total_correct/total)


    # def get_accuracy(self):
    #     acc = 0
    #     for i in range(len(self.preds)):
    #         count = self.test_ans_arr[i].count(self.preds[i][0])
    #         print(self.preds[i], self.test_ans_arr[i])
    #         if self.metric == "min_k":
    #             acc += min(count/3, 1)    
    #     acc /= len(self.preds)
    #     return acc

def train_FCNN(data_len=30000, vocab_size = 1000, embed_size = 300, output_size = 1000, pointwise_layer_size = 1024,
        rnn_input_size = 1000, cnn_input_size = 4096, embed_type = "RNN", savedir = "model_/", verbose = True, save = True):

    data_arr = get_by_ques_type([])[:data_len]
    p = Pipeline(data_arr, embed_type=embed_type)
    p.create_split()

    train_step = 0
    curr_samples = 0

#get accuracy
# fcnn = FCNN(cnn_input_size, rnn_input_size, pointwise_layer_size, output_size, vocab_size, embed_type=embed_type, lr=1e-4)
# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     saver.restore(sess, "../saved_models/saved_RNN/RNN_749-749")
#     p.get_accuracy(p.get_accuracy_dict(fcnn, sess))

# train_step = 0
# curr_samples = 0

# train_losses = []
# test_losses = []

# for k, v in p.top_k_ans_dict.items():
#     if v == 0:
#         print(k)

# fcnn = FCNN(cnn_input_size, rnn_input_size, pointwise_layer_size, output_size, vocab_size, embed_type=embed_type, lr=1e-4)
# # w2v = Word2Vec(vocab_size + 1, embed_size)

# sess = tf.Session()
# tf.global_variables_initializer().run(session=sess)

# while p.next_batch(train=True, replace=False):
#     start_time = time.time()
#     train_qs, train_ims, train_ans = p.batch_fcnn()
#     if len(train_qs) > 0:
#         train_loss = fcnn.train_step(sess, np.array(train_ims), np.array(train_qs), np.array(train_ans))
#         print("TRAIN LOSS: %f "%(train_loss))
#         train_losses.append(train_loss)
        
#     p.next_batch(train=False, replace=True)
#     test_qs, test_ims, test_ans = p.batch_fcnn()
#     if len(test_qs) > 0:
#         test_loss = fcnn.evaluate(sess, np.array(test_ims), np.array(test_qs), np.array(test_ans))
#         print("TEST LOSS: %f "%(test_loss))
#         test_losses.append(test_loss)
 
#     # if train_step % 100 == 0:
#         # tf.train.Saver().save(sess, "%s_model_2/%s_%d"%(embed_type, embed_type, train_step), global_step=train_step)
#         # np.savez("%s_model_2/train_losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(train_losses))
#         # np.savez("%s_model_2/test_losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(test_losses))
#     train_step += 1

#     end_time = time.time()
#     print("time elapsed: ", end_time - start_time, " seconds")
# tf.train.Saver().save(sess, "%s_model_2/%s_%d"%(embed_type, embed_type, train_step), global_step=train_step)
# np.savez("%s_model_2/losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(train_losses))
# np.savez("%s_model_2/test_losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(test_losses))

# output = fcnn.get_output(sess, im, [curr_inds], ans)
# np.savez("test_out.npz", output)

# run = True
# while run:
#     p.next_batch(train=True, replace=True)
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

#     if train_step % 100 == 0:
#         tf.train.Saver().save(sess, "word2vec_model/word2vec_%d"%(train_step), global_step=train_step)
#         np.savez("word2vec_model/word2vec_train_losses_%d"%(train_step), np.array(train_losses))
#         np.savez("word2vec_model/word2vec_test_losses_%d"%(train_step), np.array(test_losses))
#     if train_step == 2000:
#         run = False
#     print("TRAIN STEP: %d | SAMPLES IN TRAIN BATCH: %d | TRAIN SAMPLES SO FAR: %d | TRAIN LOSS: %f | TEST LOSS: %f" %(train_step, batch_samples, curr_samples, train_loss, test_loss))

# tf.train.Saver().save(sess, "word2vec_model/word2vec_%d"%(train_step), global_step=train_step)
# np.savez("word2vec_model/word2vec_train_losses_%d"%(train_step), np.array(train_losses))
# np.savez("word2vec_model/word2vec_test_losses_%d"%(train_step), np.array(test_losses))

# train_steps = list(range(train_step))
# plt.plot(train_steps, train_losses)
# plt.plot(train_steps, test_losses)
# plt.show()

# p.next_batch(train=False)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run(session=sess)
#     saver = tf.train.Saver()
#     saver.restore(sess, "saved_RNN")
#     output_model = tf.get_trainable_variables(adf)

#     p.get_accuracy(p.get_accuracy_dict(saved_model, sess))

    train_losses = []
    test_losses = []

    fcnn = FCNN(cnn_input_size, rnn_input_size, pointwise_layer_size, output_size, vocab_size, embed_type=embed_type, lr=1e-4)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    while p.next_batch(train=True, replace=False):
        start_time = time.time()
        train_qs, train_ims, train_ans = p.batch_fcnn()
        if len(train_qs) > 0:
            train_loss = fcnn.train_step(sess, np.array(train_ims), np.array(train_qs), np.array(train_ans))
            if verbose:
                print("TRAIN LOSS: %f "%(train_loss))
            train_losses.append(train_loss)
            
        p.next_batch(train=False, replace=True)
        test_qs, test_ims, test_ans = p.batch_fcnn()
        if len(test_qs) > 0:
            test_loss = fcnn.evaluate(sess, np.array(test_ims), np.array(test_qs), np.array(test_ans))
            if verbose:
                print("TEST LOSS: %f "%(test_loss))
            test_losses.append(test_loss)
    
        if save:
            if train_step % 100 == 0:
                tf.train.Saver().save(sess, savedir+"%s_%d"%(embed_type, embed_type, train_step), global_step=train_step)
                np.savez(savedir+"train_losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(train_losses))
                np.savez(savedir+"test_losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(test_losses))
        train_step += 1

        end_time = time.time()
        if verbose:
            print("Time elapsed: ", end_time - start_time, " seconds")
    if save:
        tf.train.Saver().save(sess, savedir+"%s_%d"%(embed_type, embed_type, train_step), global_step=train_step)
        np.savez(savedir+"losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(train_losses))
        np.savez(savedir+"test_losses_%s_%d.npz"%(embed_type, embed_type, train_step), np.array(test_losses))

def predict():
    def get_im_embedding(img_path):
        """
        Args:
            - img_path: path to image
            
        Return:
            - (4096,) vector embedding of image
        """     
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        vision_model = VGG16(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000)
        features = vision_model.predict(x)
        fc2_features_extractor_model = Model(inputs=vision_model.input, outputs=vision_model.get_layer('fc2').output)
        
        fc2_features = fc2_features_extractor_model.predict(x)
        fc2_features = fc2_features.reshape((4096,))
        
        return fc2_features

    test = get_im_embedding("test.png")
    np.savez("test.npz", test)

    im = [np.load("test.npz")["arr_0"]]
    q = "what is the girl Alicia drinking"
    curr_inds = []
    words = q.split(" ")
    for word in words:
        if word in p.top_k_q_dict:
            curr_inds.append(p.top_k_q_dict[word])
    print(curr_inds)
    ans = [0]

    output = fcnn.get_output(sess, im, [curr_inds], ans)
    np.savez("test_out.npz", output)

def train_word2vec(data_len=30000, vocab_size = 1000, embed_size = 300, verbose = True, save = True):
    data_arr = get_by_ques_type([])[:data_len]
    p = Pipeline(data_arr, embed_type=embed_type)
    p.create_split()

    train_step = 0
    curr_samples = 0

    train_losses = []
    test_losses = []

    w2v = Word2Vec(vocab_size + 1, embed_size)
    run = True
    while run:
        p.next_batch(train=True, replace=True)
        train_inp, train_out = p.batch_word2vec()

        batch_samples = len(train_inp)
        curr_samples += batch_samples
        train_step += 1
        
        train_loss = w2v.train_step(np.array(train_inp), np.array(train_out), sess)

        p.next_batch(train=False, replace=True)
        test_inp, test_out = p.batch_word2vec()
        test_samples = len(test_inp)

        test_loss = w2v.evaluate(np.array(test_inp), np.array(test_out), sess)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if train_step % 100 == 0 and save:
            tf.train.Saver().save(sess, "saved_models/word2vec_model/word2vec_%d"%(train_step), global_step=train_step)
            np.savez("saved_models/word2vec_model/word2vec_train_losses_%d"%(train_step), np.array(train_losses))
            np.savez("saved_models/word2vec_model/word2vec_test_losses_%d"%(train_step), np.array(test_losses))
        if train_step == 2000:
            run = False
        if verbose:
            print("TRAIN STEP: %d | SAMPLES IN TRAIN BATCH: %d | TRAIN SAMPLES SO FAR: %d | TRAIN LOSS: %f | TEST LOSS: %f" %(train_step, batch_samples, curr_samples, train_loss, test_loss))

    if save: 
        tf.train.Saver().save(sess, "saved_models/word2vec_model/word2vec_%d"%(train_step), global_step=train_step)
        np.savez("saved_models/word2vec_model/word2vec_train_losses_%d"%(train_step), np.array(train_losses))
        np.savez("saved_models/word2vec_model/word2vec_test_losses_%d"%(train_step), np.array(test_losses))

def plot(train_steps, train_losses, test_losses):
    train_steps = list(range(train_step))
    plt.plot(train_steps, train_losses)
    plt.plot(train_steps, test_losses)
    plt.show()



















