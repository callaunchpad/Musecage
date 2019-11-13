from __future__ import print_function

import os
import sys
import numpy as np

from load_create_data import *
from knn import KNNModel
from word2vec import Word2Vec
from collections import Counter

class Pipeline():
    def __init__(self, data_arr, metric="min_k", batch_size=64, replace=False):
        self.data_arr = data_arr
        self.metric = metric
        self.batch_size = batch_size
        self.replace = replace
        self.curr_index = 0
        
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
            self.top_k_dict = {}
            for ind in range(len(top_k_words)):
                self.top_k_dict[top_k_words[ind][0]] = ind
            self.top_k_dict["__END__"] = len(top_k_words)

        self.train_q_id_arr = [val["question_id"] for val in self.train_data]
        self.test_q_id_arr = [val["question_id"] for val in self.test_data]

        self.train_im_id_arr = [val["image_id"] for val in self.train_data]
        self.test_im_id_arr = [val["image_id"] for val in self.test_data]
        
        self.train_ans_arr = [val["answers"] for val in self.train_data]
        self.test_ans_arr = [val["answers"] for val in self.test_data]

    def next_train_batch(self):
        if not self.replace:
            self.train_q_batch = self.train_q_arr[self.curr_index : self.curr_index + self.batch_size]
            self.train_q_id_batch = self.train_q_id_arr[self.curr_index : self.curr_index + self.batch_size]
            self.train_im_id_batch = self.train_im_id_arr[self.curr_index : self.curr_index + self.batch_size]
            self.train_ans_batch = self.train_ans_arr[self.curr_index : self.curr_index + self.batch_size]
            self.curr_index += self.batch_size
        else:
            self.train_q_batch = np.random.choice(self.train_q_arr, self.batch_size, self.replace)
            self.train_q_id_batch = np.random.choice(self.train_q_id_arr, self.batch_size, self.replace)
            self.train_im_id_batch = np.random.choice(self.train_im_id_arr, self.batch_size, self.replace)
            self.train_ans_batch = np.random.choice(self.train_ans_arr, self.batch_size, self.replace)
        return self.train_q_batch, self.train_q_id_batch, self.train_im_id_batch, self.train_ans_batch

    def batch_word2vec(self, discard=True):
        """
        if discard is True, throw away questions in which all words are not in top_k_dict (51474 out of 60k questions have all words within top 1k)
        """
        inp_inds = []
        out_inds = []
        for q in self.train_q_batch:
            words = q.split(" ")
            curr_inds = []
            for word in words:
                if word in self.top_k_dict:
                    curr_inds.append(self.top_k_dict[word])
                else:
                    break
            inp_inds.extend(curr_inds)
            out_inds.extend((curr_inds[1:] + [self.top_k_dict["__END__"]]))
        return inp_inds, out_inds

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
p = Pipeline(data_arr)
p.create_split()

# w2v = Word2Vec()


