from __future__ import print_function

import os
import sys
import numpy as np

from load_create_data import *
from knn import KNNModel
from collections import Counter

class Pipeline():
    def __init__(self, data_arr, metric="min_k", batch_size=64, replace=False):
        self.data_arr = data_arr
        self.metric = metric
        self.batch_size = batch_size
        self.replace = replace
        
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

        self.train_q_id_arr = [val["question_id"] for val in self.train_data]
        self.test_q_id_arr = [val["question_id"] for val in self.test_data]

        self.train_im_id_arr = [val["image_id"] for val in self.train_data]
        self.test_im_id_arr = [val["image_id"] for val in self.test_data]
        
        self.train_ans_arr = [val["answers"] for val in self.train_data]
        self.test_ans_arr = [val["answers"] for val in self.test_data]

    def next_train_batch(self):
        train_q_batch = np.random.choice(self.train_q_arr, self.batch_size, self.replace)
        train_q_id_batch = np.random.choice(self.train_q_id_arr, self.batch_size, self.replace)
        train_im_id_batch = np.random.choice(self.train_im_id_arr, self.batch_size, self.replace)
        train_ans_batch = np.random.choice(self.train_ans_arr, self.batch_size, self.replace)
        return train_q_batch, train_q_id_batch, train_im_id_batch, train_ans_batch

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


