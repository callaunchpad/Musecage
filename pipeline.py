from __future__ import print_function

import os
import sys
import numpy as np

from load_create_data import *

class Pipeline():
    def __init__(self, data_arr, metric='min_k':
        self.data_arr = data_arr
        self.metric = metric
        
    def create_split(self, split_val=.8):
        self.train_data = self.data_arr[:len(self.data_arr) * split_val]
        self.test_data = self.data_arr[len(self.data_arr) * split_val:]

        self.train_q_arr = [format_q_for_embed(val['question']) for val in self.train_data]
        self.test_q_arr = [format_q_for_embed(val['question']) for val in self.test_data]

        self.train_q_id_arr = [val['question_id'] for val in self.train_data]
        self.test_q_id_arr = [val['question_id'] for val in self.test_data]

        self.train_im_id_arr = [val['image_id'] for val in self.train_data]
        self.test_im_id_arr = [val['image_id'] for val in self.test_data]
        
        self.train_ans_arr = [val['answers'] for val in self.train_data]
        self.test_ans_arr = [val['answers'] for val in self.test_data]

    def get_preds(self, model_class=KNNModel, k=5):
        model = model_class(k)
        model.train(self.train_q_arr, self.train_q_id_arr, self.train_im_id_arr, self.train_ans_arr)
        
        self.preds = model.predict(self.test_q_arr, self.test_q_id_arr, self.test_im_id_arr)

    def get_accuracy(self):
        acc = 0
        for i in len(self.preds):
            count = self.test_ans_arr[i].count(self.preds[i])
            if self.metric == "min_k":
                acc += min(count/3, 1)
        
        acc /= len(self.preds)

        return acc

        
                
        

            








