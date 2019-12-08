from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from load_create_data import *

class KNNModel():

    #fix knn code to work with pipeline there are broken things in here for running this independently

    def __init__(self, k, q_embed_type="glove", glove_embed_dim=300, discard=False, output_n=1, verbose=True, pred_verb_num=5):
        """
        Creates KNNModel based on:
            - k: number of neighbors for model
            - q_embed_type: embedding to be used for questions
            - glove_embed_dim: dimension of glove embedding
            - discard: whether or not to discard questions in which words are not found (may need fixing)
            - output_n: number of top n predictionds to output
        """
        self.k = k
        self.q_embed_type = q_embed_type
        self.discard = discard
        self.glove_embed_dim = glove_embed_dim
        self.output_n = output_n
        self.verbose = verbose
        self.pred_verb_num = pred_verb_num

        self.vision_model = VGG16(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000)
        self.knn_qs = KNeighborsClassifier(n_neighbors=self.k)

    def train(self, q_arr, q_ids, im_ids, ans):
        """
        Trains KNN classifier based on question embeddings and question ids. (Stored image ids and answers for later use in prediction pipeline)
        """
        if self.q_embed_type == "glove":
            self.embed_index = load_glove()
            if self.verbose:
                print("Started embedding training data...")
            self.q_embed = embed_question(q_arr, self.embed_index, self.glove_embed_dim, self.discard)
            if self.verbose:
                print("Finished embedding training data.")

        self.q_ids = q_ids
        if self.verbose:
            print("Started fitting KNN...")
        self.knn_qs.fit(self.q_embed, self.q_ids)
        if self.verbose:
            print("Finished fitting KNN.")
        
        self.im_ids = im_ids
        self.ans = ans
        self.q_id_to_im_id = {self.q_ids[i]:self.im_ids[i] for i in range(len(self.q_ids))}
        self.im_id_to_ans = {self.im_ids[i]:self.ans[i] for i in range(len(self.im_ids))}

    def predict(self, test_q_arr, test_q_ids, test_im_ids):
        """
        Returns list of answers corresponding to full prediction pipeline output on list of test image question/question ids.
        """
        if self.q_embed_type == "glove":
            try:
                test_q_embed = embed_question(test_q_arr, self.embed_index, self.glove_embed_dim, self.discard)
            except:
                print("Not trained/no embeddings index matrix.")
        
        if self.verbose:
            print("Starting predictions...")

        preds = []
        for ind in range(len(test_q_embed)):
            closest_q_inds = self.knn_qs.kneighbors(test_q_embed)
            knearest_im_ids = [self.q_id_to_im_id[self.q_ids[q]] for q in closest_q_inds[1][0]]
            closest_im_id = self.get_closest_image(test_im_ids[ind], knearest_im_ids)
            counts = Counter(self.im_id_to_ans[closest_im_id])
            pred = [p[0] for p in counts.most_common(self.output_n)]
            preds.append(pred)
            if self.verbose and ind % self.pred_verb_num == 0:
                print("At prediction iteration: %d"%ind)

        if self.verbose:
            print("Finished predictions.")

        return preds

    def get_cos_sims(self, target_img_loc, img_loc_list):
        """
        Args:
            - target_img_loc: location of input image 
            - img_loc_list: list of locations of images to be compared with target img
            
        Returns:
            - ndarray containing cosine similarity between target image and each image in img_loc_list
        """
        X = get_embedding(target_img_loc).reshape((1, 4096), self.vision_model)
        Y = np.array([get_embedding(img, self.vision_model) for img in img_loc_list])
        
        return cosine_similarity(X, Y, dense_output=True)


    def get_closest_image(self, target_img_id, img_id_list):
        """
        Args:
            - target_img_loc: location of input image 
            - img_loc_list: list of locations of images to be compared with target img
            
        Returns:
            - location of image closest to target image
        """
        target_img_loc = img_id_to_path(target_img_id)
        img_loc_list = [img_id_to_path(img_id) for img_id in img_id_list]
        cos_sims = list(self.get_cos_sims(target_img_loc, img_loc_list)[0])
        i = cos_sims.index(max(cos_sims))
        
        return img_id_list[i]



