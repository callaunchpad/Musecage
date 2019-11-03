from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from load_create_data import *

class KNNModel():

    def __init__(self, k, q_embed_type="glove", glove_embed_dim=300, discard=False, output_n=1):
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

        self.vision_model = VGG16(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000)
        self.knn_qs = KNeighborsClassifier(n_neighbors=self.k)

    def train(self, q_arr, q_ids, im_ids, ans):
        """
        Trains KNN classifier based on question embeddings and question ids. (Stored image ids and answers for later use in prediction pipeline)
        """
        if self.q_embed_type == "glove":
            self.embed_index = load_glove()
            self.q_embed = embed_question(q_arr, self.embed_index, self.glove_embed_dim, self.discard)
        
        self.q_ids = q_ids
        self.knn_qs.fit(self.q_embed, self.q_ids)
        
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
        
        preds = []
        for ind in range(len(test_q_embed)):
            q_embed = test_q_embed[ind]
            closest_q_inds = self.knn_qs.kneighbors(test_q_embed)
            knearest_im_ids = [self.q_id_to_im_id[self.q_ids[q]] for q in closest_q_inds[1][0]]
            closest_im_id = self.get_closest_image(test_im_ids[ind], knearest_im_ids)
            counts = Counter(self.im_id_to_ans[closest_im_id])
            pred = [p[0] for p in counts.most_common(self.output_n)]
            preds.append(pred)
        return preds

    def get_embedding(self, img_path):
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
        
        features = self.vision_model.predict(x)
        fc2_features_extractor_model = Model(inputs=self.vision_model.input, outputs=self.vision_model.get_layer('fc2').output)
        
        fc2_features = fc2_features_extractor_model.predict(x)
        fc2_features = fc2_features.reshape((4096,))
        
        return fc2_features

    def get_cos_sims(self, target_img_loc, img_loc_list):
        """
        Args:
            - target_img_loc: location of input image 
            - img_loc_list: list of locations of images to be compared with target img
            
        Returns:
            - ndarray containing cosine similarity between target image and each image in img_loc_list
        """
        X = self.get_embedding(target_img_loc).reshape((1, 4096))
        Y = np.array([self.get_embedding(img) for img in img_loc_list])
        
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

data_arr = get_by_ques_type(["how many"])



