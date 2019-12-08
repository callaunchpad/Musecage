from VQA.PythonHelperTools.vqaTools.vqa import VQA
import random
import os
import re
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

data_dir = "../data"
vqa_data_dir = "vqa"

train_image_dir_sub = "abstract_images"
train_image_emb_dir_sub = "im_embed_data"
train_image_base = "abstract_v002_train2015_"
train_image_dir = "%s/%s/%s/"%(data_dir, vqa_data_dir, train_image_dir_sub)
train_image_emb_dir = "%s/%s/%s/"%(data_dir, vqa_data_dir, train_image_emb_dir_sub)

train_ann_file = "%s/%s/Annotations/abstract_v002_train2015_annotations.json"%(data_dir, vqa_data_dir)
train_q_file = "%s/%s/Questions/OpenEnded_abstract_v002_train2015_questions.json"%(data_dir, vqa_data_dir)

val_image_dir_sub = "abstract_images_val"
val_image_emb_dir_sub = "im_embed_data_val"
val_image_base = "abstract_v002_val2015_"
val_image_dir = "%s/%s/%s/"%(data_dir, vqa_data_dir, val_image_dir_sub)
val_image_emb_dir = "%s/%s/%s/"%(data_dir, vqa_data_dir, val_image_emb_dir_sub)

val_ann_file = "%s/%s/Annotations/abstract_v002_val2015_annotations.json"%(data_dir, vqa_data_dir)
val_q_file = "%s/%s/Questions/OpenEnded_abstract_v002_val2015_questions.json"%(data_dir, vqa_data_dir)

glove_dir = "glove.6B"
glove_dim = glove_dir + ".%dd.txt"%(300)
glove_file = "%s/glove/%s/%s"%(data_dir, glove_dir, glove_dim)

vqa = VQA(train_ann_file, train_q_file)
vqa_val = VQA(val_ann_file, val_q_file)

def get_by_ques_type(quesTypes, train=True):
    if train:
        ans_ids = vqa.getQuesIds(quesTypes=quesTypes)
        answers = vqa.loadQA(ans_ids)
    else:
        ans_ids = vqa_val.getQuesIds(quesTypes=quesTypes)
        answers = vqa_val.loadQA(ans_ids)
    finals = []
    for an in answers:
        if train:
            qa = vqa.qqa[an["question_id"]]
        else:
            qa = vqa_val.qqa[an["question_id"]]
        final = qa.copy()
        final["answers"] = [a["answer"] for a in an["answers"]]
        final["answer_type"] = an["answer_type"]
        finals.append(final)
    return finals

def get_by_ans_type(ansTypes, train=True):
    if train:
        ans_ids = vqa.getQuesIds(ansTypes=ansTypes)
        answers = vqa.loadQA(ans_ids)
    else:
        ans_ids = vqa.getQuesIds(ansTypes=ansTypes)
        answers = vqa.loadQA(ans_ids)
    finals = []
    for an in answers:
        if train:
            qa = vqa.qqa[an["question_id"]]
        else:
            qa = vqa_val.qqa[an["question_id"]]
        final = qa.copy()
        final["answers"] = [a["answer"] for a in an["answers"]]
        finals.append(final)
    return finals

def get_by_img_ids(img_ids, train=True):
    if train:
        ans_ids = vqa.getQuesIds(imgIds=img_ids)
        answers = vqa.loadQA(ans_ids)
    else:
        ans_ids = vqa.getQuesIds(imgIds=img_ids)
        answers = vqa.loadQA(ans_ids)
    finals = []
    for an in answers:
        if train:
            qa = vqa.qqa[an["question_id"]]
        else:
            qa = vqa_val.qqa[an["question_id"]]
        final = qa.copy()
        final["answers"] = [a["answer"] for a in an["answers"]]
        finals.append(final)
    return finals

def load_glove():
    print("Indexing word vectors.")
    embed_index = {}
    with open(glove_file, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embed_index[word] = coefs
    print("Found %s word vectors." % len(embed_index))
    return embed_index

def embed_question(q_arr, embed_index, dim, discard=False):
    """
    A function that takes in an array of different questions and returns an array of question embed. 
    If a word isn't found within Glove, that word is simply taken out of the question embedding. 
    """
    q_embeds=[]
    for i in range(len(q_arr)): #For each question
        q_embed=np.zeros((dim))
        q = q_arr[i]
        if discard:
            words_exist = True
        for word in q.split(" "): #For each word in each question
            try: #If the word embedding is found
                word_embedding = embed_index[word]
                q_embed = np.add(q_embed, word_embedding)
            except:
                if discard:
                    words_exist = False
                    break
                continue
        # print(q)
        if not discard or discard and words_exist:
            q_embeds.append(q_embed)
        else:
            q_embeds.append([None])
    return q_embeds

def embed_image_vgg(img_path, model):
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
    
    features = model.predict(x)
    fc2_features_extractor_model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
    
    fc2_features = fc2_features_extractor_model.predict(x)
    fc2_features = fc2_features.reshape((4096,))
    
    return fc2_features

def img_id_to_path(img_id, train=True):
    if train:
        return train_image_dir + train_image_base + str(img_id).zfill(12) + ".png"
    else:
        return val_image_dir + val_image_base + str(img_id).zfill(12) + ".png"

def img_id_to_embed_path(img_id, train=True):
    if train:
        return train_image_emb_dir + str(img_id) + ".npz"
    else:
        return val_image_emb_dir + str(img_id) + ".npz"

def format_q_for_embed(q_string):
    return re.sub(r'[^\w\s]','',q_string).lower()















