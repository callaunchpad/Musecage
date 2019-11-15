from VQA.PythonHelperTools.vqaTools.vqa import VQA
import random
import os
import re
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

data_dir = "data"
vqa_data_dir = "vqa"
image_dir = "abstract_images"
image_base = "abstract_v002_train2015_"
image_dir = "%s/%s/%s/"%(data_dir, vqa_data_dir, image_dir)
ann_file = "%s/%s/Annotations/abstract_v002_train2015_annotations.json"%(data_dir, vqa_data_dir)
ques_file = "%s/%s/Questions/OpenEnded_abstract_v002_train2015_questions.json"%(data_dir, vqa_data_dir)
img_dir = "%s/%s/abstract_images/"%(data_dir, vqa_data_dir)

glove_dir = "glove.6B"
glove_dim = glove_dir + ".%dd.txt"%(300)
glove_file = "%s/glove/%s/%s"%(data_dir, glove_dir, glove_dim)

vqa=VQA(ann_file, ques_file)

def get_by_ques_type(quesTypes):
    ans_ids = vqa.getQuesIds(quesTypes=quesTypes)
    answers = vqa.loadQA(ans_ids)
    finals = []
    for an in answers:
        qa = vqa.qqa[an["question_id"]]
        final = qa.copy()
        final["answers"] = [a["answer"] for a in an["answers"]]
        finals.append(final)
    return finals

def get_by_ans_type(ansTypes):
    ans_ids = vqa.getQuesIds(ansTypes=ansTypes)
    answers = vqa.loadQA(ans_ids)
    finals = []
    for an in answers:
        qa = vqa.qqa[an["question_id"]]
        final = qa.copy()
        final["answers"] = [a["answer"] for a in an["answers"]]
        finals.append(final)
    return finals

def get_by_img_ids(img_ids):
    ans_ids = vqa.getQuesIds(imgIds=img_ids)
    answers = vqa.loadQA(ans_ids)
    finals = []
    for an in answers:
        qa = vqa.qqa[an["question_id"]]
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

def img_id_to_path(img_id):
    return image_dir + image_base + str(img_id).zfill(12) + ".png"

def format_q_for_embed(q_string):
    return re.sub(r'[^\w\s]','',q_string).lower()















