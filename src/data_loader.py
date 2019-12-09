from load_create_data import *

data_len = 90000
split_val = 0.8

data_arr = (get_by_ques_type([], train=True) + get_by_ques_type([], train=False))[:data_len]
train_data = data_arr[:int(len(data_arr) * split_val)]
train_q_arr = [format_q_for_embed(val["question"]) for val in train_data]
train_ans_type_arr = [val["answer_type"] for val in train_data]
print(train_ans_type_arr)


    def next_batch(self, train=True, replace=False):
        next_batch_avail = True
        if train:
            if not replace:
                self.q_batch = self.train_q_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.ans_type_batch = self.train_ans_type_arr[self.train_curr_index : self.train_curr_index + self.batch_size]
                self.train_curr_index += self.batch_size
                if self.train_curr_index >= len(self.train_q_arr):
                    next_batch_avail = False
            else:
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
                self.ans_type_batch = self.test_ans_type_arr[self.test_curr_index : self.test_curr_index + self.batch_size]
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
