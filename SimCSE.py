# encoding=utf8
import os
import codecs
import pickle
import itertools
import sys
import math
import scipy.stats
import random
from collections import OrderedDict
sys.path.append('../../')

import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from bert import modeling, tokenization


class SimCSEBatchManager(object):
    def __init__(self, data, batch_size, ttype):
        self.ttype = ttype
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        #sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(data[int(i*batch_size) : int((i+1)*batch_size)], self.ttype))
        return batch_data

    @staticmethod
    def arrange_batch(batch, ttype):
        '''
        batch as a [3, ] array
        :param batch:
        :return:
        '''
        word_id_list = []
        word_mask_list = []
        word_segment_list = []
        labels = []
        if ttype == 'train':
            for word_ids, input_masks, word_segment_ids in batch:
                word_id_list.append(word_ids)
                word_mask_list.append(input_masks)
                word_segment_list.append(word_segment_ids)
            return [word_id_list, word_mask_list, word_segment_list]
        elif ttype == 'test':
            for word_ids, input_masks, word_segment_ids, label in batch:
                word_id_list.append(word_ids)
                word_mask_list.append(input_masks)
                word_segment_list.append(word_segment_ids)
                labels.append(label)
            return [word_id_list, word_mask_list, word_segment_list, labels]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


class BertSimCSE(object):
    def __init__(self, config):
        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.max_seq_lens = config['max_seq_lens']
        self.batch_size = config['batch_size']

        self.init_checkpoint = config['init_checkpoint']
        self.config_file = config['config_file']
        self.vocab_file = config['vocab_file']
        self.lr = config['lr']
        self.lower = config['lower']
        self.epochs = config['epochs']
        self.steps_check = config['steps_check']
        self.save_path = config['save_path']
        self.train_file = config['train_file']
        self.dev_file = config['dev_file']
        self.test_file = config['test_file']
        self.encode_file = config['encode_file']
        self.is_training = config['is_training']

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        input_ids = tf.reshape(self.input_ids, [-1, self.max_seq_lens])
        input_mask = tf.reshape(self.input_mask, [-1, self.max_seq_lens])
        segment_ids = tf.reshape(self.segment_ids, [-1, self.max_seq_lens])

        bert_config = modeling.BertConfig.from_json_file(self.config_file)
        
        # is_training: bool. true for training model, false for eval model. Controls whether dropout will be applied.
        model = modeling.BertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        self.pooled = model.get_pooled_output()  # [batch_size, 768]

        self.loss = self.simcse_loss()
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # print variables
        train_vars = []
        for var in tvars:
            init_string = ""
            # freeze bert parameters, only train the parameters of add networks
            if var.name in initialized_variable_names:
                train_vars.append(var)
                init_string = ", *INIT_FROM_CKPT*"
            else:
                train_vars.append(var)
            print("  name = %s, shape = %s%s", var.name, var.shape,
                  init_string)

        with tf.variable_scope("optimizer"):
            
            self.opt = tf.train.AdamOptimizer(self.lr)

            grads = tf.gradients(self.loss, train_vars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = self.opt.apply_gradients(
                zip(grads, train_vars), global_step=self.global_step)
            #capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                     for g, v in grads_vars if g is not None]
            #self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step, )
    
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def simcse_loss(self):
        # construct label
        idxs = tf.range(0, tf.shape(self.pooled)[0])
        idxs_1 = idxs[None, :]
        #print('idxs_1.shape: ', idxs_1.shape)
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        #print('idxs_2.shape: ', idxs_2.shape)
        y_true = tf.equal(idxs_1, idxs_2)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.l2_normalize(self.pooled, axis=1)
        self.similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
        similarities = self.similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
        similarities = similarities * 20
        loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
        return tf.reduce_mean(loss)

    def create_feed_dict(self, is_train, batch):
        """
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        if is_train:
            word_ids, input_masks, word_segment_ids = batch
        else:
            word_ids, input_masks, word_segment_ids, labels = batch
        feed_dict = {
            self.input_ids: np.asarray(word_ids),
            self.input_mask: np.asarray(input_masks),
            self.segment_ids: np.asarray(word_segment_ids)
        }
        return feed_dict
    
    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            pooled = sess.run([self.pooled], feed_dict)
            return pooled

    def compute_corrcoef(self, x, y):
        """Spearman correlation coefficient
        """
        return scipy.stats.spearmanr(x, y).correlation

    def evaluate(self, sess, data_manager):
        all_corrcoefs = []
        all_sims = []
        all_labels = []

        for batch in data_manager.iter_batch(shuffle=False):
            pair_sentence_embs = self.run_step(sess, False, batch)
            pair_sentence_embs = pair_sentence_embs[0]
            sentences_0 = pair_sentence_embs[::2]
            sentences_1 = pair_sentence_embs[1::2]
            sims = (sentences_0 * sentences_1).sum(axis=1)
            labels = batch[3]
            all_sims.extend(sims.tolist())
            # avoid nan for compute_corrcoef
            all_labels.extend(labels)
        corrcoef = self.compute_corrcoef(all_labels, all_sims)
        return corrcoef
        
    def save_model(self, sess, path):
        checkpoint_path = os.path.join(path, "bert_SimCSE.ckpt")
        self.saver.save(sess, checkpoint_path)
        print("model saved")

    def read_input(self, filename, ttype):
        """
        # TODO: add lower function
        """
        assert ttype in ['train', 'test']
        token = tokenization.FullTokenizer(vocab_file=self.vocab_file)
        sentences = []
        labels = []
        for i, line in enumerate(codecs.open(filename, 'r', 'utf8')):
            # if i == 0:
            #     continue
            items = line.strip().split('\t')
            sentence_0 = items[0]
            sentence_1 = items[1]
            label = int(items[2])
            sentences.append([sentence_0, sentence_1])
            labels.append(label)
            
        data = []
        
        def convert_single_sentence_to_bert_input(sentence):
            split_tokens = token.tokenize(sentence)
            input_masks = [1] * len(split_tokens)
            
            # cut and pad sentence to max_seq_lens-2
            if len(split_tokens) > self.max_seq_lens-2:
                split_tokens = split_tokens[:self.max_seq_lens-2]
                split_tokens.append("[SEP]")
                input_masks = input_masks[:self.max_seq_lens-2]
                input_masks = [1] + input_masks + [1]
            else:
                split_tokens.append("[SEP]")
                input_masks.append(1)
                while len(split_tokens) < self.max_seq_lens-1:
                    split_tokens.append('[PAD]')
                    input_masks.append(0)
                input_masks.append(0)
            # add CLS and SEP for tokens
            tokens = []
            tokens.append("[CLS]")
            for i_token in split_tokens:
                tokens.append(i_token)
            
            word_ids = token.convert_tokens_to_ids(tokens)
            word_segment_ids = [0] * len(word_ids)
            return word_ids, input_masks, word_segment_ids

        # convert pair sentence and concat it for input

        for sentence, label in zip(sentences, labels):
            if ttype == 'train':
                if label == 1:
                    word_ids, input_masks, word_segment_ids = convert_single_sentence_to_bert_input(sentence[0])
                    word_ids_1, input_masks_1, word_segment_ids_1 = convert_single_sentence_to_bert_input(sentence[1]) 
                    word_ids.extend(word_ids_1)
                    input_masks.extend(input_masks_1)
                    word_segment_ids.extend(word_segment_ids_1)
                    data.append([word_ids, input_masks, word_segment_ids])
                else:
                    word_ids, input_masks, word_segment_ids = convert_single_sentence_to_bert_input(sentence[0])
                    word_ids_1, input_masks_1, word_segment_ids_1 = convert_single_sentence_to_bert_input(sentence[0])
                    word_ids.extend(word_ids_1)
                    input_masks.extend(input_masks_1)
                    word_segment_ids.extend(word_segment_ids_1)
                    data.append([word_ids, input_masks, word_segment_ids])
                    word_ids, input_masks, word_segment_ids = convert_single_sentence_to_bert_input(sentence[1])
                    word_ids_1, input_masks_1, word_segment_ids_1 = convert_single_sentence_to_bert_input(sentence[1])
                    word_ids.extend(word_ids_1)
                    input_masks.extend(input_masks_1)
                    word_segment_ids.extend(word_segment_ids_1)
                    data.append([word_ids, input_masks, word_segment_ids])
            elif ttype == 'test':
                word_ids, input_masks, word_segment_ids = convert_single_sentence_to_bert_input(sentence[0])
                word_ids_1, input_masks_1, word_segment_ids_1 = convert_single_sentence_to_bert_input(sentence[1]) 
                word_ids.extend(word_ids_1)
                input_masks.extend(input_masks_1)
                word_segment_ids.extend(word_segment_ids_1)
                data.append([word_ids, input_masks, word_segment_ids, label])
            # tmp_data = list(zip(data, sentences))
            # random.shuffle(tmp_data)
            # for t in tmp_data:
            #     print(t)
            # data[:], sentences[:] = zip(*tmp_data)
        return data, sentences

    def train(self):
        train_data, train_sentences = self.read_input(self.train_file, 'train')
        
        dev_data, dev_sentences = self.read_input(self.dev_file, 'test')
        test_data, test_sentences = self.read_input(self.test_file, 'test')

        print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

        train_manager = SimCSEBatchManager(train_data, self.batch_size, 'train')
        dev_manager = SimCSEBatchManager(dev_data, self.batch_size, 'test')
        test_manager = SimCSEBatchManager(test_data, self.batch_size, 'test')

        # limit GPU memory
        tf_config = tf.ConfigProto()
        #tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data

        with tf.Session(config=tf_config) as sess:
            
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

            ckpt = tf.train.get_checkpoint_state(self.save_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                #saver = tf.train.import_meta_graph('ckpt/ner.ckpt.meta')
                #saver.restore(session, tf.train.latest_checkpoint("ckpt/"))
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())
            
            print("start training")
            loss = []
            dev_best = 0.0
            for i in range(self.epochs):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = self.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % self.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        print("iteration:{} step:{}/{}, "
                                "loss:{:>9.6f}".format(
                            iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

                dev_corrcoef = self.evaluate(sess, dev_manager)
                if dev_corrcoef > dev_best:
                    dev_best = dev_corrcoef
                    self.save_model(sess, self.save_path)
                    print('model save!')

                test_corrcoef = self.evaluate(sess, test_manager)
                print("iteration:{}, dev corrcoef: {}, test corrcoef: {}".format(
                            i+1, dev_corrcoef, test_corrcoef))

    def eval_bert_base(self):
        test_data, test_sentences = self.read_input(self.test_file, 'test')
        dev_data, dev_sentences = self.read_input(self.dev_file, 'test')

        test_manager = SimCSEBatchManager(test_data, self.batch_size, 'test')
        dev_manager = SimCSEBatchManager(dev_data, self.batch_size, 'test')
        # limit GPU memory
        tf_config = tf.ConfigProto()
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            dev_corrcoef = self.evaluate(sess, dev_manager)
            test_corrcoef = self.evaluate(sess, test_manager)
            print("dev corrcoef: {}, test corrcoef: {}".format(
                        dev_corrcoef, test_corrcoef))

    def encode(self):
        infer_data, infer_sentences = self.read_input(self.encode_file)
        infer_manager = SimCSEBatchManager(infer_data, self.batch_size)

        # limit GPU memory
        tf_config = tf.ConfigProto()
        #tf_config.gpu_options.allow_growth = True
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        embs_list = []
        with tf.Session(config=tf_config) as sess:
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                #saver = tf.train.import_meta_graph('ckpt/ner.ckpt.meta')
                #saver.restore(session, tf.train.latest_checkpoint("ckpt/"))
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())

            for batch in infer_manager.iter_batch(shuffle=False):
                word_ids, input_masks, word_segment_ids, labels = batch
                feed_dict = {
                    self.input_ids: np.asarray(word_ids),
                    self.input_mask: np.asarray(input_masks),
                    self.segment_ids: np.asarray(word_segment_ids)
                }
                batch_embs = sess.run([self.pooled], feed_dict)
                embs_list.append(batch_embs[0])
                
            embeddings = np.vstack(embs_list)
        return embeddings

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    config = {
        'init_checkpoint': "../../data/chinese_L-12_H-768_A-12/bert_model.ckpt",
        'config_file': "../../data/chinese_L-12_H-768_A-12/bert_config.json",
        'vocab_file': "../../data/chinese_L-12_H-768_A-12/vocab.txt",
        'max_seq_lens': 64,
        'batch_size': 32,
        'lr': 0.00001,
        'lower': True,
        'epochs': 100,
        'steps_check': 2,
        'save_path': 'ckpt',
        'train_file': './sim_data/ATEC.train.data',
        'dev_file': './sim_data/ATEC.valid.data',
        'test_file': './sim_data/ATEC.test.data',
        'encode_file': '',
        'is_training': False
    }
    model = BertSimCSE(config)
    #model.train()
    model.eval_bert_base()
    