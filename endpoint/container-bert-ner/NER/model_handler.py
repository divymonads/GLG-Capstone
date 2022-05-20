import pandas as pd
import os, pickle
from pandas.core.groupby import groupby
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoConfig, TFAutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk import tokenize

PRETRAINED_MODEL_NAME = 'bert-base-uncased'
FINETUNED_MODEL_NAME = 'finetuned_' + PRETRAINED_MODEL_NAME
FILE_DIR = '/opt/ml/model'
SEQUENCE_LENGTH = 128

class NERModel(object):
    def __init__(self):
        model = None
        config = None
        backbone = None

    def build_model(self, num_classes, use_finetuned=False):
        self.config = AutoConfig.from_pretrained(PRETRAINED_MODEL_NAME)
        self.backbone = TFAutoModel.from_pretrained(PRETRAINED_MODEL_NAME,config=self.config)

        tokens = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name = 'tokens', dtype=tf.int32)
        att_masks = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name = 'attention', dtype=tf.int32)

        features = self.backbone(tokens, attention_mask=att_masks)[0]

        target = tf.keras.layers.Dropout(0.5)(features)
        target = tf.keras.layers.Dense(num_classes, activation='softmax')(target)

        self.model = tf.keras.Model([tokens,att_masks],target)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=['accuracy'])
        if use_finetuned:
            self.model.load_weights(os.path.join(FILE_DIR, FINETUNED_MODEL_NAME))


class NERWrapper(object):
    initialized = False
    tag_encoder = None
    tokenizer = None
    background_class = None
    ner_modeler = None

    LABEL_CONVERT = {'org': 'ORG',
                      'tim': 'DATE',
                      'per': 'PERSON',
                      'geo': 'GEO',
                      'gpe': 'GPE',
                      'art': 'ART',
                      'eve': 'EVE',
                      'nat': 'NAT',
                      }
    @classmethod
    def initialize(self):
        self.initialized = True
        with open(os.path.join(FILE_DIR, 'ner_label_encoder.pkl'), 'rb') as le:
            self.tag_encoder = pickle.load(le)

        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, normalization=True)
        self.background_class = self.tag_encoder.transform(['O'])[0]

        n_classes = self.tag_encoder.classes_.shape[0]

        #Load the finetuned model
        self.ner_modeler = NERModel()
        self.ner_modeler.build_model(n_classes, True)

    @classmethod
    def run_ner_on_sentence(self, sample_text):
        #Tokenize the sample text, and get the word ids
        encoded = self.tokenizer.encode_plus(sample_text,
                                        add_special_tokens = True,
                                        max_length = SEQUENCE_LENGTH,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        padding = 'max_length',
                                        truncation=True,return_tensors = 'np')
        input_seq = encoded['input_ids']
        att_mask = encoded['attention_mask']
        word_ids = encoded.word_ids()

        #Predict the classes for each token
        sample_out = self.ner_modeler.model.predict([input_seq, att_mask])
        sample_out = np.argmax(sample_out, axis=2)
        word_ids = np.array(word_ids)
        valid_sample_out = sample_out[0, word_ids!=None]
        valid_word_ids = word_ids[word_ids!=None]
        names = [sample_text[i] for i in valid_word_ids[valid_sample_out!=self.background_class]]
        labels = [self.tag_encoder.inverse_transform([i])[0] for i in valid_sample_out[valid_sample_out!=self.background_class]]

        #Combine the tokens and correponding labels. Output the final names and their corresponding classes
        full_names = []
        full_labels = []
        prev_index = -1
        completed = {}
        for name, label in zip(names, labels):
            if(name not in completed):
                if(label[0]=='B'):
                    full_names.append(name)
                    full_labels.append(self.LABEL_CONVERT[label[2:]])
                    prev_index += 1
                else:
                    if(len(full_names)>0):
                        full_names[prev_index] = full_names[prev_index] + ' ' + name
                    else:
                        continue
                completed[name] = 1
        return full_names, full_labels

    @classmethod
    def postprocess_sentence(self, snames, slabels, sentence):
        start_idxs = []
        stop_idxs = []
        start_search_idx = 0

        for key, value in zip(snames, slabels):
            start = sentence.find(key, start_search_idx)
            start_idxs.append(start)
            stop_idxs.append(start + len(key))
            start_search_idx = stop_idxs[-1]

        return start_idxs, stop_idxs

    @classmethod
    def handle(self, full_text, display=False):
        sentences = tokenize.sent_tokenize(full_text)

        names = []
        labels = []
        start_idxs = []
        stop_idxs = []
        total_start_len = 0
        for sentence in sentences:
            snames, slabels = self.run_ner_on_sentence(sentence.split(' '))
            s_starts, s_stops = self.postprocess_sentence(snames, slabels, sentence)
            names.extend(snames)
            labels.extend(slabels)
            for s_i in s_starts:
                start_idxs.append(s_i + total_start_len)
            for e_i in s_stops:
                stop_idxs.append(e_i + total_start_len)

            # assumes that each sentence is separated by a space
            total_start_len += len(sentence) + 1

        if display:
            print(full_text)
            for i, j, n, l in zip(start_idxs, stop_idxs, names, labels):
                print(i, j, n, l)
        return {'start_idxs':start_idxs, 'stop_idxs':stop_idxs, 'names':names, 'labels':labels}
