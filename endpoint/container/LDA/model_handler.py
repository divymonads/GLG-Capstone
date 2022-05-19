import glob, json, os, re, pickle
from collections import namedtuple
import numpy as np
import nltk
import contractions
from nltk.tokenize import word_tokenize
# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.models.phrases import Phraser
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

class LDAResult(object):
    def __init__(self, i, prob):
        from topic_map import Mapper
        topic_mapper = Mapper()

        self.topic_index = i
        self.probability = prob
        self.topic_name = topic_mapper.get(i)
        self.topic_expert = topic_mapper.getExpert(i)

    def toDict(self):
        return self.__dict__

class LDAWrapper(object):
    initialized = False

    # main lda model
    lda_model = None

    bigram_model = None
    trigram_model = None
    dictionary = None

    # used for preprocessing
    spacy_en_sm = None
    sw_spacy = None
    sw_nltk = None
    lemmatizer_ntlk = None
    porter_stemmer = None

    @classmethod
    def initialize(self):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True

        # Used for preprocessing
        self.spacy_en_sm = spacy.load('en_core_web_sm')
        self.sw_spacy = self.spacy_en_sm.Defaults.stop_words
        self.sw_nltk = nltk.corpus.stopwords.words('english')
        self.lemmatizer_ntlk =  WordNetLemmatizer()
        self.porter_stemmer = PorterStemmer()

        prefix = "/opt/ml/"
        model_path = os.path.join(prefix, "model")


        # Load our LDA model artifacts
        self.bigram_model = Phraser.load(os.path.join(model_path, 'bigram_model'))
        self.trigram_model = Phraser.load(os.path.join(model_path, 'trigram_model'))
        self.dictionary = corpora.Dictionary.load(os.path.join(model_path, "id2word"))

        with open(os.path.join(model_path, 'lda_model_25.pk'), 'rb') as pickle_file:
            self.lda_model = pickle.load(pickle_file)

    @classmethod
    def preprocess(self, text):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        ## (1) Convert to lower cases
        new_text = " ".join([word.lower() for word in text.split()])
        # (2) Remove words with a length below 2 characters
        new_text = ' '.join([word for word in new_text.split() if len(word) > 1 ])

        ## (3) Removal of URL's
        def remove_urls(text):
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            # remove words starting with https and with www
            return url_pattern.sub(r'', text)

        new_text = remove_urls(new_text)

        # (4) Replace multiple white spaces with one white space
        new_text = ' '.join([word for word in new_text.split() ])

        # (5) Remove numbers (how to judge if the number is relevant??)
        new_text = ' '.join([word for word in new_text.split() if not word.isdigit()])
        # number was not remove earlier
        new_text = new_text.replace(r'\d+','')

        # (7) Remove all punctuations (for example, parenthesis, comma, period, etc.)
        new_text = new_text.replace('[^\w\s]','')

        # (8) Remove Emails
        new_text = ''.join([re.sub('\S*@\S*\s?','', word) for word in new_text])

        # (9) Remove new line characters
        new_text = "".join([re.sub('\s+',' ', word) for word in new_text])

        # (10) Remove distracting single quotes
        new_text = ''.join([re.sub("\'","", word) for word in new_text])

        # (12) Expand contractions
        new_text = ' '.join([contractions.fix(word) for word in new_text.split() ])

        # (13) remove stopwords (the, to be, etc.)
        # Function to remove the stopwords
        def stopwords(text, sw_list):
            return " ".join([word for word in str(text).split() if word not in sw_list])

        # Applying the stopwords
        new_text = stopwords(new_text, self.sw_nltk)
        new_text = stopwords(new_text, self.sw_spacy)

        # (14) Lemmatization (convert words into its base form)
        new_text = ' '.join([self.lemmatizer_ntlk.lemmatize(word,'v') for word in new_text.split()])

        # (15) Stemming
        new_text = ' '.join([self.porter_stemmer.stem(word) for word in new_text.split()])

        return new_text

    @classmethod
    def tokenize_and_corpize(self, new_text):
        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            doc = self.spacy_en_sm(" ".join(texts))
            texts_out = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            return texts_out

        def make_trigrams(texts):
            return self.trigram_model[self.bigram_model[texts]]

        token_words = word_tokenize(new_text)
        token_words_trigrams = make_trigrams(token_words)
        token_words_trigrams_lemm = lemmatization(token_words_trigrams,
            allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        corpus_text = self.dictionary.doc2bow(token_words_trigrams_lemm)
        return corpus_text

    @classmethod
    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output
        """
        corpus_text = model_input
        return self.lda_model.get_document_topics(corpus_text, minimum_probability=0)

    @classmethod
    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        ret_list = [LDAResult(i, probability.item()) for i, probability in inference_output]
        best_topics = sorted([x for x in ret_list if x.probability > 0.2], key=lambda x: -x.probability)

        ret_dict = {'distribution': [x.toDict() for x in ret_list], \
                'topics': [x.toDict() for x in best_topics]}
        return ret_dict

    @classmethod
    def handle(self, data):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        processed_text = self.preprocess(data)
        model_input = self.tokenize_and_corpize(processed_text)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)
