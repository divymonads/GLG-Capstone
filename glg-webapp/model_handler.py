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


class LDAWrapper(object):
    def __init__(self):
        self.initialized = False

        # main lda model
        self.lda_model = None

        self.bigram_model = None
        self.trigram_model = None
        self.dictionary = None

        # used for preprocessing
        self.spacy_en_sm = None
        self.sw_spacy = None
        self.sw_nltk = None
        self.lemmatizer_ntlk = None
        self.porter_stemmer = None


    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True

        # Used for preprocessing
        #nltk.download('stopwords')
        #nltk.download('punkt')
        self.spacy_en_sm = spacy.load('en_core_web_sm')
        self.sw_spacy = self.spacy_en_sm.Defaults.stop_words
        self.sw_nltk = nltk.corpus.stopwords.words('english')
        self.lemmatizer_ntlk =  WordNetLemmatizer()
        self.porter_stemmer = PorterStemmer()

        prefix = './model_artifacts_50kfilter_full/'

        # Load our LDA model artifacts
        self.bigram_model = Phraser.load(os.path.join(prefix, 'bigram_model'))
        self.trigram_model = Phraser.load(os.path.join(prefix, 'trigram_model'))
        self.dictionary = corpora.Dictionary.load(os.path.join(prefix, "id2word"))

        with open(os.path.join(prefix, 'lda_model_25.pk'), 'rb') as pickle_file:
            self.lda_model = pickle.load(pickle_file)


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
        def stopwords(text):
            return " ".join([word for word in str(text).split() if word not in self.sw_nltk])
        # remove more stopwords from Spacy
        def spacy_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in self.sw_spacy])

        # Applying the stopwords
        new_text = stopwords(new_text)
        new_text = spacy_stopwords(new_text)

        # (14) Lemmatization (convert words into its base form)
        new_text = ' '.join([self.lemmatizer_ntlk.lemmatize(word,'v') for word in new_text.split()])

        # (15) Stemming
        new_text = ' '.join([self.porter_stemmer.stem(word) for word in new_text.split()])

        return new_text

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

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output
        """
        corpus_text = model_input
        return self.lda_model.get_document_topics(corpus_text, minimum_probability=0)

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        from topic_map import Mapper

        m = Mapper()
        ret_list = [(i, m.get(i), prob) for i, prob in inference_output]
        best_topic = max(ret_list, key=lambda x: x[2])
        ret_dict = {'list': ret_list, 'expert': m.getExpert(best_topic[0]), 'topic': m.get(best_topic[0])}
        return ret_dict

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        processed_text = self.preprocess(data)
        model_input = self.tokenize_and_corpize(processed_text)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)
