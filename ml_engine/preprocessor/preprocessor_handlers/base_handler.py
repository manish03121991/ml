import os
from .utils import read_data_from_file
from .constants import tokenizer, stemmer, lemmatizer, ngram, pos_tagger, base_stop_words_file_path


class BaseTextProcessing(object):
    """
    This is the base class defined for every handlers used on text preprocessing,
    the engine would set the flags and different attributes in this class.

    The attributes of this class is set for every handlers when called by engine.

    In order to make this class as base, we need to define the _handle_processing
    method in our subclass.This method would call from preprocess_text method.

    text: it is the string on which processing take place
    filter_stopwords: It is a filter on stopwords to be used as per the requirement
                      such as if using sentiment model then require sentiment stopwords
                      else default filter is set for stopwords
    flag_tokenize: True if we need to apply tokenization on the text else False
    flag_stem: True if we need to find stem from tokens else False
    flag_punctuation_removal: True if we need to remove punctuations from text else False
    flag_lemma: True if we need to find lemmatization from text else False
    flag_tokenize_without_stop_words: True if we want to remove stop words from tokens else False
    flag_stem_without_stop_words: True if we want to remove stop words from stem tokens else False
    flag_lemma_without_stop_words: True if we want to remove stop words from lemma tokens else False
    flag_unigram_without_stop_words: True if we want to remove stop words from unigram else False
    flag_bigram_without_stop_words: True if we want to remove stop words from bigram else False
    flag_trigram_without_stop_words: True if we wan to remove stop words from trigram else False
    stem_unigram: True if want to find unigram on stem tokens else False on tokens
    stem_bigram: True if want to find bigram on stem tokens else False on tokens
    stem_trigram: True if want to find trigram on stem tokens else False on tokens
    flag_pos_lemma: True if want to find the lemma tokens of specific part of speech in text else False
    flag_pos_lemma_without_stopwords: True if want to find the lemma tokens without stopwords
                                      of specific part of speech in text else False
    """

    def __init__(self, flag_tokenize=True, flag_stem=True, flag_punctuation_removal=False, flag_lemma=True,
                 flag_tokenize_without_stop_words=False, flag_stem_without_stop_words=False,
                 flag_lemma_without_stop_words=False, flag_unigram_without_stop_words=False,
                 flag_bigram_without_stop_words=False, flag_trigram_without_stop_words=False,
                 stem_unigram=False, stem_bigram=False, stem_trigram=False, flag_pos_lemma=False,
                 flag_pos_lemma_without_stopwords=False):
        self.text = None
        self.flag_tokenize = flag_tokenize
        self.flag_stem = flag_stem
        self.flag_punctuation_removal = flag_punctuation_removal
        self.flag_lemma = flag_lemma
        self.flag_tokenize_without_stop_words = flag_tokenize_without_stop_words
        self.flag_stem_without_stop_words = flag_stem_without_stop_words
        self.flag_lemma_without_stop_words = flag_lemma_without_stop_words

        self.flag_unigram_without_stop_words = flag_unigram_without_stop_words
        self.flag_bigram_without_stop_words = flag_bigram_without_stop_words
        self.flag_trigram_without_stop_words = flag_trigram_without_stop_words

        self.stem_unigram = stem_unigram
        self.stem_bigram = stem_bigram
        self.stem_trigram = stem_trigram

        self.flag_pos_lemma = flag_pos_lemma
        self.flag_pos_lemma_without_stopwords = flag_pos_lemma_without_stopwords

        self.stop_words_list = None

        self.stemmer = stemmer
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.ngram = ngram
        self.pos_tagger = pos_tagger

    def preprocess_text(self, text, filter_stopwords, **kwargs):
        self.text = text
        self.stop_words_list = self.set_stopwords_in_engine(filter_stopwords)
        response = self._handle_processing(**kwargs)
        return response

    def set_stopwords_in_engine(self, filter_stopwords):
        stop_words_file_path = os.path.join(
            base_stop_words_file_path, '%s_stopwords.csv' % (filter_stopwords)
        )
        # load stop words in memory
        stop_words_list = read_data_from_file(stop_words_file_path)
        return stop_words_list

    def _handle_processing(self, **kwargs):
        raise NotImplementedError("Must implement this method in subclass")
