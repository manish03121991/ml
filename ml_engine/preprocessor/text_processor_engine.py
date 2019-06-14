
from ml_engine.preprocessor.preprocessor_handlers import PunctuationHandler, \
    NgramHandler, TokenizerHandler, StemmerHandler, LemmatizerHandler, POSLemmaHandler

PROCESSING_SEQUENCE = [
    PunctuationHandler,
    TokenizerHandler,
    StemmerHandler,
    LemmatizerHandler,
    POSLemmaHandler,
    NgramHandler
]


class TextProcessorEngine(object):
    """
    This class acts as an engine for the text pre-processing steps and would derive the
    handlers in the sequence provided.We always need to define the handlers for text pre-processing in
    particular sequence for the engine to work in the same fashion.

    Our main motive is to fill the engine with possible flags which helps the handlers
    to run different pre-processing steps on provided text.

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
    flag_pos_lemma: True if want to find the leema tokens of specific part of speech in text else False
    flag_pos_lemma_without_stopwords: True if want to find the leema tokens without stopwords
                                      of specific part of speech in text else False

    After applying the handlers on text, engine would normalize the output received from different
    handlers.

    output:
        engine=TextProcessorEngine()
        engine.text_processor(text="I want to travel from mumbai to new delhi")
        {"data":
          {"bigrams": ["i want", "want to", "to travel", "travel from", "from mumbai", "mumbai to",
                        "to new", "new delhi"],
          "unigrams": ["i", "want", "to", "travel", "from", "mumbai", "to", "new", "delhi"],
          "tokens": ["i", "want", "to", "travel", "from", "mumbai", "to", "new", "delhi"],
          "lemma_tokens": ["i", "want", "to", "travel", "from", "mumbai", "to", "new", "delhi"],
          "tokens_without_stop_words": [],
          "text_lowercase": "i want to travel from mumbai to new delhi",
          "stem_without_stop_word": [],
          "trigrams": ["i want to", "want to travel", "to travel from", "travel from mumbai",
                       "from mumbai to", "mumbai to new", "to new delhi"],
          "text": "I want to travel from mumbai to new delhi",
          "lemma_without_stop_word": [],
          "stem_tokens": ["i", "want", "to", "travel", "from", "mumbai", "to", "new", "delhi"]}
          "flag_pos_lemma":[],
          "flag_pos_lemma_without_stopwords":[]
          }}

        engine.text_processor(text="Hey, can you please help me in buying a brand new car?",flag_punctuation_removal=True,
                             flag_unigram_without_stop_words=True,flag_bigram_without_stop_words=True,
                             flag_trigram_without_stop_words=True,stem_bigram=True)
        {"data":
          {"lemma_tokens": ["hey", "can", "you", "please", "help", "me", "in", "buying", "a",
                            "brand", "new", "car"],
           "text_lowercase": "hey, can you please help me in buying a brand new car?",
           "text": "Hey, can you please help me in buying a brand new car?",
           "lemma_without_stop_word": [],
           "trigrams": ["hey can you", "can you please", "you please help", "please help me",
                        "help me in", "me in buying", "in buying a", "buying a brand", "a brand new",
                        "brand new car"],
           "stem_tokens": ["hey", "can", "you", "pleas", "help", "me", "in", "buy", "a", "brand",
                           "new", "car"],
           "stem_without_stop_word": [],
           "bigrams": ["hey can", "can you", "you pleas", "pleas help", "help me", "me in",
                       "in buy", "buy a", "a brand", "brand new", "new car"],
           "tokens_without_stop_words": [],
           "unigrams": ["hey", "can", "you", "please", "help", "me", "in", "buying", "brand", "new", "car"],
           "tokens": ["hey", "can", "you", "please", "help", "me", "in", "buying", "a", "brand", "new", "car"],
           "flag_pos_lemma":[],
          "flag_pos_lemma_without_stopwords":[]}
           }

    """

    def __init__(self, text_processing_sequence=PROCESSING_SEQUENCE):
        self.text_processing_sequence = text_processing_sequence

    def __normalize_engine_output(self, text, **kwargs):
        normalize_dict = dict()
        normalize_dict['tokens'] = kwargs['tokens']
        normalize_dict['tokens_without_stop_words'] = kwargs['tokenize_without_stop_words']
        normalize_dict['stem_tokens'] = kwargs['stem_tokens']
        normalize_dict['text'] = text
        normalize_dict['text_lowercase'] = text.lower()
        normalize_dict['lemma_tokens'] = kwargs['lemma_tokens']
        normalize_dict['lemma_without_stop_word'] = kwargs['lemma_without_stop_words']
        normalize_dict['stem_without_stop_word'] = kwargs['stem_without_stop_words']
        normalize_dict['unigrams'] = kwargs['unigrams']
        normalize_dict['bigrams'] = kwargs['bigrams']
        normalize_dict['trigrams'] = kwargs['trigrams']
        normalize_dict['pos_lemma_tokens'] = kwargs['pos_lemma_tokens']
        normalize_dict['pos_lemma_tokens_without_stopwords'] = kwargs['pos_lemma_tokens_without_stopwords']
        return normalize_dict

    def text_processor(self, text, filter_stopwords='default', flag_tokenize=True, flag_stem=True,
                       flag_punctuation_removal=False,
                       flag_lemma=True, flag_tokenize_without_stop_words=False, flag_stem_without_stop_words=False,
                       flag_lemma_without_stop_words=False, flag_unigram_without_stop_words=False,
                       flag_bigram_without_stop_words=False, flag_trigram_without_stop_words=False,
                       stem_unigram=False, stem_bigram=False, stem_trigram=False,
                       flag_pos_lemma_without_stopwords=False,
                       flag_pos_lemma=False, **kwargs):
        for processor in self.text_processing_sequence:
            # log processing time
                kwargs = processor(
                    flag_tokenize=flag_tokenize,
                    flag_stem=flag_stem,
                    flag_punctuation_removal=flag_punctuation_removal,
                    flag_lemma=flag_lemma,
                    flag_tokenize_without_stop_words=flag_tokenize_without_stop_words,
                    flag_stem_without_stop_words=flag_stem_without_stop_words,
                    flag_lemma_without_stop_words=flag_lemma_without_stop_words,
                    flag_unigram_without_stop_words=flag_unigram_without_stop_words,
                    flag_bigram_without_stop_words=flag_bigram_without_stop_words,
                    flag_trigram_without_stop_words=flag_trigram_without_stop_words,
                    stem_unigram=stem_unigram,
                    stem_bigram=stem_bigram,
                    stem_trigram=stem_trigram,
                    flag_pos_lemma=flag_pos_lemma,
                    flag_pos_lemma_without_stopwords=flag_pos_lemma_without_stopwords,

                ).preprocess_text(text, filter_stopwords,**kwargs)
        return self.__normalize_engine_output(text, **kwargs)
