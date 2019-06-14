from .base_handler import BaseTextProcessing
from ml_engine.config import ml_logger


class NgramHandler(BaseTextProcessing):
    """
    This handler is used to convert token/stem_tokens into ngram list using the
    Ngram class.

    Attributes:
        gram: it is the number of grams to be used to convert tokens.
        flag_stem_gram: it is True if need to convert stem tokens into ngrams else False then
        tokens without stem would convert.
        flag_stop_word_list: True if need to remove stop words from ngram else False
    """

    def __get_ngram(self, gram, flag_stem_gram, flag_stop_word_list, tokens, stem_tokens=None):
        gram_tokens = stem_tokens if flag_stem_gram else tokens
        stop_word_list = self.stop_words_list if flag_stop_word_list else []
        return self.ngram.ngram_data(gram, gram_tokens, stop_word_list)

    def ngram_handling(self, tokens, stem_tokens=None):
        """
        Note: stem_tokens would be None if flag_stem is not set else pass stem_tokens in args,
              tokens which we need to process if stem_ngram flag isn't set
        """
        ngram_dict = dict()
        unigrams = self.__get_ngram(1, self.stem_unigram, self.flag_unigram_without_stop_words,
                                    tokens, stem_tokens)
        bigrams = self.__get_ngram(2, self.stem_bigram, self.flag_bigram_without_stop_words,
                                   tokens, stem_tokens)
        trigrams = self.__get_ngram(3, self.stem_trigram, self.flag_trigram_without_stop_words,
                                    tokens, stem_tokens)
        ngram_dict.update({'unigrams': unigrams, 'bigrams': bigrams, 'trigrams': trigrams})
        return ngram_dict

    def _handle_processing(self, **kwargs):
        ngram_dict = self.ngram_handling(kwargs['tokens'], kwargs['stem_tokens'])
        kwargs.update(ngram_dict)

        ml_logger.debug("Ngram handler ==>> unigram:{} == bigram:{} == trigram:{}".format(
            kwargs['unigrams'], kwargs['bigrams'], kwargs['trigrams']
        ))
        return kwargs
