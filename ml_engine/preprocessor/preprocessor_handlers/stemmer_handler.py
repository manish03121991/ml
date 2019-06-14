from .base_handler import BaseTextProcessing
from .utils import remove_common_item_from_lists
from ml_engine.config import ml_logger


class StemmerHandler(BaseTextProcessing):
    """
    This is a stemming handler class, which uses Stemmer class to convert tokens to its
    root.Stemmer object has been set in base processing class and used in this handler.
    After converting to its root, it would update the kwargs dict for further processing.

    if flag_stem is set,
      and if flag_tokenize is not set,
         then first convert text_to_process into tokens and then do stemming
      and if flag_tokenize is set,
         then convert tokens to its root i.e stemming.

    if flag_stem_without_stop_word is set,
       remove all the stop words from stem_tokens

    Output:
        tokens = ['ate','eating']
      After stemming i.e flag_stem is set,
        kwargs['stem_tokens'] = ['ate','eat']

    """

    def convert_tokens_to_stem(self, tokens):
        return self.stemmer.stem_tokens(tokens)

    def stemmer_handling(self, text_to_process, tokens=None):
        """
        Note: tokens would be None if flag_tokenize is not set else pass tokens in args,
              text_to_process is the text which we need to process
        """
        stem_tokens = []
        stem_without_stop_words = []
        stemmer_dict = dict()

        if self.flag_stem:
            if not self.flag_tokenize:
                stem_tokens = self.convert_tokens_to_stem(
                    self.tokenizer.tokenizer(text_to_process))
            else:
                stem_tokens = self.convert_tokens_to_stem(
                    tokens
                )
        if self.flag_stem_without_stop_words and stem_tokens:
            stem_without_stop_words = remove_common_item_from_lists(stem_tokens, self.stop_words_list)
        elif self.flag_stem_without_stop_words and not stem_tokens:
            stem_without_stop_words = remove_common_item_from_lists(
                self.convert_tokens_to_stem(
                    self.tokenizer.tokenizer(text_to_process)), self.stop_words_list)

        stemmer_dict['stem_tokens'] = stem_tokens
        stemmer_dict['stem_without_stop_words'] = stem_without_stop_words
        return stemmer_dict

    def _handle_processing(self, **kwargs):
        stemmer_dict = self.stemmer_handling(
            kwargs['text_to_process'], kwargs['tokens']
        )
        kwargs.update(stemmer_dict)

        ml_logger.debug("Stemmer handler ==>> stem_tokens:{} and stem_without_stop_words:{}".format(
            kwargs['stem_tokens'], kwargs['stem_without_stop_words']
        ))
        return kwargs
