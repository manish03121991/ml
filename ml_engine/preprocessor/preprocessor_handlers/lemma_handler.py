from .base_handler import BaseTextProcessing
from .constants import lemmatizer
from .utils import remove_common_item_from_lists
from ml_engine.config import ml_logger


class LemmatizerHandler(BaseTextProcessing):
    """
    This handler is used to find the lemma of the tokens using Lemmatizer class.
    Lemma object has been set in base processing class and used in this handler
    After converting to its lemma, it would update the kwargs dict for further processing.

    if flag_lemma is set,
      and if flag_tokenize is not set,
         then first convert text_to_process into tokens and then do lemmatization
      and if flag_tokenize is set,
         then convert tokens to its lemma.

    if flag_lemma_without_stop_word is set,
        remove all the stop words from lemma_tokens

    output:
        tokens = ['plays','shopping','shop']
      After lemmatization,
        kwargs['lemma_tokens'] = ['play','shopping','shop']

    """

    def convert_tokens_to_lemma(self, tokens):
        return lemmatizer.lemma_tokens(tokens)

    def lemmatization_handling(self, text_to_process, tokens=None):
        """
        Note: tokens would be None if flag_tokenize is not set else pass tokens in args,
              text_to_process is the text which we need to process
        """
        lemma_tokens = []
        lemma_without_stop_words = []
        lemma_dict = dict()

        if self.flag_lemma:
            if not self.flag_tokenize:
                lemma_tokens = self.convert_tokens_to_lemma(
                    self.tokenizer.tokenizer(text_to_process)
                )
            else:
                lemma_tokens = self.convert_tokens_to_lemma(
                    tokens
                )
        if self.flag_lemma_without_stop_words and lemma_tokens:
            lemma_without_stop_words = remove_common_item_from_lists(lemma_tokens, self.stop_words_list)
        elif self.flag_lemma_without_stop_words and not lemma_tokens:
            lemma_without_stop_words = remove_common_item_from_lists(
                self.convert_tokens_to_lemma(
                    self.tokenizer.tokenizer(text_to_process)), self.stop_words_list)

        lemma_dict['lemma_tokens'] = lemma_tokens
        lemma_dict['lemma_without_stop_words'] = lemma_without_stop_words

        return lemma_dict

    def _handle_processing(self, **kwargs):
        lemma_dict = self.lemmatization_handling(kwargs['text_to_process'],
                                                 kwargs['tokens'])
        kwargs.update(lemma_dict)

        ml_logger.debug("lemmatization handler ==>> lemma_tokens:{} and lemma_without_stop_words:{}".format(
            kwargs['lemma_tokens'], kwargs['lemma_without_stop_words']
        ))
        return kwargs
