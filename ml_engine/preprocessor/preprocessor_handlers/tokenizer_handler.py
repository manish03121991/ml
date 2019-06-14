from .base_handler import BaseTextProcessing
from .utils import remove_common_item_from_lists
from ml_engine.config import ml_logger


class TokenizerHandler(BaseTextProcessing):
    """
    Tokenizer class would tokenize the text which is derived from previous
    handler, by default It would use word tokenizer of NLTK library.

    This handler would also take under considerations of stop words.

    Output:
        if flag_tokenize is set,
            text = "Hey, How are you dude?"
            tokens = ["Hey",",","How","are","you","dude","?"]
        else:tokens=[]
        if flag_tokenize_without_stop_words is set,
            then output the tokens without stopwords
    """

    def text_tokenize(self, text_to_process):
        tokens = self.tokenizer.tokenize(text_to_process)
        return tokens

    def tokenization_handling(self, text_to_process):
        tokens = []
        tokenize_without_stop_words = []
        tokenization_dict = dict()

        # check if flag tokenize is set, then convert it into tokens
        if self.flag_tokenize:
            tokens = self.text_tokenize(text_to_process)
        # check for stop words during tokenization of input string
        if self.flag_tokenize_without_stop_words and tokens:
            tokenize_without_stop_words = remove_common_item_from_lists(tokens, self.stop_words_list)
        elif self.flag_tokenize_without_stop_words and not tokens:
            tokenize_without_stop_words = remove_common_item_from_lists(self.text_tokenize(text_to_process),
                                                                        self.stop_words_list)
        tokenization_dict['tokens'] = tokens
        tokenization_dict['tokenize_without_stop_words'] = tokenize_without_stop_words
        return tokenization_dict

    def _handle_processing(self, **kwargs):
        tokenization_dict = self.tokenization_handling(kwargs['text_to_process'])
        kwargs.update(tokenization_dict)

        ml_logger.debug("Tokenization handler ==>> takens:{} and tokenize_without_stop_words:{}".format(
            kwargs['tokens'], kwargs['tokenize_without_stop_words']
        ))
        return kwargs
