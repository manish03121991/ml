import nltk

WORD_TOKENIZER = 'WORD_TOKENIZER'


class Tokenizer(object):
    """
    This class helps us to convert the text into tokens, we are using word tokenize from nltk
    by default.But it is customizable and can use any other tokenizer as per our need.

    :param tokenizer_selected: tokenizer which needs to be selected for processing
    :param tokenizer_dict: Dictionary consists of keys as tokenizer selected and value is tokenizer function
    :param tokenizer: tokenizer which is selected for processing

    for example:
        token = Tokenizer()
        token.tokenize("Hey, How are you man?")
    output:
        ["Hey",",","How","are","you","man","?"]

    """

    def __init__(self, tokenizer_selected=WORD_TOKENIZER):
        """Initialize the tokenizer with selected tokenizer"""

        self.tokenizer_selected = tokenizer_selected
        self.tokenizer_dict = {
            WORD_TOKENIZER: self.__word_tokenize
        }
        self.tokenizer = None
        self.tokenizer_dict[tokenizer_selected]()

    def __word_tokenize(self):
        """Initializes word tokenizer"""
        self.tokenizer = nltk.word_tokenize

    def get_tokenizer(self):
        """Get selected tokenizer"""
        return self.tokenizer

    def tokenize(self, text):
        """Tokenize the text into tokens"""
        if self.tokenizer_selected == WORD_TOKENIZER:
            return self.tokenizer(text)


tokenizer = Tokenizer()