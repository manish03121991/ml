import nltk

WORDNET_LEMMATIZER = 'WORDNET_LEMMATIZER'


class Lemmatizer(object):
    """
    This class would detect the lemma of a word,

    Lemmatization uses context and part of speech to determine the inflected
    form of the word and applies different normalization rules for each part
    of speech to get the root word.

    By default, wordnet lemmatizer is used for lemmatization.

    Attributes:
        lemma_selected: this is string contains the value of selected lemma
        lemmatizer: it has the object of selected lemmatizer class
        lemmatizer_dict: it would have all the lemmatization class used

    Output:
        tokens: ['ate','running','shopping','shopped','shop','plays']
      After lemmatization,
        lemma_tokens: ['ate','running','shopping','shopped','shop','play']
    """

    def __init__(self, lemma_selected=WORDNET_LEMMATIZER):
        self.lemma_selected = WORDNET_LEMMATIZER
        self.lemmatizer = None
        self.lemmatizer_dict = {
            WORDNET_LEMMATIZER: self.__wordnet_lemmatizer
        }
        self.lemmatizer_dict[lemma_selected]()

    def __wordnet_lemmatizer(self):
        self.lemmatizer = nltk.WordNetLemmatizer()

    def lemma_word(self, word):
        return self.lemmatizer.lemmatize(word)

    def lemma_tokens(self, tokens):
        lemma_tokens_list = []
        for token in tokens:
            lemma_tokens_list.append(
                self.lemma_word(token.lower().strip())
            )
        return lemma_tokens_list

    def get_lemmatizer(self):
        return self.lemmatizer
