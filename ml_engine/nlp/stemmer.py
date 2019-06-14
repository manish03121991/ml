import nltk

PORTER_STEMMER = 'PORTER_STEMMER'


class Stemmer(object):
    """
    Stemming class would find the root/stem of the word/tokens,
    By default we are using porter stemming (this is widely used stemmer)
    but is fully configurable according to our needs

    :param: stemmer_selected: stemmer name which we have selected for stemming
    :param: stemmer_dict: It is used to define different stemmers and called the selected stemmer
    :param: stemmer: it would have the object of selected stemmer class

    Output:
        tokens = ['eating','shopping','beautiful']
        After stemming,
            stemmed_tokens = ['eat','shop','beauti']
    """

    def __init__(self, stemmer_selected=PORTER_STEMMER):
        self.stemmer_selected = stemmer_selected
        self.stemmer = None
        self.stemmer_dict = {
            PORTER_STEMMER: self.__porter_stemmer
        }
        self.stemmer_dict[stemmer_selected]()

    def __porter_stemmer(self):
        self.stemmer = nltk.stem.PorterStemmer()

    def stem_word(self, word):
        if self.stemmer_selected == PORTER_STEMMER:
            return self.stemmer.stem(word)

    def get_stemmer(self):
        return self.stemmer

    def stem_tokens(self, tokens):
        stem_tokens_list = []
        for token in tokens:
            stem_tokens_list.append(
                self.stem_word(token.lower().strip())
            )
        return stem_tokens_list


