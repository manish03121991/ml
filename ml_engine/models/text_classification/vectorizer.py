from sklearn.feature_extraction import text

COUNT_VECTORIZER = "COUNT_VECTORIZER"


class Vectorizer(object):
    """
    This class would convert a collection of text documents to a matrix of token counts.
    The input parameter such as analyzer,preprocessing to the vectorizer can be set using
    kwargs dict.

    Attributes:
        selected_vectorizer: vectorizer which is used to convert the text docs into vector,
                             it can be count_vectorizer or hashing_vectorizer depending on
                             application requirement.By default, it is set to count_vectorizer
        vectorizer: it has object of selected vectorizer class
        vectorizer_dict: dict used for handling multiple vectorizer classes

    Example:
        X = ['God is love', 'OpenGL on the GPU is fast']
        vector = Vectorizer(analyzer='some__callable__function')
        vect = vector.fit_transform(X)
        vect.toarray()
    output:
         ['fast', 'god', 'gpu', 'is', 'love', 'on', 'opengl', 'the']
         [[  0      1      0     1       1      0      0        0]
          [  1      0      1     1       0      1      1        1]]

    """

    def __init__(self, selected_vectorizer=COUNT_VECTORIZER, **kwargs):
        self.selected_vectorizer = selected_vectorizer
        self.vectorizer = None
        self.vectorizer_dict = {
            COUNT_VECTORIZER: self._vectorizer
        }
        self.vectorizer_dict[self.selected_vectorizer](**kwargs)

    def _vectorizer(self, analyzer=None):
        self.vectorizer = text.CountVectorizer(analyzer=analyzer)

    def fit_transform(self, raw_documents, y=None):
        vector = self.vectorizer.fit_transform(raw_documents, y)
        return vector

    def transform(self, raw_documents):
        return self.vectorizer.transform(raw_documents)
