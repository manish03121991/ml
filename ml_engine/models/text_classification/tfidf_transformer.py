from sklearn.feature_extraction import text

TFIDF_TRANSFORMER = 'TFIDF_TRANSFORMER'


class Transformer(object):
    """
    This class would transform a count matrix to a normalized tf or tf-idf representation.

    Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency.
    This is a common term weighting scheme in information retrieval, that has also found good
    use in document classification.

    The formula that is used to compute the tf-idf of term t is
             tf-idf(d, t) = tf(t) * idf(d, t)

    Example:
          tfidf_transform = TfidfTransformer()
          tfidf_transform.fit_transform(count__vector__object)
    """

    def __init__(self, selected_transformer=TFIDF_TRANSFORMER):
        self.selected_transformer = selected_transformer
        self.transformer = None
        self.transformer_dict = {
            TFIDF_TRANSFORMER: self._transformer
        }
        self.transformer_dict[selected_transformer]()

    def _transformer(self):
        self.transformer = text.TfidfTransformer()

    def fit_transform(self, X, y=None):
        transform = self.transformer.fit_transform(X, y)
        return transform

    def transform(self, X, copy=True):
        return self.transformer.transform(X, copy=copy)
