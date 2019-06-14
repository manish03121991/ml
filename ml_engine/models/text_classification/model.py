from sklearn import svm

SVM_CLASSIFIER = 'SVM_CLASSIFIER'


class TextClassificationModel(object):
    """
    Different models such as svn,naive,neural can be used for text classification that
    depends on you application accuracy result which fits best.
    This class helps us in selecting the different models and fit the models on the basis of data
    provided.

    By default, for text classification we are using svm model and that can be customizable depending
    on the accuracy results.

    Attributes:
        model_selected: model which is used for text classification
        model: it has object of selected model class
        model_dict: it is the dict which contains different model references

    """

    def __init__(self, model_selected=SVM_CLASSIFIER, **kwargs):
        self.model_selected = model_selected
        self.model = None
        self.model_dict = {
            SVM_CLASSIFIER: self._svm_classifier
        }
        self.model_dict[self.model_selected](**kwargs)

    def _svm_classifier(self, C=1.0, kernel='rbf', gamma='auto', probability=False, random_state=0):
        self.model = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability, random_state=random_state)

    def fit_model(self, X, y):
        return self.model.fit(X, y)

    def model_accuracy(self, X, y):
        return self.model.score(X, y)

    def predict_model(self, X):
        return self.model.predict(X)

    def predict_model_proba(self, X):
        return self.model.predict_proba(X)
