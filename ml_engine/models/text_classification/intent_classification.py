import os
from sklearn.externals import joblib
from .model import TextClassificationModel
from ml_engine.preprocessor.text_processor_engine import TextProcessorEngine
from .vectorizer import Vectorizer
from .tfidf_transformer import Transformer
from sklearn.metrics import accuracy_score
from ml_engine.config import MODEL_FILES_PATH
from ml_engine.config import model_logger
from ml_engine.models.text_classification.helpers import check_for_vector_similarity
from ml_engine.helpers import get_model_threshold


class IntentClassification(object):
    """
    Intent classification is a process of classifying the intents on the basis of sentences provided.
    This class has been used for text classification model training and tuning, we need to configure
    the classifier from this class which would do all the basic steps of training.

    attributes:
        model: it is an object of model classifier used for text classification purpose
        train_set: it is a list of dict having a certain format as [{'class':'I1','sentence':"S1"},{....},...]
                          used for model training
        X_train: it is a list consists of data used for model training
        X_test: it is a list consists of data used for model testing
        y_train: it is a list of target values for model training
        y_test: it is a list of target values for model testing
        X_train_tfidf: It is a sparse matrix of type numpy class and having tf-idf frequencies of features,used
                       for training purpose
        X_test_tfidf: It is a sparse matrix of type numpy class and having tf-idf frequencies of features,used
                       for testing purpose
        text_vectorizer: object of vectorizer used for text classification
        tfidf_transformer: object of tfidf_tranformer used for text classification

    Example:
        train_set = [{"class":"greeting", "sentence":"how are you?"},
            {"class":"greeting", "sentence":"how is your day?"},{"class":"greeting", "sentence":"good day"},
            {"class":"greeting", "sentence":"how is it going today?"},{"class":"goodbye", "sentence":"havea nice day"},
            {"class":"goodbye", "sentence":"see you later"},{"class":"goodbye", "sentence":"have a nice day"},
            {"class":"goodbye", "sentence":"talk to you soon"},{"class":"sandwich", "sentence":"make me a sandwich"},
            {"class":"sandwich", "sentence":"can you make a sandwich?"},{"class":"sandwich", "sentence":"having a sandwich today?"},
            {"class":"sandwich", "sentence":"what's for lunch?"},{"class":"sandwich","sentence":"make me some lunch?"},
            {"class":"sandwich","sentence":"sudo make me a sandwich"},{"class":"greeting","sentence":"how are you doing today?"},
            {"class":"goodbye","sentence":"talk to you tomorrow"},{"class":"greeting","sentence":"who are you?"}]

        model = IntentClassification(data_to_classify)
        model.configure_classifier()

      check for model training accuracy:
          model.model_training_accuracy()

      check for model testing accuracy:
          model.model_testing_accuracy([{'class':'....','sentence':'....'},{....}.....])

      check for model prediction:
          raw_documents = ["Hey,how are you?",bye, take care""]
          model.model_predict(raw_documents)

    """

    def __init__(self, train_set):
        self.model = None
        self.train_set = train_set

        self.X_train = None
        self.y_train = None

        self.X_train_vectors = None
        self.X_train_tfidf = None

        self.text_vectorizer = None
        self.tfidf_transformer = None
        self.training_filename = None

    def _preprocess_text(self, text):
        """
        This function would use the text preprocessing engine to process/clean the data for
        text classification.After passing data from engine,it would normalize the data in the
        format which is  suitable for model.

        :param text: It is the sentence which is used for classification
        :return: normalization of text into list consists of unigrams, bigrams and trigrams
                 and also remove the stop words.

        Example:
            text = "How is it going today?"
            stopwords: how,is
        output:
            ['it','go','today','is it','it go','go today','how is it','is it go','it go today']
        """
        normalize_text = list()
        text_engine = TextProcessorEngine()
        processed_data = text_engine.text_processor(text, flag_punctuation_removal=True,
                                                    stem_unigram=True, flag_lemma=False,
                                                    stem_bigram=True, stem_trigram=True,
                                                    flag_unigram_without_stop_words=True,
                                                    flag_bigram_without_stop_words=True,
                                                    flag_trigram_without_stop_words=True)
        normalize_text.extend(processed_data['unigrams'] + processed_data['bigrams'] + \
                              processed_data['trigrams'])
        return normalize_text

    def configure_classifier(self, C=1.0, gamma='auto', kernel='linear', probability=True, random_state=0):
        """
        This function is used to configure classifier by feeding different paramateric
        values such C,kernel,gamma for svm and also do all the transformation and training
        for text classification model.

        :param C: penalty value used for controlling regularization in model

        :param kernel: kernel used in svm for dimensional transformations

        :param probability: Whether to enable probability estimates. This must be enabled prior
                                to calling `fit`, and will slow down that method.

        :param random_state: int, RandomState instance or None, optional (default=None)
                             The seed of the pseudo random number generator to use when shuffling
                             the data.  If int, random_state is the seed used by the random number
                             generator; If RandomState instance, random_state is the random number
                             generator; If None, the random number generator is the RandomState
                             instance used by `np.random`.

        :return: object of the classifier
        """

        model_logger.debug("== Configuring group model ==")
        self.model = TextClassificationModel(C=C, kernel=kernel, gamma=gamma, probability=probability,
                                             random_state=random_state)

        self.X_train, self.y_train = self._get_training_data()

        self.text_vectorizer = Vectorizer(analyzer=self._preprocess_text)
        self.X_train_vectors = self.text_vectorizer.fit_transform(self.X_train)

        self.tfidf_transformer = Transformer()
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(self.X_train_vectors)
        model_logger.debug("== conversion of training data into tf-idf completed ==")

        self._model_training(self.X_train_tfidf, self.y_train)
        model_logger.debug("== model is fitted with training data ==")

    def _get_training_data(self):
        """
        This function would spilt the data for training and testing purpose of model.
        :return: tuple of (X_train,y_train)
        """
        X = list()
        y = list()
        for data in self.train_set:
            X.append(data['sentence'])
            y.append(data['class'])
        return X, y

    def _model_training(self, X, y):
        """
        This fundtion is used for model fitting by using X_train and y_train data.
        :param X: sparse matrix of class numpy used for training
        :param y: list of categories to be classified
        """
        model_logger.debug("== fitting model with training data ==")
        self.model.fit_model(X, y)

    def model_training_accuracy(self):
        """
        This would give the training score of model which was fitted for text classification
        :return: score float value
        """
        score = self.model.model_accuracy(self.X_train_tfidf, self.y_train)
        model_logger.debug("== model training accuracy:%s ==" % score)
        return score

    def model_testing_accuracy(self, raw_documents):
        """
        This would give the testing score of fitted model,that how model would behave
        on unseen data
        :param: raw_documents: list of docs to be tested for defining model accuracy
        :return: score float value
        """
        model_logger.debug("== model is fitting for testing accuracy ==")
        X_test = list()
        y_test = list()
        for doc in raw_documents:
            X_test.append(doc['sentence'])
            y_test.append(doc['class'])
        model_logger.debug("== model is fitted with testing data:%s ==" % X_test)
        model_logger.debug("== target set of model for testing:%s ==" % y_test)

        X_test_tfidf = self.tfidf_transformer.transform(self.text_vectorizer.transform(X_test))

        y_pred = self.model.predict_model(X_test_tfidf)
        model_logger.debug("== model has predicted the testing data in classes as %s" % list(zip(X_test, y_pred)))
        print("== model has predicted the testing data in classes as %s" % list(zip(X_test, y_pred)))

        score = accuracy_score(y_test, y_pred)
        model_logger.debug("==model testing accuracy:%s ==" % score)
        return score

    def classify(self, raw_document, active_trigger_intent_status):
        """
        This function would classify the raw documents to be categorized in a particular class.
        :param raw_document: single text doc to be predicted
        :return: list of tuple [(raw_document,predicted_class),(.....).....]
        """
        model_logger.debug("== classifying raw document in some target values ==")
        model_logger.debug("== list of documents to be classified:%s ==" % raw_document)

        classified_docs = list()
        if not isinstance(raw_document, list):
            raw_document = [raw_document]
        model_logger.debug("==raw document for classification %s" % raw_document)

        # transform raw document in some vector form to analyse on plane mathematically
        document_vector = self.text_vectorizer.transform(raw_document)
        text_tfidf = self.tfidf_transformer.transform(document_vector)

        model_target_values = self.model.model.classes_.tolist()
        model_logger.debug("== model target classes:%s ==" % model_target_values)

        y_pred = self.model.predict_model(text_tfidf)
        model_logger.debug("== Before applying threshold,model has classified doc in %s" % y_pred)
        # check if document vector is sufficiently similar to predicted class
        group_id = self.training_filename.split('-')[-1].split('.')[0]
        threshold_value = get_model_threshold(group_id)
        if check_for_vector_similarity(
                y_pred, document_vector, self.X_train_vectors, self.y_train, threshold_value, raw_document, group_id, active_trigger_intent_status=active_trigger_intent_status,
  ):
            classified_docs = list(zip(raw_document, list(y_pred)))
            model_logger.debug("== model has classified doc in class %s" % classified_docs)
            return classified_docs

        model_logger.debug("== model has classified doc in class %s" % classified_docs)
        return classified_docs
