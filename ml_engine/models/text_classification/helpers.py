import numpy as np
from ml_engine.config import model_logger
from ml_engine.tasks import store_model_prediction_data
from django.conf import settings
import pandas as pd


def check_for_vector_similarity(y_pred, raw_document_vector, X_train_vectors, Y_train, threshold_value, query = None, group_id = None,active_trigger_intent_status=None):
    """
     This function is used to find the similarity between two vectors.

     Logic:
        The below logic would find the similarity between raw document vector (which came for model prediction)
        and X_train_vectors of class y_pred which model has predicted by finding the union and intersection between
        the vectors.
        
        The basic idea behind this is to find the ratio between intersection and union of feature vectors and raw
        document which act as a similarity score.

        if similarity_Score > threshold_value:
            then it would return True else False


    :param y_pred: class that model has predicted
    :param raw_document_vector: vector that came for model prediction
    :param X_train_vectors: value of training vectors
    :param Y_train: all target classes
    :return: True/False boolean depending on the threshold value
    """

    similarity_score = list()
    y_pred = y_pred.tolist()[0]
    # find non zero vector which helps us in finding the matching features in raw document
    non_zero_raw_document_vector = [(index, x) for index, x in np.ndenumerate(raw_document_vector.toarray()) if x > 0]
    raw_document_vector_list = [doc[1] for doc in non_zero_raw_document_vector]

    model_logger.debug('== Threshold value for model %s ==' % threshold_value)
    prediction_logging_payload = {
        'group_id':group_id,
        'y_pred':y_pred,
        'threshold_value':threshold_value,
        'active_trigger_intent_status':active_trigger_intent_status,
        'query':query[0]
    }
    if not raw_document_vector_list:
        async_model_prediction_logging(**prediction_logging_payload)
        return False

    empty_data_frame = pd.DataFrame()
    empty_data_frame['y_train'] = Y_train
    index_list = empty_data_frame.index[empty_data_frame['y_train'] == y_pred].tolist()
    feature_vectors_new = X_train_vectors.toarray()[index_list,:].tolist()

    # # all feature vectors of target class which model has predicted
    # feature_vectors = [mapping[0] for mapping in zip(X_train_vectors.toarray().tolist(), Y_train) if
    #                    mapping[1] == y_pred]

    for feature_vector in feature_vectors_new:
        # find intersection feature between feature vector and non zero raw document
        intersection_feature = list(map(lambda x: feature_vector[x[0][1]] if feature_vector[x[0][1]] == 0 else x[1],
                                        non_zero_raw_document_vector))
        # take intersection between raw document and intersection feature
        intersection = len(
            [i for i, j in zip(intersection_feature, raw_document_vector_list) if i == j])
        if not intersection:
            # nothing interesting here. There is nothing similar in this feature and given user input
            continue

        # find lenght of feature vector which act as union in our case
        union = sum(x > 0 for x in feature_vector)
        similarity_score.append(intersection / union)

    model_logger.debug("== similarity score with vectors of class %s ==" % similarity_score)
    prediction_logging_payload['similarity_score'] = similarity_score
    async_model_prediction_logging(**prediction_logging_payload)

    if similarity_score and max(similarity_score) > threshold_value:
        return True
    return False


def async_model_prediction_logging(
        group_id,
        y_pred,
        threshold_value,
        query,
        active_trigger_intent_status,
        similarity_score = []
):
    if group_id and settings.MODEL_PREDICTION_STORE:
        predicted_intent_id = y_pred.split('-')[-1].split('.')[0]
        ml_engine_payload = {
            'group_id':group_id,
            'threshold_value':threshold_value,
            'similarity_score':similarity_score,
            'query':query,
            'predicted_intent_id':predicted_intent_id,
            'active_trigger_intent_status':active_trigger_intent_status

        }
        model_logger.debug(ml_engine_payload)
        store_model_prediction_data.delay(**ml_engine_payload)


