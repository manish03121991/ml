from group.models import GroupIntentRelationship, Group
from ml_engine.api.api_exceptions import TrainingErrorException
from ml_engine.config import model_logger
from ml_engine.models.text_classification import IntentClassification
from ml_engine.helpers import update_model_training_s3_file, get_group_model_training_filename
from ml_engine.tasks import individual_group_training
from ml_engine.utils import update_training_status_in_cache


def train_intent_classification_model(training_filename, training_data):
    """
    This function would validate the group for training and after validating, would prepare
    training data and feed it into the model classifier for training purpose and
    then upload pickled model file to s3

    :param group_intent: object of GroupIntentRelationship table
    :param training_data: data used for model training

    :return: None
    """
    classifier = IntentClassification(training_data)
    classifier.configure_classifier()
    update_model_training_s3_file(training_filename, classifier)
    return


def group_model_training():
    """
    This function would initiate the group training and apply the following logic:

      a) if (group id and intent id are both None) or (both are not None) then raised exception

      b) if only group id comes,it would find its corresponding mapped intent ids,
         prepare the data for model training and then pickle the training file to s3 server

      c) if training file already exists on s3, it would update the file by updating the model

      d) if only intent id comes,then find all the corresponding mapped group ids and then prepare
         the data for model training

      e) after preparing the data, it would load it into model and would start the training

      d) if flag_group_delete is set, it would then delete the corresponding group model from s3

      e) it would also update the group training status in redis cache

    :return: None
    """

    individual_group_training()
