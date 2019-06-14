import logging
from datetime import datetime

from ml_engine.api.api_exceptions import ModelException
from ml_engine.helpers import update_model_training_s3_file, get_group_model_training_filename, validate_group_model, \
    prepare_data_for_training
from ml_engine.utils import update_training_status_in_cache, delete_training_status_from_cache

logger = logging.getLogger(__name__)


def update_user_query_training_data(user_query_training_obj, query, new_intent):
    """
    Update the user query training data.
    """

    try:
        # update whole data
        all_obj = UserQueryTraining.objects.filter(
            intent=user_query_training_obj.intent,
            query=query)

        if user_query_training_obj.intent:
            old_intent = user_query_training_obj.intent.intent_name
        else:
            old_intent = "None"

        logger.info("#### UserQueryTraining data update started ####")
        logger.info("UserQueryTraining data: old_intent={old_intent}, new_intent={new_intent}, "
                    "query={query}".format(old_intent=old_intent,
                                           new_intent=new_intent.intent_name,
                                           query=query))

        for user_query_training_obj in all_obj:
            # update the training stats, user_query_training
            user_obj_date = user_query_training_obj.time_created.replace(hour=0, minute=0, second=0, microsecond=0)
            training_stat = TrainingStats.objects.get(group=user_query_training_obj.group, time_created=user_obj_date)

            # matched/mapped case
            if user_query_training_obj.flag_matched:
                # matched case
                if not user_query_training_obj.flag_mapped:
                    user_query_training_obj.flag_mapped = True
                    training_stat.matched_count -= 1
                    training_stat.mapped_count += 1

                user_query_training_obj.intent = new_intent
            # unmatched case
            else:
                user_query_training_obj.flag_mapped = True
                user_query_training_obj.flag_matched = True
                user_query_training_obj.intent = new_intent
                training_stat.unmatched_count -= 1
                training_stat.mapped_count += 1

            training_stat.save()
            user_query_training_obj.save()

        logger.info("Total updated count: {}".format(len(all_obj)))
        logger.info("#### UserQueryTraining data update completed ####")

    except Exception as e:
        logger.info("#### Exception in update_user_query_training_data ####")
        logger.info(str(e))


def store_model_prediction_data(**kwargs):

    try:
        similarity_score = kwargs.get('similarity_score')
        threshold_value = kwargs.get('threshold_value')
        matched_status = check_matched_status(similarity_score,threshold_value)
        participate_data, predicted_intent_phrase = prepare_participate_data(kwargs.get('group_id'),kwargs.get('predicted_intent_id'))
        kwargs['predicted_intent_phrase'] = predicted_intent_phrase
        kwargs['participate_data'] = participate_data
        kwargs['matched_status'] = matched_status
        kwargs['active_trigger_intent_status']=eval(kwargs['active_trigger_intent_status'])
        p_object = PredictionLogging(**kwargs)
        p_object.save()
    except Exception as e:
        logger.info("#### Exception in store_model_prediction_data #### {error_message}".format(error_message=str(e)))


def check_matched_status(similarity_score, threshold_value):
    if similarity_score and max(similarity_score) > threshold_value:
        return True
    return False


def prepare_participate_data(group_id, predicted_intent_id):

    all_intent_ids = GroupIntentRelationship.objects.get(group_id=ObjectId(group_id)).intent_ids
    all_intent_info = Intent.objects.filter(id__in=all_intent_ids)
    participate_data = {}
    predicted_intent_phrase = []
    for intent_info in all_intent_info:
        if intent_info.phrases:
            if str(intent_info.id) == predicted_intent_id :
                predicted_intent_phrase = intent_info.phrases
            participate_data[str(intent_info.id)] = intent_info.phrases

    return participate_data, predicted_intent_phrase

def train_intent_classification_model(training_filename, training_data):
    """
    This function would validate the group for training and after validating, would prepare
    training data and feed it into the model classifier for training purpose and
    then upload pickled model file to s3

    :param group_intent: object of GroupIntentRelationship table
    :param training_data: data used for model training

    :return: None
    """
    from ml_engine.models.text_classification import IntentClassification
    classifier = IntentClassification(training_data)
    classifier.configure_classifier()
    update_model_training_s3_file(training_filename, classifier)


def individual_group_training():
    import time
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    training_filename = timestr+'-testing.pkl.gz'
    training_file_path = 'test.csv'
    # prepare data for training
    training_data = prepare_data_for_training(training_file_path)
    # train valid group for intent classification model
    train_intent_classification_model(training_filename, training_data)





