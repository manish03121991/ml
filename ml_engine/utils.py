import datetime
from ml_engine.api import api_exceptions
from ml_engine.config import model_logger
from django.conf import settings


def format_query_training_data(object_list):
    """
    To convert the  query training data to a particular format
    :param object_list: List of all the user query training objects
    :return: list consisting of all the attributes of the objects
    """
    return [
        {"user_query_id": str(obj.id), "group_id": str(obj.group.id), "group_name": str(obj.group.name),
         "intent_id": [str(obj.intent.id) if obj.intent else '' if obj.flag_matched else ''][0],
         "intent_name": [str(obj.intent.name) if obj.intent else '' if obj.flag_matched else ''][0],
         "query": obj.query,
         "flag_matched": obj.flag_matched,
         "flag_mapped": obj.flag_mapped,
         "last_updated": str(obj.last_updated),
         "time_created": str(obj.time_created.date())
         } for obj in object_list]


def format_training_stats_data(object_list, query_training_data):
    """
    To convert the training stats data to a particular format
    :param object_list: List of all the user query training objects
    :param query_training_data: ?write?
    :return: list consisting of all the attributes of the objects
    """
    group_data = [
        {"group_id": str(obj.group.id), "group_name": str(obj.group.name), "unmatched_count": obj.unmatched_count,
         "matched_count": obj.matched_count, "mapped_count": obj.mapped_count,
         "time_created": str(obj.time_created.date())}
        for obj in object_list if obj.matched_count or obj.unmatched_count or obj.mapped_count]
    query_training_data.extend(group_data)
    return query_training_data


def get_date():
    """
    Output date with time params to 0
    :return: todays date
    """
    d = datetime.datetime.today()
    today_date = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return today_date


def extract_associated_group_intents(group_id):
    group_intent = GroupIntentRelationship.objects.get(group_id=ObjectId(group_id))
    intent_ids_list = group_intent.intent_ids
    intent_id_object_list = Intent.objects.filter(id__in=intent_ids_list)
    intent_data_list = [{"id": str(obj.id), "name": obj.name} for obj in intent_id_object_list]
    return intent_data_list


def store_model_results(group, text, model_prediction):
    """
    This function would store the results of model prediction in database, which helps us in training
    the model from dashboard.

    Two tables such as TrainingStats and UserQueryTraining are used for storing all the model predictions.
        Trainingstats table is used for storing summary per group in a day
        UserQueryTraining table is used for storing all matched and unmatched predictions of model.

    :param group: object of Group table
    :param text: input string from user
    :param model_prediction: prediction of model if any else empty list
    :return: None

    """
    # Note: update and modify, both methods work same in case of single document update.
    # in case of multiple doc updates, there might be case in update method that may have modified
    # the document between your update and the document retrieval,but findAndModify is safe in this case.

    stats = TrainingStats.objects(group=group, time_created=get_date()).modify(upsert=True, new=True,
                                                                               set__group=group,
                                                                               set__time_created=get_date())

    # if model hasn't predicted the text into class, then it would increment the counter of unmatched query
    if not model_prediction:
        stats.unmatched_count += 1
        stats.save()
        UserQueryTraining.objects.create(group=group, query=text, flag_matched=False)
        return
    # if model has predicted the query in some class, then would increase the counter of matched query
    for doc in model_prediction:
        predicted_intent = doc[1].split('-')[-1]
        try:
            intent = Intent.objects.get(id=predicted_intent)
            UserQueryTraining.objects.create(group=group, query=doc[0], flag_matched=True, intent=intent)
        except Intent.DoesNotExist:
            model_logger.debug("== Intent Id object doesn't exist %s==" % predicted_intent)
            raise api_exceptions.BadRequestData("Intent id :%s is invalid" % predicted_intent)
        stats.matched_count += 1
        stats.save()
    return


def update_training_status_in_cache(creator_id, group_id, flag_completed=False):
    """
    This function would set a hash map in redis, In hash there would be a key-value pair
    which contains the status of every group that comes for training.

    Redis cache Hash key;
            Hash Name: __app_name__:__logical-functionality-name__:__unique-identifier__
            Key Name: group-id:__id of group__
            Value: True/False

    :param creator_id: Id of user
    :param group_id: Id of group for training
    :return: None
    """
    set_hash_maps_in_cache(
        system=settings.CACHES['ml-cache']['CACHE_NAME'],
        hash_name='ml_engine:training-status:user:%s' % creator_id,
        data={'group-id:%s' % group_id: flag_completed},
        expiry_time=settings.TRAINING_STATUS_EXPIRATION
    )


def get_training_status_from_cache(creator_id):
    """
    This function would get the status of model training specific
    to creator id

    :param creator_id: Id of user
    """
    hash_value = get_hash_from_cache(
        system=settings.CACHES['ml-cache']['CACHE_NAME'],
        hash_name='ml_engine:training-status:user:%s' % creator_id
    )
    return hash_value


def delete_training_status_from_cache(creator_id):
    delete_hash_from_cache(
        system=settings.CACHES['ml-cache']['CACHE_NAME'],
        hash_name='ml_engine:training-status:user:%s' % creator_id
    )
    return True
