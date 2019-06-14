import base64
import os
from functools import wraps
from ml_engine.config import MODEL_FILE_CAPTURE
from django.conf import settings
from sklearn.externals import joblib
from ml_engine.api.api_exceptions import AuthenticationFailed, NotAuthenticated, ModelException, BadRequestData
from ml_engine.api.api_exceptions import TrainingErrorException
from ml_engine.config import MODEL_FILES_PATH
from ml_engine.config import model_logger


def http_basic_auth(func):
    """Basic http auth"""

    @wraps(func)
    def _decorator(request, *args, **kwargs):
        if 'HTTP_AUTHORIZATION' in request.META:
            authmeth, auth = request.META['HTTP_AUTHORIZATION'].split(' ', 1)
            if authmeth.lower() == 'basic':
                auth = base64.b64decode(auth).decode('utf-8')
                username, password = auth.split(':', 1)
                if (username != settings.ML_ENGINE['USERNAME'] or
                            password != settings.ML_ENGINE['PASSWORD']):
                    raise AuthenticationFailed
        else:
            raise NotAuthenticated
        return func(request, *args, **kwargs)

    return _decorator

def update_model_training_s3_file(training_filename, classifier_obj=None, flag_group_delete=False):
    """
    This function would update the model file on s3,
    Logic:
        It would first check if file exists for training,
            if True:
                then first remove the file and upload new training file on s3
            else:
                then upload new model training file on s3
    :param classifier: object of the classifier used as classification
    :param training_filename: filename used for model training
    :return: None
    """
    classifier_obj.training_filename = training_filename
    filepath = os.path.join(MODEL_FILES_PATH, training_filename)
    joblib.dump(classifier_obj, filename=filepath, compress=('gzip', 3))



def prepare_data_for_training(file_path):
    training_data = list()
    import csv
    with open(file_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            print(row)
            phrase = row[2]
            all_phrases = phrase.split("@@")
            intent_name = row[0]
            intent_id = row[1]
            target_value = 'INTENT' + '-' + intent_name + '-' + str(intent_id)
            for phrase in all_phrases:
                training_data.append({'class': target_value, 'sentence': phrase})
    return training_data


def validate_group_model(group_intent):
    """
    This function would validate the group for training as per the following logic
           if there is one intent mapped to group, no training would occured
           if group type is sequenced, no training required
    :param group_intent: object of group intent relationship table
    :return: True else raise exception
    """
    # model needs atleast two classes for classification
    target_values_count = check_for_number_of_targets(group_intent)
    if target_values_count <= 1:
        update_model_training_s3_file(get_group_model_training_filename(group_intent.group_id), flag_group_delete=True)
        model_logger.debug(
            "The number of intents for group training must be more than one,got :%s" % target_values_count)
        return False
    if group_intent.sequence_type == '2':
        raise ModelException("== Group type is sequenced, no need for group training ==")
    return True


def load_model(filename):
    """
    This function would take the filename and download that file from s3,
    and then using load function it would load the model in memory.

    :param filename: Name of the file which was used for model pickling
    :return: object of the model
    """
    model_logger.debug("== model loading is started ==")
    filepath = os.path.join(MODEL_FILES_PATH, filename)

    try:
        model_logger.debug("== model file:%s is downloading from s3 ==" % filename)
        s3_client.download_file(settings.AWS_STORAGE_BUCKET_NAME, filepath, filepath)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            model_logger.debug("== model file on s3 not found while loading in memory")
            raise ModelException("Model file on s3 not found while loading in memory")

    model_logger.debug("== model file:%s is loading in memory ==" % filename)
    clf = joblib.load(filepath)

    os.remove(filepath)
    return clf


def get_group_model_training_filename(group):
    """
    This funtion would give the training filename on the basis of group name and id
    :param group_intent: object of group class
    :return: training filename
    """
    return MODEL_FILE_CAPTURE + '-' + group.name + '-' + str(group.id) \
           + '.pkl' + '.gz'


def check_for_number_of_targets(group_intent):
    """
    This function would check for the number of targets that model
    is going to classify
    :param group_intent: group intent model object
    :return: count of target values
    """
    intent_ids = group_intent.intent_ids
    intents = Intent.objects.filter(id__in=intent_ids)
    count = 0
    for intent in intents:
        if intent.phrases:
            count += 1
    return count


def get_model_threshold(group_id):
    """
    Get threshold value of group id
    :param group_id:
    :return:
    """
    try:
        group = Group.objects.get(id=group_id)
        return group.threshold_value
    except Group.DoesNotExist:
        raise BadRequestData("Group id is invalid")
