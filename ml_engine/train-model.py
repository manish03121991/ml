from sklearn.externals import joblib
import os
MODEL_FILES_PATH = os.path.join('pkl/')

def train_intent_classification_model(training_filename, training_data):
    """
    This function would validate the group for training and after validating, would prepare
    training data and feed it into the model classifier for training purpose and
    then upload pickled model file to s3

    :param group_intent: object of GroupIntentRelationship table
    :param training_data: data used for model training

    :return: None
    """
    # import ipdb
    # ipdb.set_trace()
    from models.text_classification import IntentClassification
    classifier = IntentClassification(training_data)
    classifier.configure_classifier()
    update_model_training_s3_file(training_filename, classifier)

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
            phrase = row[2]
            all_phrases = phrase.split("@@")
            intent_name = row[0]
            intent_id = row[1]
            target_value = 'INTENT' + '-' + intent_name + '-' + str(intent_id)
            for phrase in all_phrases:
                training_data.append({'class': target_value, 'sentence': phrase})
    return training_data

def individual_group_training():
    training_filename = 'testing.pkl.gz'
    training_file_path = 'test.csv'
    # prepare data for training
    training_data = prepare_data_for_training(training_file_path)
    print(training_data)
    # train valid group for intent classification model
    train_intent_classification_model(training_filename, training_data)

individual_group_training()