import json

from django.views.decorators.http import require_http_methods
from django.http import HttpResponse

from bot.common.decorators import log_execution_time
from bot.common.metrices import MODEL_PREDICTION
from group.models import Group
from ml_engine.api.api_exceptions import api_exception_handler
from ml_engine.helpers import http_basic_auth
from ml_engine.config import model_logger, MODEL_FILE_CAPTURE
from ml_engine.api.schema import GroupModelPredictionSchema
from ml_engine.api import api_exceptions
from ml_engine.helpers import load_model
from ml_engine.utils import store_model_results


@log_execution_time(MODEL_PREDICTION)
@api_exception_handler
@http_basic_auth
@require_http_methods(['GET'])
def group_model_prediction(request):
    """
    Text data would be sent in request corresponding to group_id for model prediction.
    This Api would load the training model in memory from s3 and then call the model
    prediction on text provided in request.

    :param request: HTTP request
    :return: Document prediction response: {'data':[{'text':".....",'predicted_class':'........'}]}

    Request query param: group_id and text (both are required)

    For example:
        API endpoint:
            http://127.0.0.1:8000/group-model-prediction/?group_id=59ca4e0a0674ed70dde7dbcf&text=hey,how are you?
            response: {'data':[{'text':'hey,how are you?','predicted_class':'.......'}]}

    """
    model_logger.debug("== request data received for group model prediction:%s ==" % request.GET)

    # check for schema validation
    schema_class = GroupModelPredictionSchema()
    schema, errors = schema_class.load(request.GET)
    if errors:
        model_logger.debug("== error in request while model prediction %s==" % str(errors))
        raise api_exceptions.BadRequestData('Request data is invalid')

    try:
        group = Group.objects.get(id=schema.get('group_id'))
    except Group.DoesNotExist:
        model_logger.debug("== Group Id object doesn't exist %s==" % str(schema.get('group_id')))
        raise api_exceptions.BadRequestData("Group id is invalid")
    model_filename = MODEL_FILE_CAPTURE + '-' + group.name + '-' + str(group.id) \
                     + '.pkl' + '.gz'
    # load model in memory
    classifier = load_model(model_filename)
    # call model member function for prediction

    active_trigger_intent_status = schema.get('active_trigger_intent_status', False)
    model_prediction = classifier.classify(schema.get('text'),active_trigger_intent_status)

    # store model results in db
    store_model_results(group, schema.get('text'), model_prediction)

    # prepare data for output format
    document_prediction = [{'text': doc[0], 'predicted_class': doc[1]} for doc in model_prediction]
    return HttpResponse(
        json.dumps({'data': document_prediction, 'status_code': 200, 'message': 'model has predicted'}),
        status=200,
        content_type='application/json')
