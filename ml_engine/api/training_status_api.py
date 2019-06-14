import json
import ast
from ml_engine.config import model_logger
from .schema import TrainingStatusSchema
from ml_engine.api import api_exceptions
from ml_engine.utils import get_training_status_from_cache, delete_training_status_from_cache
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from ml_engine.helpers import http_basic_auth
from ml_engine.api.api_exceptions import api_exception_handler


@api_exception_handler
@http_basic_auth
@require_http_methods(['GET'])
def model_training_status(request):
    """
    This is a training status API which depicts the training status corresponding to
    user.

    Logic:
        When user's training started from dashboard, then it's status has been updated
        in redis cache, and that status is checked in this API using flag_completed
        boolean as True/False.

    status_code:
        200: if user training in progress or training is completed depending on flag_completed boolean
        204: if no group found in cache for training corresponding to the user

    :return: Json data as follows
        if no group found for training then,
            {'message': 'No group found for training', 'status_code': 204}

        if training status in progress:
            {'message': 'Bot Training in progress', 'status_code': 200,'flag_completed':False}

        if training status is completed:
            {'message': 'Bot Training Completed', 'status_code': 200,'flag_completed':True}

    """
    model_logger.debug("== Received request for model training status ==")
    schema_class = TrainingStatusSchema()
    schema, errors = schema_class.load(request.GET)

    if errors:
        model_logger.debug("== Group Training status API schema error:%s ==" % errors)
        raise api_exceptions.BadRequestData("Request data is not valid")

    model_logger.debug("== request data %s ==" % request.GET)
    group_wise_training_status = get_training_status_from_cache(schema.get('creator_id'))
    # check cache data if not exists
    if not group_wise_training_status:
        model_logger.debug("== No group found for training in status API ==")
        return HttpResponse(json.dumps({'message': 'No need for Group Model Training', 'status_code': 204}), status=200)

    # check bot training status
    # HACk: group_wise_training_status has been convert into python object using ast
    # TODO: Need to find a way to store proper data format in redis
    flag_completed = all([ast.literal_eval(b.decode('utf-8')) for b in group_wise_training_status.values()])
    # delete key from cache if training status is completed
    if flag_completed:
        delete_training_status_from_cache(schema.get('creator_id'))

    model_logger.debug("== Bot training status is completed ==" if flag_completed else "== Bot training in progress ==")
    return HttpResponse(
        json.dumps({'message': 'Bot Training in progress' if not flag_completed else 'Group Model Training Completed',
                    'flag_completed': flag_completed,
                    'status_code': 200}),
        status=200,
        content_type='application/json'
    )
