import json
from ml_engine.api.api_exceptions import api_exception_handler
from ml_engine.helpers import http_basic_auth
from django.views.decorators.http import require_http_methods
from chatbot.models import BotGroupRelationship
from ml_engine.api.schema import GroupThresholdGETSchema, GroupThresholdPOSTSchema
from ml_engine.config import model_logger
from ml_engine.api import api_exceptions
from django.http import HttpResponse
from group.models import Group, GroupIntentRelationship


@api_exception_handler
@http_basic_auth
@require_http_methods(['GET'])
def get_group_threshold(request):
    """
    This API would use on dashboard for showing the threshold values for group model
    which are random types and are associated with bots.

    API Endpoint:
        http://192.168.1.240:8000/get-group-threshold/?bot_id=59ddd881f3b4a93f30a834f1&creator_id=2

    Response:
        {"data": [
                    {
                        "group_name": "Cab service",
                        "sequence_type": "1",
                        "group_id": "59ddd89ff3b4a93f30a834f3",
                        "threshold": 0.2
                        }

                    ],
        "status_code": 200
        }

    :param request: http request
    :return: json response
    """
    schema_class = GroupThresholdGETSchema()
    schema, errors = schema_class.load(request.GET)
    if errors:
        model_logger.debug("== Group Threshold GET request schema error:%s ==" % errors)
        raise api_exceptions.BadRequestData("Request data is not valid")
    bot_groups = BotGroupRelationship.objects.filter(bot_id=schema.get('bot_id'))
    if not bot_groups:
        return HttpResponse(json.dumps({'message': 'No bot group relation found', 'status_code': 200}), status=200)
    groups = Group.objects.filter(id__in=bot_groups[0].group_ids)
    group_intents = GroupIntentRelationship.objects.filter(group_id__in=groups).filter(sequence_type='1')
    return HttpResponse(json.dumps({'data': [
        {'sequence_type': group_intent.sequence_type, 'threshold': group_intent.group_id.threshold_value,
         'group_id': str(group_intent.group_id.id),
         'group_name': group_intent.group_id.name} for group_intent in group_intents], 'status_code': 200}), status=200)


@api_exception_handler
@http_basic_auth
@require_http_methods(['POST'])
def update_group_threshold(request):
    """
    This POST request would update the threshold value of associated group.

    API Endpoint:
        http://192.168.1.240:8000/update-group-threshold/

    request body:
            [
        {
        "group_id": "59ddd89ff3b4a93f30a834f3",
        "creator_id": 2,
        "threshold_value": 0.4,
        "bot_id": "59ddd881f3b4a93f30a834f1"
            }
        ]

    Response:
        {"status_code": 201, "message": "ML settings are updated"}

    Status_code:
        201: successful
        400: Bas request data

    :param request:
    :return:
    """
    schema_class = GroupThresholdPOSTSchema()
    schema, errors = schema_class.load(json.loads(request.body.decode('utf-8')), many=True)
    if errors:
        model_logger.debug("== Group Threshold post request schema error:%s ==" % errors)
        raise api_exceptions.BadRequestData("Request data is not valid")
    for data in schema:
        schema_class.update_threshold_value(data)
    return HttpResponse(json.dumps({'message': 'ML settings are updated', 'status_code': 201}), status=201)
