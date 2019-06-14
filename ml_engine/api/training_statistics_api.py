import json

import dateutil
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods

from group.models import TrainingStats
from ml_engine.api.api_exceptions import api_exception_handler
from ml_engine.config import model_logger
from ml_engine.helpers import http_basic_auth
from ml_engine.utils import format_training_stats_data
from .schema import TrainingStatsSchema


@api_exception_handler
@http_basic_auth
@require_http_methods(['GET'])
def training_stats(request):
    """
    This function takes in as group_id(list of group ids), date_from and date_to as request params and
    outputs data to be shown on training stats page.

    Sample Endpoint:
    http://127.0.0.1:8000/query-training-stats/?group_id=58ac9e63e1ad9f79a691da80,58ac9e63e1ad9f79a691da80&date_from=2017-09-28&date_to=2017-09-28

    Sample output:
    {
    "status_code": 200,
    "message": "Training Stats",
    "data": [
        {
            "group_id": "58ac9e63e1ad9f79a691da80",
            "time_created": "2017-09-28",
            "matched_count": 0,
            "unmatched_count": 0
        },
        {
            "group_id": "59a15b56a3de397e74e59070",
            "time_created": "2017-09-29",
            "matched_count": 0,
            "unmatched_count": 0
        }
    ]
}
    """
    model_logger.debug("Training stats get data")
    try:
        model_logger.debug("request data : {}".format(request.GET))
        schema_class = TrainingStatsSchema()
        schema, errors = schema_class.load(request.GET)
        if errors:
            model_logger.debug("Error in training stats request data : {error}".format(error=errors))
            return HttpResponse(
                json.dumps({'message': "failed to parse request data", "errors": errors, 'status_code': 400}),
                status=400,
                content_type='application/json')
        group_id_list = request.GET.get('group_id').split(',')
        date_from = dateutil.parser.parse(request.GET.get('date_from'), dayfirst=True)
        date_to = dateutil.parser.parse(request.GET.get('date_to'), dayfirst=True)
        query_training_data = list()
        for group_id in group_id_list:
            query_training_data = format_training_stats_data(
                TrainingStats.objects.filter(group=group_id, time_created__gte=date_from, time_created__lte=date_to),
                query_training_data)
        return HttpResponse(
            json.dumps({'message': 'Training Stats', 'status_code': 200, 'data': query_training_data}),
            status=200,
            content_type='application/json')
    except Exception as e:
        model_logger.exception("Training Stats Data Exception")
        return HttpResponse(status=400)
