import json
from datetime import timedelta

import dateutil
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods

from group.models import UserQueryTraining
from ml_engine.api.api_exceptions import api_exception_handler
from ml_engine.config import model_logger
from ml_engine.helpers import http_basic_auth
from ml_engine.utils import extract_associated_group_intents
from ml_engine.utils import format_query_training_data
from .schema import UserQueryTrainingSchema

PER_PAGE_DATA_SIZE = 10


def pagination(data_object, page, per_page_query_size=10):
    paginator = Paginator(data_object, per_page_query_size)
    data = list()
    try:
        data = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        data = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        data = paginator.page(paginator.num_pages)
    return data.object_list, paginator.num_pages


@api_exception_handler
@http_basic_auth
@require_http_methods(['GET'])
def user_query_training(request):
    # TODO whether to return the page number or not, remove group name
    """
    This function performs the task of pagination on training page on the basis of page number and returns the
    ten user query training entry to api call.
    sample endpoint : http://127.0.0.1:8000/user-query-training/?group_id=58ac9e63e1ad9f79a691da80&time_created=2017-09-28&type=matched
    input params:
        group_id : a group id for which we need the data
        time_created : as the data is grouped by time created on therefore it is required to identify the entry.
        type : Whether it is a matched query or unmatched query
        page : which page number to return on pagination
    output : A list of dictionary containing the query of the group (matched/unmatched) on a particular day.

    matched case: flag_matched=true & flag_mapped=false
    unmatched case: flag_matched=false & flag_mapped=false
    mapped case: flag_matched=true & flag_mapped=true

    {
    "message": "User Query Training data Sent",
    "status_code": 200,
    "data": [
        {
            "intent_id": "",
            "intent_name": "",
            "flag_matched": False,
            "flag_mapped": False,
            "group_id": "58ac9e63e1ad9f79a691da80",
            "query": "hello how are you"
        }
    ]
}
    """
    model_logger.debug("User query intent assignment data sent")
    try:
        model_logger.debug("request data : {}".format(request.GET))
        schema_class = UserQueryTrainingSchema()
        schema, errors = schema_class.load(request.GET)
        if errors:
            model_logger.debug("Error in user query intent assignment : {error}".format(error=errors))
            return HttpResponse(
                json.dumps({'message': "failed to parse request data", "errors": errors, 'status_code': 400}),
                status=400,
                content_type='application/json')
        group_id = schema.get('group_id')
        try:
            intent_data = extract_associated_group_intents(group_id)
        except Exception as e:
            model_logger.exception("Exception in Group intent association")
            return HttpResponse({"message": "Group Intent Association Error"}, status=400,
                                content_type="application/json")
        time_created = schema.get('time_created')
        post_date = dateutil.parser.parse(time_created)
        incremented_date = post_date + timedelta(days=1)
        flag_matched = schema.get('flag_matched')
        flag_mapped = schema.get('flag_mapped')

        user_query_training_obj = UserQueryTraining.objects.filter(group=group_id,
                                        flag_matched=flag_matched,
                                        flag_mapped=flag_mapped,
                                        time_created__gte=post_date,
                                        time_created__lt=incremented_date)

        query_training_data = format_query_training_data(user_query_training_obj)

        page = schema.get('page', 1)
        paginated_query_data, page_count = pagination(query_training_data, page, PER_PAGE_DATA_SIZE)

        model_logger.debug("User query intent assignment completed")
        return HttpResponse(
            json.dumps({'message': 'User Query Training data Sent', 'status_code': 200, 'data': paginated_query_data,
                        "total_pages": page_count,
                        "total_count": user_query_training_obj.count(),
                        'intent_data': intent_data}),
            status=200,
            content_type='application/json')
    except Exception as e:
        model_logger.exception("User query intent assignment exception")
        return HttpResponse(status=400)
