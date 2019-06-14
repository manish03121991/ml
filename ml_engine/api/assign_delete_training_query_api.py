import json

import dateutil
from datetime import timedelta

from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings
from group.models import UserQueryTraining, TrainingStats
from intent.models import Intent
from ml_engine.api import api_exceptions
from ml_engine.api.api_exceptions import api_exception_handler
from ml_engine.config import model_logger
from ml_engine.helpers import http_basic_auth
from ml_engine.tasks import update_user_query_training_data
from .schema import AssignDeleteQuerySchema


def assign_query_request(assign_payload):
    """
    This function is called by the endpoint function to assign query to an intent
    :param assign_payload: consists list of dictionary with intent_id,query
    :return: None
    """
    try:
        for data in assign_payload:
            intent_id = data.get('intent_id')
            query = data.get('query')
            user_query_id = data.get('user_query_id')
            user_query_training_obj = UserQueryTraining.objects.get(id=user_query_id)
            if not user_query_training_obj:
                continue

            # matched/unmatched/mapped case: assign phrase to new intent
            new_intent = Intent.objects.get(id=intent_id)

            # Donot update the intent even when matched in chat
            # if user_query_training_obj.flag_matched:
            #     user_query_training_obj.intent = intent

            # update in phrase after updating intent
            new_intent.phrases.append(query)
            phrases_list = list(set(new_intent.phrases))

            # phrases limit in 1000 characters in edit intent form
            if len(json.dumps(phrases_list)) > settings.PHRASE_LIMIT:
                raise api_exceptions.BadRequestData(
                        "Ensure {} intent have phrases with at most 1000 "
                        "characters. It has {} characters.".format(new_intent.name, len(json.dumps(phrases_list))))

            new_intent.phrases = phrases_list
            new_intent.save()

            # matched/mapped case: remove phrase from old intent
            if user_query_training_obj.flag_matched and not intent_id == str(user_query_training_obj.intent):
                old_intent = Intent.objects.get(id=str(user_query_training_obj.intent))
                if query in old_intent.phrases:
                    old_intent.phrases.remove(query)
                    phrases_list = list(set(old_intent.phrases))
                    old_intent.phrases = phrases_list
                    old_intent.save()

            # update whole data
            update_user_query_training_data.delay(user_query_training_obj, query, new_intent)

    except api_exceptions.BadRequestData as exc:
        model_logger.exception(" == Error in query assignment module == ")
        raise api_exceptions.BadRequestData(exc.detail)

    except Exception as e:
        model_logger.exception(" == Error in query assignment module == ")
        raise api_exceptions.BadRequestData("Something went wrong.")
    return


@api_exception_handler
@http_basic_auth
@require_http_methods(['POST'])
def assign_delete_training_query(request):
    """
    This endpoint is used to assign a user query to particular intent as well as delete a query from training page.

    sample endpoint : http://127.0.0.1:8000/api/v1/query-assign-delete/
    Method : POST
    sample_payload_data
        {
        "approve":[{
            "intent_id": "intent_id",
            "intent_name": "intent_name",
            "user_query_id": "58ac9e63e1ad9f79a691da80",
            "query": "hello how are you"
        }],
        "deleted":['58ac9e63e1ad9f79a691da80']
        }


    :param request: request data consists of both query assignment data and query delete data
    :return: success message with status code 200
    """
    model_logger.debug(" == request for delete and assign query == ")
    try:
        data = json.loads(request.body.decode('ascii'))
        model_logger.debug("request data : {}".format(data))
        # nested schema for delete and approve is used
        schema_class = AssignDeleteQuerySchema()
        schema, errors = schema_class.load(data)
        if errors:
            model_logger.debug("Error in assign and delete request data : {error}".format(error=errors))
            return HttpResponse(
                json.dumps({'message': "failed to parse assign and  delete request data", "errors": errors,
                            'status_code': 400}),
                status=400,
                content_type='application/json')
        # parsing the user_query_id
        query_id_list = schema.get('deleted')
        for id in query_id_list:
            user_query_obj = UserQueryTraining.objects.get(id=id)
            user_obj_date = user_query_obj.time_created.replace(hour=0, minute=0, second=0, microsecond=0)

            training_stat = TrainingStats.objects.get(group=user_query_obj.group, time_created=user_obj_date)
            if user_query_obj.flag_mapped:
                training_stat.mapped_count -= 1
            elif user_query_obj.flag_matched:
                training_stat.matched_count -= 1
            else:
                training_stat.unmatched_count -= 1
            training_stat.save()
        UserQueryTraining.objects.filter(id__in=query_id_list).delete()
        model_logger.info(" == Query deletion done successfully == ")
    except Exception as e:
        model_logger.exception(" == Assign Delete training query exception == ")
        return HttpResponse(
            json.dumps({'message': "failed to parse assign and delete request data", "errors": "Something went wrong.",
                        'status_code': 400}),
            status=400,
            content_type='application/json')

    try:
        assign_query_request(schema.get('approve'))
    except Exception as e:
        return HttpResponse(
            json.dumps({'message': "failed to parse assign and delete request data", "errors": e.detail,
                        'status_code': 400}),
            status=400,
            content_type='application/json')

    return HttpResponse(
        json.dumps({'message': 'Delete and Query assignment done successfully', 'status_code': 200}),
        status=200,
        content_type='application/json')
