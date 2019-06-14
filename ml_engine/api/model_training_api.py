import json
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from bot.common.decorators import log_execution_time
from bot.common.metrices import GROUP_MODEL_TRAINING
from ml_engine.api import api_exceptions
from ml_engine.api.api_exceptions import api_exception_handler, TrainingErrorException, ModelException
from ml_engine.config import model_logger
from ml_engine.helpers import http_basic_auth
from ml_engine.tasks import individual_group_training
from ml_engine.training.group_model_training import group_model_training
from .schema import GroupTrainingSchema
from ml_engine.utils import delete_training_status_from_cache

# TODO:
# intent field changes from keywords to phrases
# make correct representation of required in api request param
# is_delete field in group model
# need to set threshold value default in db
# elastic index deletion
# create data store for entities in es
# create system entity
# elasticsearch management command reindex and deletion
# chatbot api sub domain

# Points for accuracy
# a. stemming and lemma
# b. eliminating features with very low frequency
# c. grid search
# d. confusion metrics

def group_training(request):
    individual_group_training()
    return HttpResponse(json.dumps({'message': 'Group Training Completed', 'status_code': 200}), status=200,
                        content_type='application/json')
