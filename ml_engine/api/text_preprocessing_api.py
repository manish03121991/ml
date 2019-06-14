import json

from django.http import HttpResponse
from django.views.decorators.http import require_http_methods

from bot.common.decorators import log_execution_time
from bot.common.metrices import TEXT_PREPROCESSING
from ml_engine.api.api_exceptions import api_exception_handler
from ml_engine.config import ml_logger
from ml_engine.helpers import http_basic_auth
from ml_engine.preprocessor.text_processor_engine import TextProcessorEngine
from .schema import TextProcessingSchema


@log_execution_time(TEXT_PREPROCESSING)
@api_exception_handler
@http_basic_auth
@require_http_methods(['GET'])
def text_processing(request):
    """
    This is a api endpoint used for text preprocessing and filling processor engine
    with different parametric flags and text.

    Marshmallow schema is used to validate the request data.
    """
    ml_logger.debug("Text processing start")
    try:
        schema_class = TextProcessingSchema()
        req_params, errors = schema_class.load(request.GET)
        if errors:
            ml_logger.debug("Error in schema validation:%s" % (errors))
            return HttpResponse(json.dumps(errors), status=400)
        ml_logger.debug("Engine filled with request parameter dict:%s" % req_params)
        text_preprocessing_output = TextProcessorEngine().text_processor(**req_params)
        ml_logger.debug("Text preprocessing is completed")
    except Exception as e:
        ml_logger.debug("Exception raised in text processing:%s" % e)
        return HttpResponse(status=400)
    return HttpResponse(json.dumps({'data': text_preprocessing_output, 'status_code': 200}), status=200,
                        content_type='application/json')
