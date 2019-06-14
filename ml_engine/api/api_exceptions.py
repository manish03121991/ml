from django.utils.encoding import force_text
from functools import wraps
from django.http import JsonResponse


def api_exception_handler(f):
    @wraps(f)
    def decorated_function(request, *args, **kwargs):
        try:
            return f(request, *args, **kwargs)
        except APIException as e:
            # if isinstance(e.detail, str):
            response = {'error': force_text(e.detail)}
            # response['errors'] = e.errors
            response['status_code'] = e.code
            return JsonResponse(response, status=e.status_code)

    return decorated_function


class APIException(Exception):
    """
       Base class for API exceptions.
       Subclasses should provide `.status_code` and `.default_detail` properties.
       """
    # default - for unhandled exceptions - should ideally never happen, since
    # these will be mostly raised by devs themselves while handling requests.
    status_code = 500
    default_detail = 'A server error occurred.'

    def __init__(self, detail=None, errors=None, code=None):
        if detail is None:
            self.detail = force_text(self.default_detail)
        else:
            self.detail = force_text(detail)
        # self.errors = errors
        if code is None:
            self.code = self.status_code
        else:
            self.code = code

    def __str__(self):
        return self.detail


class AuthenticationFailed(APIException):
    status_code = 401
    default_detail = 'Incorrect authentication credentials.'


class NotAuthenticated(APIException):
    status_code = 401
    default_detail = 'Authentication credentials were not provided.'


class BadRequestData(APIException):
    status_code = 400
    default_detail = "Bad Request data"


class TrainingErrorException(APIException):
    status_code = 400
    default_detail = "Bad Request data"


class ModelException(APIException):
    status_code = 400
    default_detail = "Bad Request data"
