import bson
from marshmallow import Schema, fields, validates, ValidationError
from group.models import Group
from intent.models import Intent
from marshmallow.decorators import post_load, pre_load
from user.models import BotUser
from chatbot.models import ChatBot
from ml_engine.api import api_exceptions
from ml_engine.preprocessor.preprocessor_handlers.constants import stopwords_allow_filter
from group.models import UserQueryTraining


class TextProcessingSchema(Schema):
    """
    This class would validate the different parameters coming in request
    for text processing API
    """
    text = fields.String(required=True)
    flag_tokenize = fields.Boolean(required=False, default=True)
    flag_stem = fields.Boolean(required=False, default=True)
    flag_punctuation_removal = fields.Boolean(required=False, default=False)
    flag_lemma = fields.Boolean(required=False, default=True)
    flag_tokenize_without_stop_words = fields.Boolean(required=False, default=False)
    flag_stem_without_stop_words = fields.Boolean(required=False, default=False)
    flag_lemma_without_stop_words = fields.Boolean(required=False, default=False)
    flag_unigram_without_stop_words = fields.Boolean(required=False, default=False)
    flag_bigram_without_stop_words = fields.Boolean(required=False, default=False)
    flag_trigram_without_stop_words = fields.Boolean(required=False, default=False)
    stem_unigram = fields.Boolean(required=False, default=False)
    stem_bigram = fields.Boolean(required=False, default=False)
    stem_trigram = fields.Boolean(required=False, default=False)
    filter_stopwords = fields.String(required=False, default='default')
    flag_pos_lemma = fields.Boolean(required=False, default=False)
    flag_pos_lemma_without_stopwords = fields.Boolean(required=False, default=False)

    @validates('filter_stopwords')
    def validate_filter_stopwords(self, value):
        if value not in stopwords_allow_filter:
            raise ValidationError("Stopwords filter is not allowed")
        return value


class GroupTrainingSchema(Schema):
    """
    Group training schema would be validated
    """
    group_id = fields.String(required=False)
    intent_id = fields.String(required=False)
    creator_id = fields.Integer(required=True)
    flag_group_delete = fields.Boolean(required=False, default=False)

    @validates('group_id')
    def validate_group_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("Group Id is not valid")
        return value

    @validates('intent_id')
    def validate_intent_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("Intent Id is not valid")
        return value

    @post_load
    def validate(self, data, many=None, partial=None):
        """
        Validation on creator id and its corresponding group and intent in request
        """
        if data.get('group_id'):
            group = Group.objects.filter(id=data.get('group_id'), creator_id=data.get('creator_id'))
            if not group:
                raise ValidationError("No group exists corresponding to given creator id")
        elif data.get('intent_id'):
            intent = Intent.objects.filter(id=data.get('intent_id'), creator_id=data.get('creator_id'))
            if not intent:
                raise ValidationError("No intent exists corresponding to given creator id")
        return data


class UserQueryTrainingSchema(Schema):
    """
    User query Training scheme validation
    """
    group_id = fields.String(required=True)
    intent_id = fields.String(required=False)
    time_created = fields.String(required=True)
    flag_matched = fields.Boolean(required=True)
    flag_mapped = fields.Boolean(required=True)
    page = fields.Int(required=False)

    # validation of group id mandatory

    @validates('group_id')
    def validate_group_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("Group Id is not valid")
        return value

    @validates('intent_id')
    def validate_intent_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("Intent Id is not valid")
        return value


    @post_load
    def validate(self, data, many=None, partial=None):
        """
        Validation on group id. (exists or not)
        """
        if data.get('group_id'):
            group = Group.objects.filter(id=data.get('group_id'))
            if not group:
                raise ValidationError("No group exists corresponding to given creator id")
        return data


class TrainingStatsSchema(Schema):
    """
    Training Stats validation schema
    """

    group_id = fields.String(required=False)
    date_from = fields.String(required=True)
    date_to = fields.String(required=True)

    # validation of group id mandatory
    @validates('group_id')
    def validate_group_id(self, value):
        group_id_list = value.split(',')
        for group_id in group_id_list:
            if not bson.objectid.ObjectId.is_valid(group_id):
                raise ValidationError("Group Id is not valid")
        return value


class TrainingQueryDeleteSchema(Schema):
    """
    User query deletion
    """

    user_query_id = fields.String(required=True)

    # validation of user query id mandatory
    @validates('user_query_id')
    def validate_group_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("user query Id is not valid")
        return value

    @post_load
    def validate(self, data, many=None, partial=None):
        """
        Validation on user query id. (exists or not)
        """
        from group.models import UserQueryTraining
        if data.get('user_query_id'):
            user_query_id = UserQueryTraining.objects.filter(id=data.get('user_query_id'))
            if not user_query_id:
                raise ValidationError("No such user query id %s exists in db" % (data.get('user_query_id')))
        return data


class GroupModelPredictionSchema(Schema):
    """
    Group model training schema would be validated
    """
    group_id = fields.String(required=True)
    text = fields.String(required=True)
    active_trigger_intent_status=fields.String(required=False)

    @validates('group_id')
    def validate_group_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("Group Id is not valid")
        return value


class IntentQueryAssignSchema(Schema):
    user_query_id = fields.String(required=True)
    intent_id = fields.String(required=True)
    query = fields.String(required=True)
    intent_name = fields.String(required=True)
    time_created = fields.String(required=True)


    @validates('intent_id')
    def validate_group_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("Intent id is not valid")
        return value

    @validates('user_query_id')
    def validate_group_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("user query id is not valid")
        return value

    @post_load
    def validate(self, data, many=None, partial=None):
        """
        Validation on user query id and intent id. (exists or not)
        """

        if data.get('user_query_id'):
            user_query_id = UserQueryTraining.objects.filter(id=data.get('user_query_id'))
            if not user_query_id:
                raise ValidationError("No such user query id %s exists in db" % (data.get('user_query_id')))
        if data.get('intent_id'):
            intent_id = Intent.objects.filter(id=data.get('intent_id'))
            if not intent_id:
                raise ValidationError("No such intent id %s exists in db" % (data.get('intent_id')))
        return data


class TrainingStatusSchema(Schema):
    """
    Schema for training status API endpoint
    """
    creator_id = fields.Integer(required=True)

    @validates('creator_id')
    def validates_creator_id(self, value):
        try:
            BotUser.objects.get(id=value)
            return value
        except BotUser.DoesNotExist:
            raise ValidationError("No such creator id exists")


class GroupThresholdBaseSchema(Schema):
    @post_load
    def validate(self, data, many=None, partial=None):
        """
        Validation on creator id and its corresponding chatbot in request
        """

        bot = ChatBot.objects.filter(id=data.get('bot_id'), creator_id=data.get('creator_id'))
        if not bot:
            raise ValidationError("No bot exists corresponding to given creator id")
        return data

    @validates('bot_id')
    def validate_bot_id(self, value):
        if not bson.objectid.ObjectId.is_valid(value):
            raise ValidationError("bot Id is not valid")
        return value


class GroupThresholdGETSchema(GroupThresholdBaseSchema):
    bot_id = fields.String(required=True)
    creator_id = fields.Integer(required=True)


class GroupThresholdPOSTSchema(GroupThresholdBaseSchema):
    group_id = fields.String(required=True)
    creator_id = fields.Integer(required=True)
    bot_id = fields.String(required=True)
    threshold_value = fields.Float(required=True)

    def update_threshold_value(self, data):
        if data.get('threshold_value') > 1.0:
            raise api_exceptions.BadRequestData("Threshold values can't be more than 1.0")
        group = Group.objects.filter(id=data.get('group_id'))
        if not group:
            raise api_exceptions.BadRequestData("Group id %s doesn't exist" % (data.get('group_id')))
        Group.objects(id=data.get('group_id')).update(set__threshold_value=data.get('threshold_value'))
        return


class AssignDeleteQuerySchema(Schema):
    approve = fields.Nested(IntentQueryAssignSchema, many=True)
    deleted = fields.List(fields.String())

    @validates('deleted')
    def validate_user_query_id(self, value):
        for user_query_id in value:
            if not bson.objectid.ObjectId.is_valid(user_query_id):
                raise ValidationError("user query Id is not valid")
            else:
                query_id = UserQueryTraining.objects.filter(id=user_query_id)
                if not query_id:
                    raise ValidationError("No such user query id %s exists in db" % (user_query_id))

        return value
