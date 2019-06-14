from django.core.management.base import BaseCommand
from bot.common.utils import model_queryset_generator
from group.models import UserQueryTraining

class Command(BaseCommand):
    help = 'Add flag_mapped field in user_query_training collection.'

    def handle(self, *args, **options):
        self.stdout.write("user_query_training updation started")
        obj_ids_list = model_queryset_generator(UserQueryTraining)

        for obj in obj_ids_list:
        	obj.update(flag_mapped=False)

        self.stdout.write("done updation of user_query_training table")
