from django.core.exceptions import PermissionDenied
from django.views import generic

from bot.common.mixins import LoginRequiredMixin
from group.models import Group


# Create your views here.

class TrainingDashboard(LoginRequiredMixin, generic.TemplateView):
    template_name = 'training_dashboard.html'
    http_method_names = ['get']

    def get_context_data(self, **kwargs):
        if not self.request.user_features['training']:
            raise PermissionDenied

        context_data = super().get_context_data(**kwargs)
        group_list = Group.objects.filter(creator_id=self.request.user.id)
        group_data = [{"id": str(obj.id), "group_name": obj.name} for obj in group_list]
        context_data['group_data'] = group_data
        return context_data


class QueryTraining(LoginRequiredMixin, generic.TemplateView):
    template_name = 'query_training.html'
    http_method_names = ['get']
