from django.urls import path
from .views import CaptumVisualizationBase, CaptumVisualizationOthers, Prediction

urlpatterns = [
    path('captum/', CaptumVisualizationBase.as_view(), name = 'captum_viz'),
    path('captum/others', CaptumVisualizationOthers.as_view(), name = 'captum_viz_others'),
    path('predict/', Prediction.as_view(), name = 'prediction'),

]
