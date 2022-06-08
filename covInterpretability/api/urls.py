from django.urls import path
from .views import CaptumVisualizationBase, CaptumVisualizationOthers, Prediction, TextPredict, Tweet

urlpatterns = [
    path('captum/', CaptumVisualizationBase.as_view(), name='captum_viz'),
    path('captum/others', CaptumVisualizationOthers.as_view(),
         name='captum_viz_others'),
    path('predict/', Prediction.as_view(), name='prediction'),
    path('tweet/', Tweet.as_view(), name='tweet'),
    path('text_predict/', TextPredict.as_view(), name='text_predict'),

]
