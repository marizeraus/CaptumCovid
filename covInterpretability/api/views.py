from django.shortcuts import render

import numpy as np

from .eval import predict
from .eval import preProcessText

from .captum import captumView
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response


class CaptumVisualizationBase(APIView):
    def post(self, request):
        data = request.data
        model = ApiConfig.modelBert

        text = preProcessText(data['text'])
        value = int(data['class'])

        viz = captumView(text, model, value, 0)
        response_dict = {"viz": viz}
        return Response(response_dict, status=200)


class CaptumVisualizationOthers(APIView):
    def post(self, request):
        model = ApiConfig.modelBertOthers
        data = request.data
        text = preProcessText(data['text'])
        value = int(data['class'])
        viz = captumView(text, model, value, 1)
        response_dict = {"viz": viz}
        return Response(response_dict, status=200)


class Prediction(APIView):
    def post(self, request):
        data = request.data
        text = preProcessText(data['text'])

        viz = predict(text, False)
        response_dict = {
            "prediction": ApiConfig.label_dict[viz],
            "class": viz
            }
        return Response(response_dict, status=200)
