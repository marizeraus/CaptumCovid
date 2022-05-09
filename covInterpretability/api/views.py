from django.shortcuts import render

import numpy as np
import pandas as pd

from .eval import predict

from .captum import captumView
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response


class CaptumVisualizationBase(APIView):
    def post(self, request):
        data = request.data
        model = ApiConfig.modelBert

        text = data['text']

        viz = captumView(text, model, 0)
        response_dict = {"Predicted Weight (kg)": viz}
        return Response(response_dict, status=200)


class CaptumVisualizationOthers(APIView):
    def post(self, request):
        model = ApiConfig.modelBertOthers
        data = request.data
        text = data['text']

        viz = captumView(text, model, 1)
        response_dict = {"Predicted Weight (kg)": viz}
        return Response(response_dict, status=200)


class Prediction(APIView):
    def post(self, request):
        data = request.data
        text = data['text']

        viz = predict(text, False)
        response_dict = {"prediction": ApiConfig.label_dict[viz]}
        return Response(response_dict, status=200)
