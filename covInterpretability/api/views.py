from django.shortcuts import render

import numpy as np
import requests
from sympy import content
import json

from .eval import predict
from .eval import preProcessText

from .captum import captumView, text_base
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response


class CaptumVisualizationBase(APIView):
    def post(self, request):
        data = request.data
        model = ApiConfig.modelBert

        text = preProcessText(data['text'])
        value = int(data['class'])

        viz, _, __ = captumView(text, model, value, 0)
        response_dict = {"viz": viz}
        return Response(response_dict, status=200)


class CaptumVisualizationOthers(APIView):
    def post(self, request):
        model = ApiConfig.modelBertOthers
        data = request.data
        text = preProcessText(data['text'])
        value = int(data['class'])
        viz, _, __ = captumView(text, model, value, 1)
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


class Tweet(APIView):
    def post(self, request):
        data = request.data
        tweetId = data['id']
        requestUrl = "https://api.twitter.com/2/tweets?ids=" + tweetId
        header = {
            'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAM8ecwEAAAAAE5nohI79E67hkyOHfVM2MZbgfiQ%3D5PhXQqiLWn3jgtctbGBIhNHNOOnns59I2VmejUsHi19j5nG4mJ'}

        r = requests.get(requestUrl, headers=header)
        if(r.status_code == 200):
            text = json.loads(r.content)
            response_dict = {
                "text": text['data'][0]['text']
            }
            return Response(response_dict, status=200)
        return Response("", status=500)


class TextPredict(APIView):
    def post(self, request):
        data = request.data
        lines = data["text"]
        textReturn = []
        for line in lines:
            linePreproc = preProcessText(line)
            viz = predict(linePreproc, False)
            prediction = ApiConfig.label_dict[viz]
            type = 0
            model = ApiConfig.modelBert
            target = viz
            if(viz > 2):
                type = 1
                model = ApiConfig.modelBertOthers
                target = target - 2
            captumviz, attr, text = captumView(
                linePreproc, model, int(target), type)

            obj = {
                "text": line,
                "class": viz,
                "prediction": prediction,
                "attr": attr,
                "attrtext": text,
                "captumviz": captumviz
            }
            textReturn.append(obj)
        response_dict = {
            "return": textReturn
        }
        return Response(response_dict, status=200)
