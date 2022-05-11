import os
from os.path import exists
from django.apps import AppConfig
from django.conf import settings
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import botocore
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def downloadModels():
    BUCKET_NAME = 'captum-conecta' 
    PATH = 'bertimbau3.model' 
    if(not exists(os.path.join(settings.MODELS, "bertimbau3.model"))):
        s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

        try:
            print("start downloading")
            s3.Bucket(BUCKET_NAME).download_file(PATH, 'bertimbau3.model')
            print("end downloading")
        except botocore.exceptions.ClientError as e: 
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
    PATH = "bertimbauOthers.model"
    if(not exists(os.path.join(settings.MODELS, "bertimbauOthers.model"))):
        s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

        try:
            print("start downloading")
            s3.Bucket(BUCKET_NAME).download_file(PATH, 'bertimbauOthers.model')
            print("end downloading")
        except botocore.exceptions.ClientError as e: 
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

class ApiConfig(AppConfig):
    downloadModels()
    name = 'api'
    MODEL_FILE = os.path.join(settings.MODELS, "bertimbau3.model")
    modelBert = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased",
                                                      num_labels=3,
                                                      output_attentions=False,
                                                      output_hidden_states=False).cpu()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelBert.to(device)

    modelBert.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))

    # load tokenizer
    tokenizer_bertimbau = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

    MODEL_FILE = os.path.join(settings.MODELS, "bertimbauOthers.model")

    modelBertOthers = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased",
                                                      num_labels=7,
                                                      output_attentions=False,
                                                      output_hidden_states=False).cpu()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelBertOthers.to(device)

    modelBertOthers.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))


    label_dict = {
    0: "Verdadeira",
    1: "Falsa",
    2: "Boato",
    3: "Duvidosa",
    4: "Manipulada",
    5: "Fora de contexto",
    6: "Questionável",
    7: "Sem evidências",
    8: "Meia Verdade"
    }

