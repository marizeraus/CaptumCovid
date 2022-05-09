import os
from django.apps import AppConfig
from django.conf import settings
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch


class ApiConfig(AppConfig):
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