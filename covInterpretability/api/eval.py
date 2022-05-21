import numpy as np
import torch
from unidecode import unidecode
from .apps import ApiConfig
import re 
label_others = {
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8
}



def predict(_text, is_other):
    tokenizer = ApiConfig.tokenizer_bertimbau
    model = ApiConfig.modelBert
    other_model = ApiConfig.modelBertOthers
    if len(_text) == 0:
        return 0
    with torch.no_grad():
        currText = tokenizer.encode_plus(
              _text,
              add_special_tokens = True,
              return_attention_mask = True,
             return_tensors = 'pt')        
        output = model(currText['input_ids'])
        logits = output.logits.detach().cpu().numpy()
        prediction = np.argmax(logits)
        if (prediction == 2 and not is_other):
          prediction = predict(_text, True)
          prediction = label_others[prediction]
        return prediction

def preProcessText(text):
    text = unidecode(text)
    text = text.replace(r"[^a-zA-Z ]","")
    text = re.sub(r"http\S+", "", text)
    return text
