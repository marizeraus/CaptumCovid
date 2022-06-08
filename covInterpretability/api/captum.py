import string
import torch
from .apps import ApiConfig
from captum.attr import IntegratedGradients, LayerIntegratedGradients, InputXGradient
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import numpy as np


def predict_bertimbau(inputs):
    model = ApiConfig.modelBert

    pooled_output = model(inputs).logits
    return pooled_output


def predict_bertimbau_others(inputs):
    model = ApiConfig.modelBertOthers

    pooled_output = model(inputs).logits
    return pooled_output


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    device = ApiConfig.device

    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor(
        [[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(
        token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pair(text, tokenizer):
    device = ApiConfig.device
    # A token used for generating token reference
    ref_token_id = tokenizer.pad_token_id
    # A token used as a separator between question and text and it is also added to the end of the text.
    sep_token_id = tokenizer.sep_token_id
    # A token used for prepending to the concatenated question-text word sequence
    cls_token_id = tokenizer.cls_token_id

    text_ids = tokenizer.encode(
        text, add_special_tokens=False, max_length=512, truncation=True)
    # construct input token ids
    input_ids = text_ids
    # construct reference token ids
    ref_input_ids = [ref_token_id] * len(text_ids)

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)


def construct_input_ref_pos_id_pair(input_ids):
    device = ApiConfig.device
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def custom_forward_bertimbau(inputs):
    return predict_bertimbau(inputs)


def custom_forward_bertimbau_others(inputs):
    return predict_bertimbau_others(inputs)


def captumView(text, model, value, type):
    if(type == 0):
        ig_base = LayerIntegratedGradients(custom_forward_bertimbau,
                                           model.bert.embeddings)
    else:
        ig_base = LayerIntegratedGradients(custom_forward_bertimbau_others,
                                           model.bert.embeddings)
    data, attr, texto = viz_base(text, value, value, ig_base, model)
    return data, attr, texto


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    attributions, text = detokenize_attributions(attributions.tolist(), text)
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),
                            text,
                            delta))
    return attributions, text


def detokenize_attributions(attributions, textt):
    new_attr = []
    text = []
    pos = -1
    contador = -1
    for i in range(0, len(textt)):
        token = textt[i]
        if token.startswith("##"):
            if(contador == -1):
                contador = 1
            new_attr[pos] += attributions[i]
            text[pos] += token[2:]
            contador = contador+1
        else:
            if(contador > -1):
                new_attr[pos] = new_attr[pos]/contador
            pos = pos + 1
            contador = -1
            new_attr.append(attributions[i])
            text.append(token)
    return np.array(new_attr), text


def viz_base(text, label, target, ig_base, model):
    tokenizer = ApiConfig.tokenizer_bertimbau

    currText = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt')
    input_ids = currText['input_ids']
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    baseline = torch.zeros(input_ids.shape, dtype=torch.int64)
    attributions, delta = ig_base.attribute(inputs=input_ids,
                                            baselines=baseline,
                                            target=target,
                                            n_steps=200,
                                            internal_batch_size=5,
                                            return_convergence_delta=True)
    viz = []
    output = model(input_ids)
    logits = output.logits.detach().cpu().numpy()
    prediction = np.argmax(logits)
    pred = np.max(logits)
    attributions, texto = add_attributions_to_visualizer(
        attributions, all_tokens, pred, prediction, label, delta, viz)
    _ = visualization.visualize_text(viz)
    return _.data, attributions, texto


def text_base(text, label, type, target, model):
    if(type == 0):
        ig_base = LayerIntegratedGradients(custom_forward_bertimbau,
                                           model.bert.embeddings)
    else:
        ig_base = LayerIntegratedGradients(custom_forward_bertimbau_others,
                                           model.bert.embeddings)

    tokenizer = ApiConfig.tokenizer_bertimbau

    currText = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt')
    input_ids = currText['input_ids']
    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    baseline = torch.zeros(input_ids.shape, dtype=torch.int64)
    print(type, target)
    attributions, delta = ig_base.attribute(inputs=input_ids,
                                            baselines=baseline,
                                            target=target,
                                            n_steps=200,
                                            internal_batch_size=5,
                                            return_convergence_delta=True)
    viz = []
    output = model(input_ids)
    logits = output.logits.detach().cpu().numpy()
    prediction = np.argmax(logits)
    pred = np.max(logits)

    add_attributions_to_visualizer(
        attributions, all_tokens, pred, prediction, label, delta, viz)
    _ = visualization.visualize_text(viz)

    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    attributions, text = detokenize_attributions(attributions.tolist(), text)

    return attributions, text, _.data
