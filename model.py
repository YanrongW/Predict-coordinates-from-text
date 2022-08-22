import copy
import json
import math

from transformers import BertModel, BertPreTrainedModel
from transformers import MBartModel, MBartPreTrainedModel, MBartConfig
from transformers import GPT2Model, GPT2PreTrainedModel, GPT2Config
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Bert v1
class RBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(RBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.max_seq_length = 64

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 2)

        self.linear1 = nn.Linear(64*config.hidden_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden_states = outputs.last_hidden_state # outputs[0]
        batch_size = hidden_states.shape[0]
        hidden_states = self.dropout(hidden_states.reshape(batch_size, -1))

        nn1 = self.linear1(hidden_states)
        nn1 = self.relu(nn1)

        nn2 = self.linear2(nn1)
        nn2 = self.relu(nn2)

        nn3 = self.linear3(nn2)

        logits = nn3.clamp(-1.5, 1.5)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss(reduction='none')
                loss_mse = loss_fct(logits.view(-1), labels.view(-1)).view(-1, 2)
                loss_sum = torch.sum(loss_mse, dim=1).sqrt()
                loss_ori = torch.mul(loss_sum, 1e+5)
                loss = loss_ori.clamp(0, 50000).mean()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs



# gpt2
class RGPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(p=config.embd_pdrop)
        self.linear = nn.Linear(config.n_embd, 512, bias=False)
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, past_key_values=None,
                head_mask=None, inputs_embeds=None, labels=None
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear(hidden_states)
        nn1 = self.relu(hidden_states)

        nn2 = self.linear1(nn1)
        nn2 = self.relu(nn2)

        logits = self.linear2(nn2)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1

        pooled_logits = logits[torch.arange(batch_size, device=self.device), sequence_lengths]
        outputs = (pooled_logits,) + transformer_outputs[1:]
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss(reduction='none')
                loss_mse = loss_fct(pooled_logits.view(-1), labels.view(-1)).view(-1, 2)
                loss_sum = torch.sum(loss_mse, dim=1).sqrt()
                loss_ori = torch.mul(loss_sum, 1e+5)
                loss = loss_ori.clamp(0, 50000).mean()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs



# MBart
class MBartClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class RMBartForSequenceClassification(MBartPreTrainedModel):
    def __init__(self, config: MBartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.model = MBartModel(config)
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=None,
            use_cache=None,
            return_dict=None,
    ):
        if labels is not None:
            use_cache = False


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state # [batch_size, max_seq, d_model]  d_model = 1024
        eos_mask = input_ids.eq(self.config.eos_token_id)
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)

        output = (logits,) + outputs[1:]

        if labels is not None:
            if self.num_labels == 2:
                loss_fct = MSELoss(reduction='none')
                loss_mse = loss_fct(logits.view(-1), labels.view(-1)).view(-1, 2)
                loss_sum = torch.sum(loss_mse, dim=1).sqrt()
                loss = torch.mul(loss_sum, 1e+05).mean()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = (loss,) + output
        return output


# Bert v2
class RBert2ForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(RBert2ForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 2)

        self.linear1 = nn.Linear(config.hidden_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden_output = outputs.last_hidden_state # outputs[0]
        # pooled_output = outputs.pooler_output # outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits_ori = self.linear(pooled_output)
        # logits = logits_ori.clamp(-1.5,1.5)

        eos_mask = input_ids.eq(self.config.eos_token_id)
        sentence_representation = hidden_output[eos_mask, :].view(hidden_output.size(0), -1, hidden_output.size(-1))[:,-1, :]
        hidden_states = self.dropout(sentence_representation)
        nn1 = self.linear1(hidden_states)
        nn1 = self.relu(nn1)

        nn2 = self.linear2(nn1)
        nn2 = self.relu(nn2)

        nn3 = self.linear3(nn2)

        logits = nn3.clamp(-1.5, 1.5)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss(reduction='none')
                loss_mse = loss_fct(logits.view(-1), labels.view(-1)).view(-1, 2)
                loss_sum = torch.sum(loss_mse, dim=1).sqrt()
                loss_ori = torch.mul(loss_sum, 1e+5)
                loss = loss_ori.clamp(0, 50000).mean()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
