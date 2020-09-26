import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../')

from transformers.modeling_bert import BertLayerNorm, BertSelfAttention
from transformers.GlobalLayers.ElementwiseAttention import GlobalElementwiseAttention

#TEST IS BIAS MATTERS AND IF BN IS BETTER
class AppendElementwiseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.elementwise_attention = GlobalElementwiseAttention(config.hidden_size, config.hidden_size, config.num_elementwise, use_bn=False)

        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.elementwise_attention(hidden_states, attention_mask=attention_mask)
        #dense_outputs = self.dense(attention_outputs)

        norm_outputs = self.LayerNorm(attention_outputs)

        cat_output = torch.cat((hidden_states, norm_outputs), axis=1)
        return cat_output



class ElementwiseBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.append_elementwise_attention = AppendElementwiseAttention(config)
        self.attention = BertSelfAttention(config)
        self.num_elementwise = config.num_elementwise
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        hidden_states = self.append_elementwise_attention(
            hidden_states,
            attention_mask.squeeze()
        )

        mask_addition = torch.zeros(*attention_mask.shape[:-1], self.num_elementwise, device=attention_mask.get_device())
        updated_mask = torch.cat((attention_mask, mask_addition), axis=-1)

        attention_outputs = self.attention(
            hidden_states,
            updated_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions
        )[0]

        sliced_outputs = attention_outputs[:,:-self.num_elementwise,:]

        return (sliced_outputs,)


