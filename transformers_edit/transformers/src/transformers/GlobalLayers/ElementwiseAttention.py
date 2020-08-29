import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../')

from transformers.GlobalLayers.masked_softmax import masked_softmax
from batchrenorm import BatchRenorm1d
class GlobalElementwiseAttention(nn.Module):
    def __init__(self, input_size, output_size, num_vectors, use_bn=True):
        super(GlobalElementwiseAttention, self).__init__()

        self.output_size = output_size
        self.num_vectors = num_vectors

        self.predict_global = nn.Linear(input_size, output_size * num_vectors, bias=False)
        self.predict_weight = nn.Linear(input_size, output_size * num_vectors, bias=False)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn = BatchRenorm1d(output_size * num_vectors)

    def forward(self, x, attention_mask=None, reshape_output=True):
        #DIVIDE BY SQRT(len) TO STABALIZE? Or is that just for dot prod?
        global_vecs = self.predict_global(x).permute(0, 2, 1)

        weight = self.predict_weight(x).permute(0, 2, 1)

        if(attention_mask != None):
            weight += attention_mask.unsqueeze(1)

        weight_vec = torch.softmax(weight, dim=-1)

        global_vec = torch.matmul(global_vecs.unsqueeze(-2), weight_vec.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        if self.use_bn:
            global_norm_vec = self.bn(global_vec)

        if reshape_output:
            out_vec = global_norm_vec.view(global_norm_vec.shape[0], self.num_vectors, self.output_size)
        else:
            out_vec = global_norm_vec

        return out_vec
