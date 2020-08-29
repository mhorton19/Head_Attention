import sys
sys.path.insert(0, '../')

from transformers.configuration_roberta import *

class ElementwiseRobertaConfig(RobertaConfig):
    def __init__(self, use_elementwise=True, num_elementwise=5, **kwargs):
        self.use_elementwise = use_elementwise
        self.num_elementwise = num_elementwise
        super().__init__(**kwargs)
