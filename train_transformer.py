from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import torch

tokenizer = ByteLevelBPETokenizer(
    "./EsperBERTo/vocab.json",
    "./EsperBERTo/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

from transformers_weighted_head.src.transformers import RobertaConfig

WEIGHT_HEADS = True

config = RobertaConfig(
    weight_heads=WEIGHT_HEADS,
    vocab_size=52_000,
    max_position_embeddings=514,
    hidden_size=480,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers_weighted_head.src.transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

from transformers_weighted_head.src.transformers.modeling_roberta import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

print(model.num_parameters())

from transformers_weighted_head.src.transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)

from transformers_weighted_head.src.transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers_weighted_head.src.transformers import Trainer, TrainingArguments


training_args = TrainingArguments(
    output_dir='./EsperBERTo' + ('_weighted_heads' if WEIGHT_HEADS else '_standard_heads'),
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_gpu_train_batch_size=8,
    save_steps=10_000,
    logging_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()