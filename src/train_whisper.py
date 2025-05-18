import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

output_dir_name = "whisper-small-mal-finetuned"
output_dir = os.path.join("./outputs", output_dir_name)

print("Loading dataset...")
imasc_real = DatasetDict()
imasc_real = load_dataset(
    "thennal/IMaSC",
    split="train",
)

print("Preprocessing dataset...")
imasc_real = imasc_real.remove_columns(["speaker"])

test_size = 0.2
split_dataset = imasc_real.train_test_split(test_size=test_size, seed=42)
imasc_train = split_dataset["train"]
imasc_test = split_dataset["test"]

print(f"Train dataset size: {len(imasc_train)}")
print(f"Test dataset size: {len(imasc_test)}")

print("Loading feature extractor, tokenizer, and processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="malayalam", task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="malayalam", task="transcribe"
)


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


print("Mapping train dataset...")
imasc_train = imasc_train.map(prepare_dataset, num_proc=4)
print("Mapping test dataset (for eval)...")
imasc_test = imasc_test.map(prepare_dataset, num_proc=4)

print("Loading pre-trained model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


print("Initializing data collator...")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
)

print("Loading WER metric...")
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True, group_tokens=False
    )
    label_str = tokenizer.batch_decode(
        label_ids, skip_special_tokens=True, group_tokens=False
    )

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


print("Defining training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to="azure_ml",
)

print("Initializing Trainer...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=imasc_train,
    eval_dataset=imasc_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Starting training...")
trainer.train()

print("Training complete.")
print(f"Model and training outputs saved to {output_dir}")
