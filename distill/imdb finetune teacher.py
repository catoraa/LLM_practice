import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import logging
import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

train = load_dataset("glue", "sst2", split="train")
validation = load_dataset("glue", "sst2", split="validation")
split_result = train.train_test_split(test_size=0.2)  # 返回 DatasetDict
train = split_result["train"]
test = split_result["test"]

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train_dict = {'sentence': train["sentence"], 'label': train['label']}
    val_dict = {'sentence': validation["sentence"], 'label': validation['label']}
    test_dict = {"sentence": test['sentence']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    model_id = "microsoft/deberta-v3-base"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=510)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = DebertaV2ForSequenceClassification.from_pretrained(model_id)

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # LORA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        learning_rate=3e-4,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"accuracy: {eval_results['eval_accuracy']}")
    model.save_pretrained('./deberta-v3-base-sst2')
    print("model saved!")