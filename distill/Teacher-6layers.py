import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch import nn
from torch.functional import F
import sys
import logging
import datasets
from datasets import load_dataset
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding, \
    DebertaV2Model, DebertaV2Config, Trainer, TrainingArguments
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder
import numpy as np
import evaluate
from scipy.stats import pearsonr
from scipy.special import softmax, kl_div


# 创建学生模型，隐藏层减半
def create_student(teacher_model):
    # 读取教师模型结构config
    config = teacher_model.config.to_dict()
    # 结构config中的隐藏层减半，整数除法
    config["num_hidden_layers"] //= 2
    # 重新生成配置对象
    config = DebertaV2Config.from_dict(config)
    # 初始化学生模型，保持和教师模型为同一类
    student_model = type(teacher_model)(config)
    # 复制教师模型的权重，进行学生模型权重初始化
    copy_deberta_weights(teacher_model, student_model)
    return student_model


# 权重复制
def copy_deberta_weights(teacher, student):
    # 检查模型类型
    # 1.DebertaV2Model或DebertaV2For开头的
    if isinstance(teacher, DebertaV2Model) or type(teacher).__name__.startswith('DebertaV2For'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            copy_deberta_weights(teacher_part, student_part)
    # 2.DebertaV2Encoder
    elif isinstance(teacher, DebertaV2Encoder):
        teacher_encoding_layers = [layer for layer in next(teacher.children())]
        student_encoding_layers = [layer for layer in next(student.children())]
        for i in range(len(student_encoding_layers)):
            # 学生模型的第i层对应教师模型的第2i层，实现层数减半的压缩
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2 * i].state_dict())
    # 3.其他模块
    else:
        student.load_state_dict(teacher.state_dict())


# 蒸馏参数
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


# 蒸馏trainer
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()
        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                                     F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (
                               self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

    def evaluate_step(self, model, inputs):
        outputs_student = model(**inputs)
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        return (
            outputs_student.logits.detach().cpu().numpy(),
            outputs_teacher.logits.detach().cpu().numpy(),
            inputs["labels"].detach().cpu().numpy()
        )

    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs_student = model(**inputs)
            outputs_teacher = self.teacher(**inputs)

        loss = outputs_student.loss

        if prediction_loss_only:
            return loss, None, None

        return outputs_student.logits, outputs_teacher.logits, inputs["labels"]


# 蒸馏数据加载
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

    # 加载teacher和student
    teacher_id = "./deberta-v3-base-sst2"
    teacher_model = DebertaV2ForSequenceClassification.from_pretrained(teacher_id)
    origin_id = "microsoft/deberta-v3-base"
    origin_model = DebertaV2ForSequenceClassification.from_pretrained(origin_id)
    student_model = create_student(origin_model)  #创建学生模型
    tokenizer = DebertaV2Tokenizer.from_pretrained(origin_id)


    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=510)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 初始化指标计算器
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")


    # 计算指标
    def compute_metrics(eval_pred):
        student_logits, teacher_logits, labels = eval_pred
        temperature = getattr(trainer.args, 'temperature', 1.0)  # 默认温度1.0

        # 确保logits是numpy数组
        student_logits = np.array(student_logits)
        teacher_logits = np.array(teacher_logits)
        # 计算softmax概率（如需）
        student_probs = softmax(student_logits / temperature, axis=-1)
        teacher_probs = softmax(teacher_logits / temperature, axis=-1)

        metrics = {
            "accuracy": accuracy_metric.compute(
                predictions=np.argmax(student_logits, axis=-1),
                references=labels
            )["accuracy"],
            "f1": f1_metric.compute(
                predictions=np.argmax(student_logits, axis=-1),
                references=labels,
                average="macro"
            )["f1"],
            "kl_div": kl_div(
                predictions=student_probs,
                references=teacher_probs
            )["kl_div"],
            "pearson_r": pearsonr(
                student_logits.flatten(),
                teacher_logits.flatten()
            )[0]
        }
        return metrics


    training_args = DistillationTrainingArguments(
        output_dir="checkpoint",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="step",
        # distillation parameters
        alpha=0.5,
        temperature=4.0
    )
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        teacher_model=teacher_model,
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)
    student_model.save_pretrained('./deberta-v3-base-simple-sst2')
    print("model saved!")
