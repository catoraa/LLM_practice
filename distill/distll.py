import torch
from torch import nn
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification


# 继承自huggingface的TrainingArguments
# alpha: Weighting Coefficient between teacher and label
# alpha=0.5: Equal weighting (default)
# alpha=1.0: Pure distillation (ignore true labels)
# alpha=0.0: Standard supervised training (ignore teacher)
# Temperature:Softmax Smoothing
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


# 继承自huggingface的Trainer
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


teacher_id = "deberta-v3-base-sst2"
student_id = "deberta-v3-base-sst2-student"
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_id)
student_model = AutoModelForSequenceClassification.from_pretrained(student_id)



training_args = DistillationTrainingArguments(
    output_dir="checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=100,
    save_strategy="no",
    evaluation_strategy="epoch",
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
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
