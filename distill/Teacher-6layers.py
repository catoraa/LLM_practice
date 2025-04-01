import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Config, DebertaV2Model, TrainingArguments, Trainer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder


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

# 加载teacher和student
teacher_id = "deberta-v3-base-sst2"
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_id)
student = create_student(teacher_model)
student.save_pretrained('deberta-v3-base-sst2-student')