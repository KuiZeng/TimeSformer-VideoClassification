##### ##### 0，导入所需库 ##### ##### 
import os
import numpy as np
import imageio
import cv2

import evaluate
import torch
from IPython.display import Image
import pytorchvideo.data

# 
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

# 数据预处理
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
    

from transformers import AutoImageProcessor, TimesformerForVideoClassification, VideoMAEImageProcessor
from transformers import TrainingArguments, Trainer

print("库导入完成")

##### ##### 1，参数设置 ##### ##### 
# 原模型
model_checkpoint = "facebook/timesformer-base-finetuned-ssv2"
# 数据集根目录
dataset_root_path = "Dyson_UCF101/" 
# 微调后的模型名字
new_model_name = "timesformer-Dyson_UCF101"


# 训练参数设置
batch_size = 1
epochs = 10
# model.config.num_frames = 8                  # 默认帧数长度
sample_rate = 4                                # 抽样率   # 标记：需要设置
fps = 30                                       # 标记：需要获取



# ##### ##### 2，模型加载  数据标签id设置 ##### ##### 
# # 获取数据集类别
# class_labels = os.listdir(dataset_root_path + "/train/")
# # my_label2id = {label: i+174 for i, label in enumerate(class_labels)}
# my_label2id = {label: i for i, label in enumerate(class_labels)}
# my_id2label = {i: label for label, i in my_label2id.items()}
# print(my_label2id)
# print(my_id2label)


# 模型加载  图像处理方法加载
model = TimesformerForVideoClassification.from_pretrained(model_checkpoint,ignore_mismatched_sizes=True)
image_processor = VideoMAEImageProcessor.from_pretrained(model_checkpoint)


# # 更新模型标签id映射
# # model.config.label2id.update(my_label2id)
# # model.config.id2label.update(my_id2label)
# # model.config.num_labels = len(model.config.id2label)
# # 更新模型标签id映射
# model.config.label2id = my_label2id
# model.config.id2label = my_id2label


##### ##### 3，数据预处理参数设置  数据集加载 ##### ##### 
# 参数
mean = image_processor.image_mean
std = image_processor.image_std
num_frames_to_sample = model.config.num_frames # 采样帧数
clip_duration = num_frames_to_sample * sample_rate / fps   # 抽帧持续时间   =   抽样率*帧数/fps


if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

# 训练集加载处理
train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)


train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)


# 验证集加载处理
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)


print("训练集，验证集，视频数量：", train_dataset.num_videos, val_dataset.num_videos)



##### ##### 4，其他处理 ##### ##### 
# 可视化预处理的视频，以便更好地进行调试

def unnormalize_img(img):
    """非标准化图像像素."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    """
    从视频张量中准备GIF.
    视频张量形状:(N, C, H, W).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def display_gif(video_tensor, gif_name="sample.gif"):
    """从视频张量中准备并显示GIF."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)


sample_train_video = next(iter(train_dataset))
video_tensor = sample_train_video["video"]
display_gif(video_tensor)

print("可视化预处理的视频完成")


# 定义批处理函数
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    """
    将examples一起批处理。每个批次由2个键组成，即pixel_values和labels。
    permute to (N, C, H, W)
    """
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

print("批定义完成")



##### ##### 5，训练配置 ##### ##### 

steps = (train_dataset.num_videos // batch_size) * epochs
# 参数
args = TrainingArguments(
    new_model_name,                         # 新模型的名称
    remove_unused_columns=False,            # 是否在训练时删除未使用的列。如果设置为True，默认删除未使用的列
    evaluation_strategy="epoch",            # 在训练过程中进行模型评估的策略。可以选择"no"（不评估），"epoch"（每个epoch评估一次）或"steps"（每个指定的训练步骤评估）。
    save_strategy="epoch",                  # 在训练过程中进行模型保存的策略。可以选择"no"（不保存），"epoch"（每个epoch保存一次）或"steps"（每个指定的训练步骤保存）。
    learning_rate=1e-5,                     # 模型的学习率。
    per_device_train_batch_size=batch_size, # 每个设备用于训练的批量大小。
    per_device_eval_batch_size=batch_size,  # 每个设备用于评估的批量大小。
    warmup_ratio=0.1,                       # 控制模型在训练开始时的学习率。
    logging_steps=10,                       # 控制在训练过程中将日志信息打印到输出的频率
    load_best_model_at_end=True,            # 是否在训练结束后加载最佳模型。
    metric_for_best_model="accuracy",       # 用于评估最佳模型的指标。在训练过程中，将根据该指标选择最佳的模型进行保存。
    push_to_hub=False,                      # 是否将最终训练的模型推送到模型中心。
    max_steps=steps,                        #指定训练的总步骤数。在这个指定的步骤数之后，训练将停止。这取决于训练数据集的大小和批量大小。
)

metric = evaluate.load("accuracy")
# 训练器
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print("训练参数设置完成") 



##### ##### 6，训练 ##### ##### 
print("开始训练")

train_results = trainer.train()

print("训练完成")


# # 训练完成后，使用.push_to_hub()方法将模型共享到 Hub
# trainer.push_to_hub()