# 输入参数，模型，视频片段，标签



# 导入库
import os
import av
import random
import torch
import numpy as np
import json
from transformers import AutoImageProcessor, TimesformerForVideoClassification

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (N, H, W, C).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def extract_frame(len_give, nums=8, rate=2): # 给的长度，要取多少帧，抽帧频率
    len_need = nums*rate
    if len_need>len_give:   # 如果要的帧大于给的帧长度，rate设为1
        rate = 1
        len_need = nums*rate
    if len_need>=len_give:   # 如果rate设置为1，仍然大于，直接返回
        return list(range(len_give))
        
    end = random.randint(len_need, len_give)
    start = end - len_need
    frame_id = range(start, end, rate)
    return list(frame_id)

    
def model_predict(model_checkpoint, class_labels, video):
    # 加载模型
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = TimesformerForVideoClassification.from_pretrained(model_checkpoint,ignore_mismatched_sizes=True)
    
    # 覆盖标签id
    # class_labels = class_labels[1:]
    my_label2id = {label: i for i, label in enumerate(class_labels)}
    my_id2label = {i: label for label, i in my_label2id.items()}
    
    model.config.label2id = my_label2id
    model.config.id2label = my_id2label

    # 推理
    with torch.no_grad():
        inputs = image_processor(list(video), return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
    
        softmax_probs = torch.softmax(logits[0,:len(model.config.id2label)], dim=0)
        topk_values, topk_indices = torch.topk(softmax_probs, k=5, dim=0)
        
        print("top5:")
        for i in range(5):
            print("      ", topk_values[i].item(), topk_indices[i].item(), model.config.id2label[topk_indices[i].item()])
        




model_checkpoint = "timesformer-Dyson_UCF101/checkpoint-31863"
class_labels = os.listdir("308C整题/DataSet/train")
file_path = "308C整题/mp4/782667.mp4"
container = av.open(file_path)

print()
for start in range(0, container.streams.video[0].frames, 30):
        # 随机抽帧进行预测
    if container.streams.video[0].frames-start<30:
        break
    indices = extract_frame(len_give=30, nums=8, rate=3)
    indices = [start + i for i in indices]

    print("所取的帧:", indices)
    video = read_video_pyav(container, indices)
    
    model_predict(model_checkpoint, class_labels, video)
