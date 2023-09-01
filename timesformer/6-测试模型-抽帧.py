# 该代码会将test下所有的视频片段都进行测试，并记录测试结果，保存在6-测试输出日志中

你至少需要设置模型的checkpoint  和   需要测试的根目录   和  class_labels


# 导入库
import os
import av
import random
import torch
import numpy as np
import json
from transformers import AutoImageProcessor, TimesformerForVideoClassification



# 选择模型
model_checkpoint = "timesformer-UCF101/checkpoint-8240"
# 需要测试的根目录
test_root_path = "DataSet/test/"

# 加载模型
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = TimesformerForVideoClassification.from_pretrained(model_checkpoint,ignore_mismatched_sizes=True)


# 覆盖标签id
class_labels = os.listdir("DataSet/train")
my_label2id = {label: i for i, label in enumerate(class_labels)}
my_id2label = {i: label for label, i in my_label2id.items()}

model.config.label2id = my_label2id
model.config.id2label = my_id2label

print(model.config.id2label)




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

    

# 输出
log = []


# 测试集统计
label_names = os.listdir(test_root_path)

data_true = []
data_false = []
total_right = 0
total_num = 0
        
alltrue = []    # 100
allfalse = []   # 0
half = []       # 40-60
moreture = []   # 60-100
morefalse = []  # 0-40

for label in label_names:
    right = 0
    num = 0
    filenames = os.listdir(test_root_path + label)
    print("【正在测试】", label)
    for filename in filenames:
        if filename == ".ipynb_checkpoints":
            continue

        file_path = test_root_path + label + "/" + filename
        # print("本次测试的视频是:", label + "/" + filename)
        container = av.open(file_path)
        # print("总帧数:", container.streams.video[0].frames)
        
        # 随机抽帧进行预测
        # model.config.num_frames = 8
        indices = extract_frame(len_give=container.streams.video[0].frames, nums=8, rate=3)
        # 取全部帧进行预测
        # model.config.num_frames = container.streams.video[0].frames
        # indices = list(range(container.streams.video[0].frames))
        
        
        # print("所取的帧:", indices)
        video = read_video_pyav(container, indices)
        
        # 推理
        inputs = image_processor(list(video), return_tensors="pt")
        # print(inputs["pixel_values"].shape)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        
        softmax_probs = torch.softmax(logits[0,:len(model.config.id2label)], dim=0)
        
        topk_values, topk_indices = torch.topk(softmax_probs, k=2, dim=0)
        
        # print("top3:")
        # for i in range(3):
        #     print("      ", topk_values[i].item(), topk_indices[i].item(), model.config.id2label[topk_indices[i].item()])
        
        predict = model.config.id2label[topk_indices[0].item()]
        # 解码
        out = f"预测标签: {predict},   预测概率: {topk_values[0].item()},   是否正确: {predict==label}"
        log.append(out)
        print(out)
        
        
        if predict==label:
            data_true.append(topk_values[0].item()*100)
        else:
            data_false.append(topk_values[0].item()*100)

        
        right += predict==label
        num += 1
        
        total_right += predict==label
        total_num+=1

    
    accuracy = int(right/num * 100)
    # print(accuracy)
    if accuracy == 100:
        alltrue.append(label)
    elif accuracy == 0:
        allfalse.append(label)
    elif 40 <= accuracy <= 60:
        half.append(label)
    elif 60 < accuracy < 100:
        moreture.append(label)  
    elif 0 < accuracy < 40:
        morefalse.append(label)

    

print(total_right, total_num)
print("全部正确：", alltrue)
print("正确居多：", moreture)
print("对错参半：", half)
print("错误居多：", morefalse)
print("全部错误：", allfalse)



# 字典数据
json_data = {
    "print": log,
    "accuary": [total_right, total_num, total_right/total_num],
    "全部正确": alltrue,
    "正确居多": moreture,
    "对错参半": half,
    "错误居多": morefalse,
    "全部错误": allfalse,
    "data_true": data_true,
    "data_false": data_false,
}

# 写入文件
with open("6-输出日志.json", "w") as json_file:
    # 将json数据写入json文件
    json.dump(json_data, json_file)

