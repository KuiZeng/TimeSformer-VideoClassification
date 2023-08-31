
import json
import random
import os
from moviepy.editor import VideoFileClip



def getPosSlice(video, label_data, save_path):             # 获取视频正例时间段列表
    pos_slice = []
    for data in label_data:
        label = data["label_name"]
        if label not in class_labels:
            class_labels.append(label)
            os.mkdir(save_path + label)
        # 保存到三元组
        start = round(data["start"]*video.fps)
        end = round(data["end"]*video.fps)
        pos_slice.append([label, start, end])
        
    pos_slice = sorted(pos_slice, key=lambda x: x[1])     # 排序为正序
    return pos_slice   # [[label, start, end],[],]



def getStartEnd(old_start, old_end, fps):                 # 区间随机抽取反例时间段列表
    if old_end-old_start<2*fps:
        return []
    start,end = old_start, old_end
    mark = 0
    half = round(fps//2)
    for i in range(round(fps)):
        a = random.randint(old_start+half, old_end-half)
        b = random.randint(old_start+half, old_end-half)
        start,end = sorted([a,b])
        if end-start>fps:
            mark = 1
            break
    if mark==0:  # 如果没有结果，再试试固定值
        start, end = old_start+half, old_end-half
        if end-start<fps:
            return []
    return ["背景",start,end]
        

        
def getTotalSlice(video, pos_slice):          # 获取视频反例时间段列表
    total_slice = []
    new_slice = getStartEnd(0, pos_slice[0][1], video.fps)
    if new_slice != []:
        total_slice.append(new_slice)
    total_slice.append(pos_slice[0])

    for i in range(1, len(pos_slice)):
        old_start, old_end = pos_slice[i-1][2], pos_slice[i][1]
        new_slice = getStartEnd(old_start, old_end, video.fps)
        if new_slice != []:
            total_slice.append(new_slice)
        total_slice.append(pos_slice[i])
    new_slice = getStartEnd(pos_slice[-1][2], int(video.duration * video.fps), video.fps)
    if new_slice != []:
        total_slice.append(new_slice)
    return total_slice


# 视频对象     视频名    裁剪三元列表  保存地址
def writeVideoSlice(video, video_name, total_slice, save_path):   # 制作视频切片
    # 获取视频名
    video_name_head = video_name.split(".")[0]
    video_name_tail = video_name.split(".")[1]

    # 制作视频切片
    i = 0
    for label, start, end in total_slice:
        # 尾部_表示裁剪帧数 同时保证同类别不被覆盖
        new_video_name =  video_name_head + "_" + str(start) + "_" + str(end) + "." + video_name_tail
        print(label, start, end, new_video_name)
        
        new_video = video.subclip(start/video.fps, end/video.fps)
        new_video.write_videofile(save_path + label + "/" + new_video_name, fps=video.fps)

    video.close()




# 读取文件
with open("MyData/精简数据.json", "r") as json_file:
    # 将json文件读取为json数据
    json_data = json.load(json_file)



# 设置保存地址 及 初始标签
os.makedirs("MyData/original/背景", exist_ok=True)
save_path = "MyData/original/"
class_labels = os.listdir(save_path)
print("初始标签：", class_labels)



for video_data in json_data:
    label_data = video_data["labels"]
    # 获取该视频信息
    video_id = str(video_data["video_id"]) + ".mp4"
    video_path = "MyData/mp4/" + video_id
    print(video_path)
    video = VideoFileClip(video_path)
    

    # 获取正例列表
    pos_slice = getPosSlice(video, label_data, save_path)
    print("pos_slice", pos_slice)
    # 获取反例列表
    total_slice = getTotalSlice(video, pos_slice)
    print("total_slice", total_slice)
    
    # 裁剪视频，制作正反数据集
    writeVideoSlice(video, video_id, total_slice, save_path)

    
print("视频处理结束")