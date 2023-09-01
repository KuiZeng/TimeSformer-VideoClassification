# 这只是下载视频的代码，保存在MyData/mp4/，如果视频已在本地，移动到对应文件夹就好


import json
import requests


with open("MyData/精简数据.json", "r") as json_file:
    json_data = json.load(json_file)


i = 0
for video_data in json_data:
    i+=1
    video_id = video_data["video_id"]
    video_url = video_data["video_history_url"][0]
    print(video_url)

    response = requests.get(video_url)
    if response.status_code == 200:
        with open("MyData/mp4/" + str(video_id) + ".mp4", "wb") as f:
            f.write(response.content)
        print(video_id, "Video downloaded successfully.", i, "/", len(json_data))
    else:
        print(video_id, "Failed to download the video.")
    
    
