# 标记数据很多无用的信息，提取出需要的字段，至少需要的字段，video_id(video_name),label_id,label_name,start,end



import json


with open("MyData/标记数据.json", "r") as json_file:
    json_data = json.load(json_file)


new_json_data = []

video_data_list = json_data["data"]
with open("MyData/精简数据.json", "w") as json_file:
    for video_data in video_data_list:
        print(video_data["id"], video_data["videoHistoryUrl"][0])
        new_video_dict = {
            "video_id":video_data["id"],
            "video_url":video_data["videoUrl"],
            "video_history_url":video_data["videoHistoryUrl"],
            "labels":[],
        }
        #print(video_data["answerVo"]["dataModels"][0]["dimensionVos"][0]["subDimensions"])
        flag = False
        
        
        # 第一个维度
        for label_data in video_data["answerVo"]["dataModels"][0]["dimensionVos"][0]["subDimensions"]:   
            if label_data["markMedias"]:
                flag = True
                for data in label_data["markMedias"]:
                    action_dict = {
                        "label_name": label_data["subDimensionCode"],
                        "label_id": label_data["subDimensionPrimaryId"],
                        "start": data["start"],
                        "end": data["end"],
                    }
                    new_video_dict["labels"].append(action_dict)

        # # 第二个维度
        for label_data in video_data["answerVo"]["dataModels"][0]["dimensionVos"][1]["subDimensions"]:   
            if label_data["markMedias"]:
                flag = True
                for data in label_data["markMedias"]:
                    action_dict = {
                        "label_name": label_data["subDimensionCode"],
                        "label_id": label_data["subDimensionPrimaryId"],
                        "start": data["start"],
                        "end": data["end"],
                    }
                    new_video_dict["labels"].append(action_dict)
        
        # # 第三个维度
        for label_data in video_data["answerVo"]["dataModels"][0]["dimensionVos"][2]["subDimensions"]:   
            if label_data["markMedias"]:
                flag = True
                for data in label_data["markMedias"]:
                    action_dict = {
                        "label_name": label_data["subDimensionCode"],
                        "label_id": label_data["subDimensionPrimaryId"],
                        "start": data["start"],
                        "end": data["end"],
                    }
                    new_video_dict["labels"].append(action_dict)
            
        
        # new_json_data[video_data["id"]].append({"start":video_data["videoUrl"][0]})
        if flag:
            new_json_data.append(new_video_dict)

    json.dump(new_json_data, json_file)

print(new_json_data)