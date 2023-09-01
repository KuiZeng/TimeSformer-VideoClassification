# 这只是获取标记数据的代码，会产生一个标记数据.json，位置在MyData/标记数据.json

from xxx import DataGen  # 公司库
import json


host = "xxx.xxx.xxx.xxx" # 公司
port = xxxx
data_gen = DataGen(host, port)


def export(project_id, output_file=None):
    data = data_gen.export_v2(project_id=project_id)
    data_obj = json.loads(data)
    print(data_obj)

    with open("MyData/标记数据.json", "w") as json_file:
        json.dump(data_obj, json_file)


# 执行
export("1692073672765956098")

