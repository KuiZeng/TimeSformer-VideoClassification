import matplotlib.pyplot as plt
import numpy as np
import json


with open("6-输出日志.json", "r") as json_file:
    json_data = json.load(json_file)
    


fig = plt.figure(figsize=(10, 10))
# 定义区间边界
bin_edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]




# 计算频数1
data_true = json_data["data_true"] 
hist, bins = np.histogram(data_true, bins=bin_edges)


# 绘制直方图1
plt.subplot(2, 1, 1)
plt.bar(bins[:-1], hist, width=5, align='edge')

# 在柱子中间位置添加频率标签
for i in range(len(hist)):
    if hist[i] > 0:
        plt.text(bins[:-1][i] + 2.5, hist[i], str(hist[i]), ha='center', va='bottom')

plt.xlabel("概率")
plt.ylabel(f"频次（总频次：{str(len(data_true))})")
plt.title("预测对时的概率-频次直方图")





# 计算频数2
data_false = json_data["data_false"] 
hist, bins = np.histogram(data_false, bins=bin_edges)

# 绘制直方图2
plt.subplot(2, 1, 2)
plt.bar(bins[:-1], hist, width=5, align='edge')

# 在柱子中间位置添加频率标签
for i in range(len(hist)):
    if hist[i] > 0:
        plt.text(bins[:-1][i] + 2.5, hist[i], str(hist[i]), ha='center', va='bottom')

plt.xlabel("概率")
plt.ylabel(f"频次（总频次：{str(len(data_false))})")
plt.title("预测错时的概率-频次直方图")




plt.savefig("7-频次直方图.PNG")
plt.show()
