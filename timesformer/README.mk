# 基于Hugging Face开源模型timesformer的视频分类模型


0-获取标记数据                         # 产生标记数据.json
1-标记数据to精简数据                   # 产生精简数据.json  # 标记数据存在很多不需要的信息，提取出需要的字段，至少需要的字段，video_id(video_name),label_id,label_name,start,end
2-获取视频                             # 视频放在MyData/mp4下


如果是本地数据集,直接从这里开始   前置:视频(在MyData/mp4下),精简数据.json(在MyData下,需和示例的的json格式相同)
3-视频+精简数据to数据集                 # 视频片段放在MyData/label下	根据标记，对原视频进行裁剪，分割成标签片段，如果标记时没有负采样，还会自动进行负采样裁剪

如果仅用UCF-101模拟,直接从这里开始 前置:UCF-101数据集(在UCF-101下)
4-划分数据集                           # 划分训练集验证集测试集                                产生DataSet文件夹数据集
5-训练模型                             # 调用训练集验证集进行训练，                            产生checkpoint文件夹
6-测试模型                             # 调用生成的checkpoint，对测试集进行测试，              产生6-输出日志
7-数据统计                             # 测试结果数据统计                                      产生预测概率频次直方图
8-调优=》再训练，再测试=》调优          # 选择合适且能够实现的调优策略 
9-模型应用                             # 模型达到最优后，投入应用场景

