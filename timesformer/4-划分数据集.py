# 这是划分数据集的代码，采用随机的方式，按比例进行划分训练验证测试集，data_path是初始数据地址，new_path是新数据集所在位置，执行后会产生new_path/train,new_path/val,new_path/test,其中每个目录下又有对应的标签文件夹
import os
import random
import shutil


def make_dataset(data_path, new_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 获取data_path下的所有文件夹名
    classes = os.listdir(data_path)

    # 在new_path下创建train, val, test文件夹
    train_path = os.path.join(new_path, 'train')
    val_path = os.path.join(new_path, 'val')
    test_path = os.path.join(new_path, 'test')
    # os.makedirs(train_path)
    # os.makedirs(val_path)
    # os.makedirs(test_path)

    # 遍历每个类别的文件夹
    for class_name in classes:
        print(f"现在执行的文件是：{class_name}")
        class_dir = os.path.join(data_path, class_name)
        files = os.listdir(class_dir)
        num_files = len(files)

        # 计算划分数量
        num_train = int(train_ratio * num_files)
        num_val = int(val_ratio * num_files)
        num_test = int(test_ratio * num_files)

        # 随机打乱文件顺序
        random.shuffle(files)

        # 将文件拷贝到对应的文件夹
        for i, file_name in enumerate(files):
            src_path = os.path.join(class_dir, file_name)
            if i < num_train:
                dst_path = os.path.join(train_path, class_name)
            elif i < num_train + num_val:
                dst_path = os.path.join(val_path, class_name)
            else:
                dst_path = os.path.join(test_path, class_name)
            os.makedirs(dst_path, exist_ok=True)
            shutil.copy(src_path, dst_path)

    print("数据集划分完成！")



make_dataset("MyData/label/", "DataSet/")
# make_dataset("UCF101_subset/label/", "Dyson_UCF101/")
