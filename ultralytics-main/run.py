import csv
from ultralytics import YOLO
import torch
import os

# 加载模型，确保模型权重文件的路径正确
model = YOLO(r"C:\Users\ZhiYi\OneDrive\Desktop\ultralytics-main\ultralytics-main\runs\classify\train21\weights\best.pt")

# 指定要预测的图像或视频源，确保源文件的路径正确
source = r"C:\Users\ZhiYi\OneDrive\Desktop\ultralytics-main\datasets\cancer\ALL"
results = model.predict(source, show_conf=False)  # 对图像进行预测，设置show_conf为False以不显示置信度

# 初始化序号，用于记录处理图片的顺序编号
image_sequence_number = 1

# 打开或创建CSV文件，设置写入模式，并指定换行符为空字符串，避免写入CSV文件时出现额外空行
with open('cla_pre.csv', 'w', newline='') as csvfile:
    # 定义CSV文件中要写入的列名
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入CSV文件的表头，即列名
    writer.writeheader()

    # 遍历预测结果
    for result in results:
        # 获取类别概率张量，直接访问属性，无需括号
        probabilities = result.probs

        # 将概率张量移动到CPU内存，以便后续操作
        probabilities = probabilities.cpu()

        # 获取类别名称列表
        class_names = model.names

        # 将概率张量转换为numpy数组，以便访问具体的概率值
        probabilities = probabilities.numpy()

        # 获取最高概率的类别索引，按照Probs类提供的方式获取
        max_prob_index = probabilities.top1
        class_name = class_names[max_prob_index]  # 获取具有最高概率的类别名称

        # 获取源路径下图像文件名中的数字部分（假设文件名格式类似0095.jpg）
        image_name = os.path.splitext(os.path.basename(source))[0]

        # 将序号和预测类别信息以字典形式写入CSV文件
        writer.writerow({'id': image_sequence_number, 'label': class_name})

        # 每处理完一张图片，序号加1
        image_sequence_number += 1