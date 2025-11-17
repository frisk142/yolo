import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob
import yaml
import torch
from torch.cuda import device
from ultralytics import YOLO
import yaml


CLASS_MAP = {""} #所需的类别名字

def json_yolo (json_dir,output):
    # 在output路径下创建images,labels文件
    images = Path((output)/"images")
    labels = Path((output)/"labels")

    os.makedirs(images,exist_ok=True)
    os.makedirs(labels,exist_ok=True)

    # 在output路径下生成images and labels两个文件
    # 使用glob遍历json文件
    # 思路是，用json_dir列表接住所有的.json文件，然后导入到data,并转换成py字典格式
    json_file = list(Path(json_dir).glob("**/*.json"))
    for json_path in json_file:
        with open(json_path,"r",encoding="utf-8") as f:
            data = json.load(f)

        # 提取图像信息 在data（data为py字典格式的文件内容）内寻找关键字并返回给变量,使用path.stem返回文件名字
        img_height = data["imgHeight"]
        img_width = data["imgWidth"]
        img_name = json_path.stem

        # 写yolo标签
        # 思路是打开labels文件,将img_name格式化，将变量img_name的值插入到文件名中，并加上.txt扩展名，
        # 然后打开文件检索标注点内容（shapes），查找标签，建立标签映射，如果标签不符合上面的CLASS_MAP，则跳过标签
        with open(labels/f"{img_name}.txt","w") as f:
            for shape in data["shapes"]:
                lbl = shape["label"]
                if lbl not in CLASS_MAP:
                    continue
                # 获取x和y的标注点，通过min和max获取最大和最小值，算最小外界矩形
                x_coords = [p[0] for p in shape["points"]]
                y_coords = [p[1] for p in shape["points"]]
                x_min , x_max = min(x_coords),max(x_coords)
                y_min , y_max = min(y_coords),max(y_coords)
                # 归一化,神秘的公式(只能理解后死记硬背)
                x_c = (x_min + x_max) /2 /img_width
                y_c = (y_min + y_max) /2 /img_height
                w = (x_max - x_min) /img_width
                h = (y_max - y_min) /img_height
                f.write(f"{CLASS_MAP[lbl]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            # 返回图像和标注目录
    return str(images),str(labels)


def shujufenge (images_dir,labels_dir,output_dir,train_ratio = 0.7):
    # 思路：将图片和标注文件绑在一起，查找出.jpg的路径和标注文件的路径，然后append在一起，然后导入到空白列表
    img_paths = [p for p in Path(images_dir).glob("*.png",)] # 列表推导式按照.jpg格式筛选图像
    pairs = []
    for img_p in img_paths:
        lbl_p = Path(labels_dir)/f"{img_p.stem}.txt"
        if lbl_p.exists():
            pairs.append((img_p,lbl_p))
    # 前面将图片和标注文件的路径存放到一起，然后设置按照7：3划分，42打乱种子，创建train，val和train，val的子文件夹文件夹
    # 之后分类copy到对应的子文件夹
    train_pairs,val_pairs = train_test_split(pairs,train_size = train_ratio,random_state=42)

    train = Path(output_dir)/"train"
    val = Path(output_dir)/"val"
    train_images = Path(train)/"images"
    train_labels = Path(train)/"labels"
    val_images = Path(val)/"images"
    val_labels = Path(val)/"labels"

    os.makedirs(train,exist_ok=True)
    os.makedirs(val,exist_ok=True)
    os.makedirs(train_images,exist_ok=True)
    os.makedirs(train_labels,exist_ok=True)
    os.makedirs(val_images,exist_ok=True)
    os.makedirs(val_labels,exist_ok=True)

    for img_t,lab_t in train_pairs:
        shutil.move(img_t,train_images)
        shutil.move(lab_t,train_labels)

    for img_v,lbl_v in val_pairs:
        shutil.move(img_v,val_images)
        shutil.move(lbl_v,val_labels)
    return str(train), str(val)

def yolo (output_y,epochs = 75,img_size = 800):
    data_y = Path(output_y)
    train = data_y/"train"
    val = data_y/"val"
    yolo_name = "yolov5n"

    yaml_data = {
        "path":data_y, # 文件根目录
        "train":train, # train位置
        "val":val, # val位置
        "nc":2, # 类别
        "names":"apple,kiwi" # 类别名字（于类别分类对饮）
    }
    yaml_path = data_y/"data_dir.yaml"

    with open (yaml_path,"w",encoding="utf-8") as f:
        yaml.dump(yaml_data.f)

    print("yaml创建完成")

    model = YOLO(yolo_name)

    results = model.train(
        data = str(yaml_data),
        epochs = epochs,
        batch = 16,
        imgsz = 800,
        device = 0,
        exits_ok = True,
        project = "run",
        name = "model",
        wrokers = 0,
        # 数据优化，yolov8自动进行数据优化，学习率为yolov8自动调整

    )

    metrics = model.val(workers = 0)
    print(f"MAP50{metrics.box.map50:.4f} ")
    print(f"MAP50-95{metrics.box.map:.4f}")

if __name__ == "__main__":
    print("开始转换yolo")
    output = r"" # 数据集根目录位置
    json_dir = r"" # 标注后json文件夹的位置
    images_path, labels_path = json_yolo(json_dir, output)
    print("转换完成")
    print("开始数据划分")
    shujufenge(images_path,labels_path,output)
    print("数据划分结束")
    print("开始yolo训练")
    yolo(output)
    print("yolo训练结束")
