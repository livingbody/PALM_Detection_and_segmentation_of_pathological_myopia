# 一、飞桨常规赛：PALM病理性近视病灶检测与分割 5月第3名方案

**此方案来自大佬的基线，非原创。基线地址：[ https://aistudio.baidu.com/aistudio/projectdetail/1941312?channelType=0&channel=0](https://aistudio.baidu.com/aistudio/projectdetail/1941312?channelType=0&channel=0)**

**github地址：[https://github.com/livingbody/PALM_Detection_and_segmentation_of_pathological_myopia](https://github.com/livingbody/PALM_Detection_and_segmentation_of_pathological_myopia)**

**aistudio地址：[ https://aistudio.baidu.com/aistudio/projectdetail/1977014](https://aistudio.baidu.com/aistudio/projectdetail/1977014)**

本人有幸获得第三名。

![](https://ai-studio-static-online.cdn.bcebos.com/dc80241244184520aeb6a0b9d10ea3fc605f5563aba94ec7b1c92e793da6d56d)


## 1.**赛题简述**

	PALM病理性近视病灶检测与分割常规赛的重点是研究和发展与病理性近视诊断和患者眼底照片病变分割相关的算法。该常规赛的目标是评估和比较在一个常见的视网膜眼底图像数据集上检测病理性近视的自动算法。具体任务为：检测眼底图像是否出现视网膜萎缩病变和脱离病变，若有，需要实现病变区域的分割。

![](https://ai-studio-static-online.cdn.bcebos.com/bcb4c79dd34242e0bd83db8ea8062d7fa6a12a59999c44f09a73c9298a45fae3)
    
    
## 2.**数据基本标签**

	萎缩病变分割金标准：萎缩区域：0；背景：255；
	脱离病变分割金标准：脱离区域：0；背景：255。

## 3.**训练数据集**
文件名称：Train

Train里有两个文件夹，一个是fundus_images，一个是Lesion_Masks。

* fundus_images文件夹内包含**800**张眼底彩照，分辨率为1444×1444，或2124×2056。命名形如H0001.jpg、N0001.jpg、P0001.jpg和V0001.jpg。

* Lesion_Masks文件夹内包含两个文件夹：**Atrophy**和**Detachment**，其中，Atrophy文件夹包含fundus_images里眼底彩照的萎缩病变区域分割金标准，大小与对应的眼底彩照一致。命名前缀与对应眼底图像一致，后缀为bmp。同理，Detachment文件夹包含fundus_images里眼底彩照的脱离病变区域分割金标准，大小与对应的眼底彩照一致，命名前缀与对应眼底图像一致，后缀为bmp。**请注意，若Lesion_Masks中无某张眼底图像的病灶分割结果，说明该图像中不包含对应的病灶区域**。

## 4.**测试数据集**

文件名称：PALM-Testing400-Images.zip

压缩包里包含400张眼底彩照，命名形如T0001.jpg。


# 二、方案简介

## 1.解压数据与数据划分

    -- # 解压数据集
    
    -- !unzip -oq /home/aistudio/data/data85135/常规赛：PALM病理性近视病灶检测与分割.zip -d PaddleSeg/data
    
    -- # 划分数据
    
    -- !python utils/dataset_splited.py

## 2.数据标签预处理

    -- # 转换标签
    
    -- !python utils/dataset_pretrans.py
    
    * 原分类为1分类问题，为了问题研究的充分性和更大程度上利用多分类间的类别竞争对分类结构有一个更好的指导
    
    * 二分类问题描述，原标签为0不变，将255无效值转换为1值
    
    * 后期提交前会后处理，消去1值，换回赛题需要的255值

## 3.利用PaddleSeg套件加速赛题开发与测试: 使用套件config中的模型yml进行快速高效的实验开发——注意数据集yml的配置

## 4. 实现训练流程

## 5.实现预测流程

## 6. 完成提交结果 -- 基线方案为0.67+的得分(D_iter:500, A_iter:2000)，可从训练迭代次数、损失函数、模型入手

    -- # 提交结果后处理
    
    -- utils/post_process.py


# 三、数据处理

## 1. 先解压需要的PadleSeg套件


```python
# 解压PaddleSeg压缩包
!unzip -oq data/data88946/PaddleSeg.zip -d /home/aistudio/
# 修改文件名
!mv PaddleSeg-release-v2.0 PaddleSeg
```

上一步mv，可以将PaddleSeg加压后的文件目录改成PaddleSeg
>PaddleSeg下载至github的release2.0版本，为了方便大家使用，已添加在了数据集中供大家使用

## 2.清理data并添加数据


```python
# 删除data目录 —— 用于数据划分时，产生了意料之外的数据扩充时的数据重置
%cd /home/aistudio/
!rm -rf PaddleSeg/data
# 解压数据集到PaddleSeg目录下的data文件夹
!unzip -oq /home/aistudio/data/data85135/常规赛：PALM病理性近视病灶检测与分割.zip -d PaddleSeg/data
```

    /home/aistudio


## 3.查看数据


```python
# 查看数据集文件的树形结构
!tree -d PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割
```

    PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割
    ├── PALM-Testing400-Images
    └── Train
        ├── fundus_image
        └── Lesion_Masks
            ├── Atrophy
            └── Detachment
    
    6 directories


## 4.比赛数据集情况

PALM-Testing400-Images : 测试数据集文件夹

Train : 训练数据集文件夹

* Lesion_Masks ; 标注图片

	-- Detachment 视网膜脱落标注 -- 样本较少，存在同时萎缩的样本
  
   -- Atrophy 萎缩标注
  
* fundus_image : 原始图片

> 注意没有验证数据集，这里提供一个简单的划分程序，划分比例为0.7

> utils/dataset_splited.py

通过PIL的Image读取图片查看以下原数据与Label标注情况


```python
from PIL import Image
import numpy as np
# 读取图片
png_img = Image.open('PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割/Train/fundus_image/H0003.jpg')
png_img  # 展示真实图片
```


```python
bmp_img = Image.open('PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割/Train/Lesion_Masks/Atrophy/H0003.bmp')
bmp_img   # 展示萎缩标注图片
```


```python
bmp_img = Image.open('PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割/Train/Lesion_Masks/Detachment/P0053.bmp')
bmp_img   # 展示脱落标注图片
```

## 5.划分数据集与数据预处置

当前划分比例为0.8——可在utils文件夹下的dataset_splited.py修改**train_percent**为其它值

数据预处置-可在utils文件夹下的dataset_pretrans.py中查看相关代码--实现将255转化为1，原问题变二分类问题

> 注意：当前数据处理中，存在对数据进行扩充，因此当前程序运行一次之后会使得原分类数据数目增加——(扩增不宜过大，否则划分数据进行训练和验证时会出现偏差问题)

> 仅可运行一次，多次运行会导致填充数量过多，这是源码中扩充机制决定的(直接扩充到源文件夹中，所以下一次划分的时候就会默认把源文件中所有的文件读取)

> 感兴趣可前往查看dataset_splited.py的数据扩充区

```
# dataset_splited.py
import os
import random
import shutil
from tqdm import tqdm
from PIL import Image
import time
import numpy as np

random.seed(2021)

print('————开始数据清洗划分————')

records_png = []
records_a_bmp = []   # 记录萎缩标注
records_d_bmp = []   # 记录脱落标注
records_test = []

png_root = 'PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割/Train/fundus_image'
bmp_root = 'PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割/Train/Lesion_Masks'
test_root = 'PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割/PALM-Testing400-Images'

bmp_sub_root1 = 'Detachment'    # 脱落标注
bmp_sub_root2 = 'Atrophy'       # 萎缩标注

# 筛选原始训练数据中的jpg文件
for _, _, files in os.walk(png_root):
    for i in files:
        if i[-3:] == 'jpg':
            records_png.append(i)

# 筛选原始训练数据中的bmp文件
for _, _, files in os.walk(os.path.join(bmp_root, bmp_sub_root1)):
    for i in files:
        if i[-3:] == 'bmp':
            records_d_bmp.append(i)
for _, _, files in os.walk(os.path.join(bmp_root, bmp_sub_root2)):
    for i in files:
        if i[-3:] == 'bmp':
            records_a_bmp.append(i)

# 筛选原始测试数据中的jpg文件
for _, _, files in os.walk(test_root):
    for i in files:
        if i[-3:] == 'jpg':
            records_test.append(i)

print('BMP标注数据: Detachment: {0}张 \t Atrophy: {1}张'.format(len(records_d_bmp), len(records_a_bmp)))

# 明确保存目录方式
save_png_path_D = os.path.join('PaddleSeg/data', bmp_sub_root1)
save_png_path_A = os.path.join('PaddleSeg/data', bmp_sub_root2)
image_path = 'Image'
label_path = 'Label'
test_path = 'Test'

print('———— 开始构建数据目录 ————')
# 生成Detachment训练对应的目录
if os.path.exists(os.path.join(save_png_path_D, image_path)):
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_D, image_path)))
else:
    os.makedirs(os.path.join(save_png_path_D, image_path))
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_D, image_path)))
if os.path.exists(os.path.join(save_png_path_D, label_path)):
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_D, label_path)))
else:
    os.makedirs(os.path.join(save_png_path_D, label_path))
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_D, label_path)))
if os.path.exists(os.path.join(save_png_path_D, test_path)):
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_D, test_path)))
else:
    os.makedirs(os.path.join(save_png_path_D, test_path))
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_D, test_path)))
# 生成Atrophy训练对应的目录
if os.path.exists(os.path.join(save_png_path_A, image_path)):
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_A, image_path)))
else:
    os.makedirs(os.path.join(save_png_path_A, image_path))
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_A, image_path)))
if os.path.exists(os.path.join(save_png_path_A, label_path)):
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_A, label_path)))
else:
    os.makedirs(os.path.join(save_png_path_A, label_path))
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_A, label_path)))
if os.path.exists(os.path.join(save_png_path_A, test_path)):
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_A, test_path)))
else:
    os.makedirs(os.path.join(save_png_path_A, test_path))
    print('The Dir Has Create: {0} '.format(os.path.join(save_png_path_A, test_path)))

#############################################################################################################################################
#########################################   如果不需要数据扩充，可以注释以下部分代码(扩充到原来的数据中)     #####################################
#############################################################################################################################################
print('————开始针对类别Detachment进行数据扩充({0}张)————'.format(len(records_d_bmp)//6))

# 利用那些没有问题的jpg真实数据生成全255的标注图
# 然后扩充到少类别中进行微量数据补充
last_lens = len(records_d_bmp)        # 数据扩增前的数据长度
can_add_img = []                      # 搜寻可以被添加的图片
to_add_length = len(records_d_bmp)//6    # 扩增数量

add_tq = tqdm(records_png)
add_tq.set_description("Processing Train-{0} Split".format(bmp_sub_root1))
for i in add_tq:
    if (i[:-4] + '.bmp') not in records_d_bmp and (i[:-4] + '.bmp') not in records_a_bmp:
        img_ = Image.open(os.path.join(png_root, i))
        img_ = img_.convert('L')
        img_ = np.asarray(img_).copy()
        img_[:] = 255
        img_ = Image.fromarray(img_)
        img_.save(os.path.join(os.path.join(bmp_root, bmp_sub_root1), i[:-4] + '.bmp'))  # 保存到对应少类比原标注文件夹中
        can_add_img.append(i[:-4] + '.bmp')         # 可以被添加到少数类别中的图片
random.shuffle(can_add_img)
records_d_bmp = records_d_bmp + can_add_img[:to_add_length]
random.shuffle(records_d_bmp)
print('实际扩充: {0} 张, 现类别 {1} 拥有数据: {2} 份'.format(len(can_add_img), bmp_sub_root1, len(records_d_bmp)))


print('————开始针对类别Atrophy进行数据扩充({0}张)————'.format(len(records_a_bmp)//18))
# 利用那些没有问题的jpg真实数据生成全255的标注图
# 然后扩充到少类别中进行微量数据补充
last_lens = len(records_d_bmp)        # 数据扩增前的数据长度
can_add_img = []                      # 搜寻可以被添加的图片
to_add_length = len(records_a_bmp)//18    # 扩增数量

add_tq = tqdm(records_png)
add_tq.set_description("Processing Train-{0} Split".format(bmp_sub_root2))
for i in add_tq:  
    if (i[:-4] + '.bmp') not in records_a_bmp and (i[:-4] + '.bmp') not in records_d_bmp:
        img_ = Image.open(os.path.join(png_root, i))
        img_ = img_.convert('L')
        img_ = np.asarray(img_).copy()
        img_[:] = 255
        img_ = Image.fromarray(img_)
        img_.save(os.path.join(os.path.join(bmp_root, bmp_sub_root2), i[:-4] + '.bmp'))  # 保存到对应少类比原标注文件夹中
        can_add_img.append(i[:-4] + '.bmp')         # 可以被添加到少数类别中的图片
random.shuffle(can_add_img)
records_a_bmp = records_a_bmp + can_add_img[:to_add_length]
random.shuffle(records_a_bmp)
print('实际扩充: {0} 张, 现类别 {1} 拥有数据: {2} 份'.format(len(can_add_img), bmp_sub_root2, len(records_a_bmp)))

#############################################################################################################################################
#########################################                如果不需要数据扩充，可以注释以上部分代码          #####################################
#############################################################################################################################################


def Split_Train_And_Eval_Data(t_per, c_kind):
    '''根据类别划分训练、验证、测试数据
        t_per: 训练集划分比例
        c_kind: 当前划分类别
    '''
    assert c_kind in ['Detachment', 'Atrophy'], \
        'Only allowed the 2 kind class: [Detachment, Atrophy].'

    if c_kind == 'Detachment':   # 不同类别的存放区域
        save_png_path = save_png_path_D
        records = records_d_bmp
    elif c_kind == 'Atrophy':
        save_png_path = save_png_path_A
        records = records_a_bmp

    train_percent = t_per  # 划分比例
    print('The Split Params: train_percent={0:.2f}'.format(train_percent))

    train_records = records[:int(train_percent*len(records))]      # 训练验证的划分
    eval_records = records[int(train_percent*len(records)):]

    train_txt_path = os.path.join(save_png_path, 'train_list.txt')
    with open(train_txt_path, 'w') as f:
        train_tq = tqdm(train_records)
        train_tq.set_description("Processing Train-{0} Split".format(c_kind))
        for bmp_ in train_tq:                                                                        # 写入并保存训练图片
            png_ = bmp_[:-4]+'.jpg'
            old_png_path = os.path.join(png_root, png_)
            old_bmp_path = os.path.join(os.path.join(bmp_root, c_kind), bmp_)
            new_png_path = os.path.join(os.path.join(save_png_path, image_path), png_)
            new_bmp_path = os.path.join(os.path.join(save_png_path, label_path), bmp_)
            shutil.copyfile(old_png_path, new_png_path) 
            shutil.copyfile(old_bmp_path, new_bmp_path)
            f.write('{0} {1}\n'.format(os.path.join('Image', png_), os.path.join('Label', bmp_)))

    eval_txt_path = os.path.join(save_png_path, 'val_list.txt')
    with open(eval_txt_path, 'w') as f:                                                              # 写入并保存验证图片
        eval_tq = tqdm(eval_records)
        eval_tq.set_description("Processing Eval-{0} Split".format(c_kind))
        for bmp_ in eval_tq:
            png_ = bmp_[:-4]+'.jpg'
            old_png_path = os.path.join(png_root, png_)
            old_bmp_path = os.path.join(os.path.join(bmp_root, c_kind), bmp_)
            new_png_path = os.path.join(os.path.join(save_png_path, image_path), png_)
            new_bmp_path = os.path.join(os.path.join(save_png_path, label_path), bmp_)
            shutil.copyfile(old_png_path, new_png_path) 
            shutil.copyfile(old_bmp_path, new_bmp_path)
            f.write('{0} {1}\n'.format(os.path.join('Image', png_), os.path.join('Label', bmp_)))
    
    test_txt_path = os.path.join(save_png_path, 'test_list.txt')              
    with open(test_txt_path, 'w') as f:                                                               # 写入并保存测试图片
        test_tq = tqdm(records_test)
        test_tq.set_description("Processing Test-{0} Split".format(c_kind))
        for png_ in test_tq:
            old_png_path = os.path.join(test_root, png_)
            new_png_path = os.path.join(os.path.join(save_png_path, test_path), png_)
            shutil.copyfile(old_png_path, new_png_path) 
            f.write('{0}\n'.format(os.path.join('Test', png_)))

train_percent = 0.8  # 划分比例
print('——开始划分Detachment——')
Split_Train_And_Eval_Data(t_per=train_percent, c_kind=bmp_sub_root1)        # 划分类别1--Detachment
print('——开始划分Atrophy——')
Split_Train_And_Eval_Data(t_per=train_percent, c_kind=bmp_sub_root2)        # 划分类别2--Atrophy

# 展示树形结构
os.system('tree -d PaddleSeg/data')
```


```
# dataset_pretrans.py

import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import os
import time


print('\n————开始数据预处理转换————')
print('转换说明:')
print('\t 1. 默认标签为255与0，为了训练方便，将255转换为1，变成2分类问题')
print('\t 2. 新标签0与1，预测结束进行后处理即可得到赛题需要的结果')


bmp_sub_root1 = 'Detachment'    # 脱落标注
bmp_sub_root2 = 'Atrophy'       # 萎缩标注

data_root  = 'PaddleSeg/data'
label_path = 'Label'


def to_work_bmp(c_kind):

    assert c_kind in ['Detachment', 'Atrophy'], \
        'Only allowed the 2 kind class: [Detachment, Atrophy].'

    # 不同类别Label对应的存放位置
    data_path = os.path.join(data_root, c_kind)
    data_path = os.path.join(data_path, label_path)

    for _, _, files in os.walk(data_path):
        for f in tqdm(files):
            img = Image.open(os.path.join(data_path, f))
            img = np.asarray(img).copy()
            img[img == 255] = 1
            img = Image.fromarray(img)
            img.save(os.path.join(data_path, f))


print('————开始Detachment的数据预处置————')
time.sleep(0.2)

to_work_bmp(bmp_sub_root1)



print('————开始Atrophy的数据预处置————')
time.sleep(0.2)

to_work_bmp(bmp_sub_root2)
```


```python
# 保证路径为初始路径
%cd /home/aistudio

# 划分数据
!python utils/dataset_splited.py

# 转换标签--预处置
!python utils/dataset_pretrans.py
```

    Processing Train-Detachment Split: 100%|██████| 800/800 [00:18<00:00, 44.03it/s]
    实际扩充: 213 张, 现类别 Detachment 拥有数据: 21 份
    ————开始针对类别Atrophy进行数据扩充(32张)————
    Processing Train-Atrophy Split: 100%|█████████| 800/800 [00:18<00:00, 44.30it/s]
    实际扩充: 210 张, 现类别 Atrophy 拥有数据: 614 份
    ——开始划分Detachment——
    The Split Params: train_percent=0.80
    Processing Train-Detachment Split: 100%|███████| 16/16 [00:00<00:00, 156.44it/s]
    Processing Eval-Detachment Split: 100%|██████████| 5/5 [00:00<00:00, 153.25it/s]
    Processing Test-Detachment Split: 100%|██████| 400/400 [00:00<00:00, 567.17it/s]
    ——开始划分Atrophy——
    The Split Params: train_percent=0.80
    Processing Train-Atrophy Split: 100%|████████| 491/491 [00:03<00:00, 161.61it/s]
    Processing Eval-Atrophy Split: 100%|█████████| 123/123 [00:00<00:00, 172.47it/s]
    Processing Test-Atrophy Split: 100%|█████████| 400/400 [00:00<00:00, 594.46it/s]
    PaddleSeg/data
    ├── Atrophy
    │   ├── Image
    │   ├── Label
    │   └── Test
    ├── Detachment
    │   ├── Image
    │   ├── Label
    │   └── Test
    ├── __MACOSX
    │   └── 常规赛：PALM病理性近视病灶检测与分割
    │       ├── PALM-Testing400-Images
    │       └── Train
    │           ├── fundus_image
    │           └── Lesion_Masks
    │               ├── Atrophy
    │               └── Detachment
    └── 常规赛：PALM病理性近视病灶检测与分割
        ├── PALM-Testing400-Images
        └── Train
            ├── fundus_image
            └── Lesion_Masks
                ├── Atrophy
                └── Detachment
    
    23 directories
    
    ————开始数据预处理转换————
    转换说明:
    	 1. 默认标签为255与0，为了训练方便，将255转换为1，变成2分类问题
    	 2. 新标签0与1，预测结束进行后处理即可得到赛题需要的结果
    ————开始Detachment的数据预处置————
    100%|███████████████████████████████████████████| 21/21 [00:00<00:00, 53.26it/s]
    ————开始Atrophy的数据预处置————
    100%|█████████████████████████████████████████| 614/614 [00:10<00:00, 61.39it/s]


## 6.删除原数据目录


```python
# 移除’常规赛：PALM病理性近视病灶检测与分割‘文件夹
!rm -rf PaddleSeg/data/常规赛：PALM病理性近视病灶检测与分割
!rm -rf PaddleSeg/data/__MACOSX 
```

# 四、选择比赛模型

基线模型为:  配置略微修改的`PaddleSeg/configs/emanet/emanet_resnet50_os8_voc12aug_512x512_40k.yml`

具体配置在
	
	-- example/emanet_resnet50_os8_voc12aug_512x512_40k_Deta.yml
	* 用于分割脱落情况
	
	-- example/emanet_resnet50_os8_voc12aug_512x512_40k_Atro.yml
	* 用于分割萎缩情况

详细数据集配置在

	-- example/pascal_voc2012_Deta.yml
	* 用于设置脱落情况数据
	
	-- example/pascal_voc2012_Atro.yml
	* 用于设置萎缩情况数据


> 针对不同分割任务配置不同的分割模型，以适配不同的任务驱动

## 1.配置_base_中对应的数据集与模型

> 具体配置信息，可以在`examples`文件夹下查看相应`yml`文件，有相应的注释。

**简要说明`example/pascal_voc2012_Deta.yml`中的配置要点**，以方便大家修改其他的数据集yml适配模型训练

```yml
# 该文件需要自行移动到PaddleSeg/configs/_base_下, 并修改模型文件中的_base_路径(建议)
# 或者根据该文件中的train_dataset与val_dataset，对_base_下相应的yml进行修改

# 批大小   -- 可通过训练时动态调整
batch_size: 4
# 迭代次数 -- 可通过训练时动态调整 
iters: 40000

# 自定义数据集加载的方式
train_dataset:
  # 自定义数据集加载方式：Dataset
  type: Dataset
  # 数据集目录--当前项目中类别Detachment的数据都放在了这里：data/Detachment
  # 不同类别的训练，可以换成不同的数据根目录
  # Atrophy类--对应data/Atrophy
  dataset_root: data/Detachment
  # 该目录下划分数据产生的txt：data/Detachment/train_list.txt
  train_path: data/Detachment/train_list.txt
  # 类别--当前已转换为2分类问题
  num_classes: 2
  # 预处理
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    # 考虑到当前数据集正确划分区域较小，RandomPaddingCrop是否有必要不清楚，可以自行尝试
    # 下边有提供Resize处理缩放--选1缩放标准即可
    # - type: RandomPaddingCrop
    #   crop_size: [600, 600]
    - type: Resize
      target_size: [800, 800]
    - type: RandomHorizontalFlip
    - type: Normalize
  # 数据集加载方式--注意要一一对应
  mode: train

val_dataset:
  type: Dataset
  dataset_root: data/Detachment
  # 注意验证数据集的path和训练数据集path的区别
  val_path: data/Detachment/val_list.txt
  num_classes: 2
  transforms:
    # 修改padding为Resize，padding仅作填充，对于大图片无法缩放
    - type: Resize
      target_size: [800, 800]
    - type: Normalize
  # mode务必对应，否则无法索引正确的路径
  mode: val

# 原数据集
# train_dataset:
#   type: PascalVOC
#   dataset_root: data/VOCdevkit/
#   transforms:
#     - type: ResizeStepScaling
#       min_scale_factor: 0.5
#       max_scale_factor: 2.0
#       scale_step_size: 0.25
#     - type: RandomPaddingCrop
#       crop_size: [512, 512]
#     - type: RandomHorizontalFlip
#     - type: RandomDistort
#       brightness_range: 0.4
#       contrast_range: 0.4
#       saturation_range: 0.4
#     - type: Normalize
#   mode: train

# val_dataset:
#   type: PascalVOC
#   dataset_root: data/VOCdevkit/
#   transforms:
#     - type: Padding
#       target_size: [512, 512]
#     - type: Normalize
#   mode: val

# 以下参数可以在模型yml中被配置
# 优化器选择
optimizer:
  type: sgd
  momentum: 0.9
  # 正则化
  weight_decay: 4.0e-5

# 学习率--多项式
learning_rate:
  value: 0.01
  decay:
    type: poly
    power: 0.9
    end_lr: 0.0

# 损失配置项--可参考其它模型yml文件
loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
```

**至于模型配置，以当前使用Emanet为例说明**:

> 大家在使用模型yml时，为了保证数可读取，可以使用提供的两个数据集yml，分别操作加载不同的数据

```yml
# _base_: '../_base_/pascal_voc12aug.yml'

# 用调整后的pascal_voc2012.yml替换原始的_base_数据集配置文件
_base_: '../_base_/pascal_voc2012_Deta.yml'

model:
  # 模型名称
  type: EMANet
  backbone:
  	  # 骨干网络选择--可参考其它模型yml
    type: ResNet50_vd
    output_stride: 8
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz
  ema_channels: 512
  gc_channels: 256
  num_bases: 64
  stage_num: 3
  momentum: 0.2
  concat_input: True
  # 多损失
  enable_auxiliary_loss: True
  align_corners: True
  
# 最后会按照模型yml中的优化器进行优化
optimizer:
  type: sgd
  momentum: 0.9
  # 正则化参数
  weight_decay: 0.0005

# 最后会按照模型yml中的损失进行计算
loss:
  types:
    - type: CrossEntropyLoss
    - type: DiceLoss
  coef: [4.,2.]
```

> 将预置的基线配置yml移动到相应的文件夹下

* _base_: 数据加载yml存放

* emanet: emanet模型yml存放


```python
%cd /home/aistudio/
!cp -u example/pascal_voc2012_Atro.yml PaddleSeg/configs/_base_/
!cp -u example/pascal_voc2012_Deta.yml PaddleSeg/configs/_base_/
!cp -u example/emanet_resnet50_os8_voc12aug_512x512_40k_Deta.yml PaddleSeg/configs/emanet/
!cp -u example/emanet_resnet50_os8_voc12aug_512x512_40k_Atro.yml PaddleSeg/configs/emanet/
```

    /home/aistudio


## 2.启动训练

### 2.1.下载依赖项

在平台上可以不用执行，环境支持；线下可能需要下载。


```python
# 下载依赖项，保证PaddleSeg正常运行
%cd PaddleSeg
%pwd
!pip install -r requirements.txt
```

    /home/aistudio/PaddleSeg
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (1.21.0)
    Requirement already satisfied: yapf==0.26.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (0.26.0)
    Requirement already satisfied: flake8 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 3)) (3.8.2)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (5.1.2)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (2.1.1)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 6)) (4.1.1.26)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 7)) (4.36.1)
    Requirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 8)) (3.0.12)
    Requirement already satisfied: scipy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 9)) (1.6.3)
    Requirement already satisfied: prettytable in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 10)) (0.7.2)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.3.4)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (0.10.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (2.0.1)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.4.10)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (16.7.9)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (0.23)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.15.0)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (0.6.1)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (2.6.0)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (2.2.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.1)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (0.8.53)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.0.0)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (2.22.0)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (3.14.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.20.3)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (0.7.1.1)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (7.1.2)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->-r requirements.txt (line 1)) (0.6.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (7.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.16.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.10.1)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.9.9)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.18.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r requirements.txt (line 5)) (2019.3)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.25.6)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.0.4)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->-r requirements.txt (line 1)) (7.2.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.1)


### 2.2. 载入模型开始训练

> 更换自定义的模型文件时，只需要修改对应的模型yml、数据集yml(_base_中的yml)、以及替换下方的yml即可进行训练了

> 不要更改输出目录，否则后边的代码也需要修改，包括预测、后处理中的所有路径


```python
%cd ~/PaddleSeg
%pwd
# 训练分割Detachment的模型，并保存到/output/Detachment
!python train.py --c configs/emanet/emanet_resnet50_os8_voc12aug_512x512_40k_Deta.yml \
--use_vdl \
--save_interval 20 \
--do_eval \
--seed 2021 \
--iters 1000 \
--learning_rate 0.01 \
--save_dir ./output/Detachment

# 训练分割Atrophy的模型，并保存到/output/Atrophy
!python train.py --c configs/emanet/emanet_resnet50_os8_voc12aug_512x512_40k_Atro.yml \
--use_vdl \
--save_interval 20 \
--do_eval \
--seed 2021 \
--iters 3000 \
--learning_rate 0.01 \
--save_dir ./output/Atrophy
```

    Aborted (core dumped)

## 3.开始预测

这可以直接使用emanet进行预测，不用修改数据集yml，也不用修改相应的模型yml，注意训练权重对应即可！

> 预测结果按类别分别放在`./output/result/Detachment` 和 `./output/result/Atrophy` 下

> 如果使用不同的模型对不同的分割类进行讨论，注意模型yml即可

> 提交结果为两种单独预测的结果

**默认使用Iou评估最好的模型训练参数--best_model**


```python
%cd ~/PaddleSeg/
# 预测Detachment
!python predict.py --config configs/emanet/emanet_resnet50_os8_voc12aug_512x512_40k_Deta.yml \
--model_path output/Detachment/best_model/model.pdparams \
--image_path data/Detachment/Test \
--save_dir ./output/result/Detachment

# 预测Atrophy
!python predict.py --config configs/emanet/emanet_resnet50_os8_voc12aug_512x512_40k_Atro.yml  \
--model_path output/Atrophy/best_model/model.pdparams \
--image_path data/Atrophy/Test \
--save_dir ./output/result/Atrophy
```

# PaTTA使用


```python
!pip install PaTTA
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting PaTTA
      Downloading https://mirror.baidu.com/pypi/packages/72/fa/3a30d550d7fc2a2decce111dbbe890dc14a30a18e3d8df3729aca961b041/patta-0.0.2-py3-none-any.whl
    Installing collected packages: PaTTA
    Successfully installed PaTTA-0.0.2


## 4. 使用 export.py 脚本进行模型导出


```python
%cd PaddleSeg/
!python export.py \
       --config  configs/emanet/emanet_resnet50_os8_voc12aug_512x512_40k_Deta.yml \
       --model_path output/Detachment/best_model/model.pdparams \
       --save_dir output/emanet

!python export.py \
       --config configs/emanet/emanet_resnet50_os8_voc12aug_512x512_40k_Atro.yml \
       --model_path output/Atrophy/best_model/model.pdparams \
       --save_dir output/atrophy
```

# 五、预测结果后处理

* 将类别值1换为255，进行赛题结果提交

## 1. PaTTA安装



```python
!git clone https://gitee.com/livingbody/PaTTA.git --depth=1
```

    Cloning into 'PaTTA'...
    remote: Enumerating objects: 30, done.[K
    remote: Counting objects: 100% (30/30), done.[K
    remote: Compressing objects: 100% (25/25), done.[K
    remote: Total 30 (delta 0), reused 25 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (30/30), done.
    Checking connectivity... done.



```python
%cd ~
import glob
path = glob.glob('/home/aistudio/Lesion_Segmentation/Detachment/*')
f = open('de_test.txt', 'w')
for i in path:
    f.write(i+'\n')
f.close()
```

    /home/aistudio



```python
%cd ~
import glob
path = glob.glob('/home/aistudio/Lesion_Segmentation/Atrophy/*')
f = open('at_test.txt', 'w')
for i in path:
    f.write(i+'\n')
f.close()
```

    /home/aistudio


## 2.将类别值1换为255


```python
%cd /home/aistudio/
!python utils/post_process.py
```

    /home/aistudio
    ————开始提交结果前的后处理————
    ————开始Detachment预测结果后处理————
    100%|█████████████████████████████████████████| 400/400 [00:24<00:00, 16.09it/s]
    ————开始Atrophy预测结果后处理————
    100%|█████████████████████████████████████████| 400/400 [00:24<00:00, 16.44it/s]
    后处理完成(cost: 49.602564334869385 s)！


## 3.PaTTA成绩提升


```python
%cd ~/PaddleSeg/

!python PaTTA/tools/seg.py --model_path='output/emanet/model' \
                 --batch_size=16 \
                 --test_dataset='/home/aistudio/de_test.txt'
```


```python
%cd ~/PaddleSeg/
!mkdir result
!python PaTTA/tools/seg.py --model_path='output/atrophy/model' \
                 --batch_size=16 \
                 --test_dataset='/home/aistudio/at_test.txt'
```

# 六、提交比赛结果

## 1.提交文件整理


```python
# 复制文件到最顶层目录
%cd /home/aistudio
!cp -r PaddleSeg/output/result/ Lesion_Segmentation

# 过程移动文件--保证不包含生成的子目录
%cd Lesion_Segmentation
!cp -r Detachment/pseudo_color_prediction/. Detachment
!cp -r Atrophy/pseudo_color_prediction/. Atrophy

# 获取指定的提交目录格式
!rm -rf Detachment/added_prediction
!rm -rf Detachment/pseudo_color_prediction

!rm -rf Atrophy/added_prediction
!rm -rf Atrophy/pseudo_color_prediction
```

    /home/aistudio
    /home/aistudio/Lesion_Segmentation


## 2.提交文件打包


```python
# 压缩文件
%cd /home/aistudio
!zip -r Lesion_Segmentation.zip Lesion_Segmentation
```

# 七、其它建议


* 1. 模型建议：注意力模型(EMANet等)或者调整unet模型
* 2. 损失建议：多损失结构，不同的coef，针对赛题的特殊损失等(Dice等)
* 3. 模型魔改建议：尝试对Unet添加注意力模块，修改参数，或者调整不同的backbone与indices组合
* 4. 优化器与学习率策略的调整
* 5. 分割任务PaTTA可以多多尝试

# 八、附件

## 1.notebook代码

[javaroom.ipynb](javaroom.ipynb)

## 2.Atrophy_best_model.zip模型

[Atrophy_best_model.zip](Atrophy_best_model.zip)

## 3.Detachment_best_model.zip模型

[Detachment_best_model.zip](Detachment_best_model.zip)

