# 目标检测例子- CIFAR-10
&emsp;&emsp;在介绍了卷积神经网络(CNNs)背后的基础知识和直觉/动机之后，我们将在一个最流行的对象检测数据集上演示这一点。我们还将看到CNN的初始层如何获得关于对象的非常基本的特性，但是最终的卷积层将获得更多语义级别的特性，这些特性是在第一层的基本特性基础上建立起来的。<br>
&emsp;&emsp;本章将介绍以下主题：<br>
&emsp;&emsp;（1）目标检测<br>
&emsp;&emsp;（2）CIFAR-10对象检测法师模型的建立和培训<br>
## 目标检测
&emsp;&emsp;维基百科指出：<br>
&emsp;&emsp;目标检测：计算机视觉领域中用于在图像或视频序列中查找和识别对象的技术。尽管物体的图像在不同的视点、不同的大小和尺度上可能会有所不同，甚至在被翻译或旋转的时候也会有所不同，但人类并不费力就能识别出图像中的大量物体。当物体部分遮挡视线时，它们甚至可以被识别。这项任务对计算机视觉系统仍然是一个挑战。在过去的几十年里，许多方法都被用来完成这项任务。<br>
&emsp;&emsp;图像分析是深度学习中最突出的领域之一。图像很容易生成和处理，它们是机器学习的正确类型的数据:人类容易理解，但计算机很难理解。毫不奇怪，图像分析在深层神经网络的历史中扮演了关键角色。<br>
<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/image/81.png?raw=true">
</div>

&emsp;&emsp;随着自动汽车、人脸检测、智能视频监控以及人口计数解决方案的兴起，快速准确的目标检测系统需求量越来越大。这些系统不仅包括图像中的对象识别和分类，而且还可以通过在它们周围绘制适当的框来定位其中的每一个。这使得目标检测比传统的计算机视觉前身—图像分类任务更困难。<br>
&emsp;&emsp;在本章中，我们将研究目标检测——找出图像中的对象。例如，想象一辆自动驾驶汽车需要检测路上的其他汽车，如中所示图11.1。有许多复杂的目标检测算法。他们经常需要巨大的数据集、非常深的卷积网络和长的训练时间。<br>
## CIFAR-10 建模、构建与训
&emsp;&emsp;此示例显示如何创建CNN以在CIFAR-10数据集中对图像进行分类。 我们将使用一个简单的卷积神经网络实现几个卷积和完全连接的层。<br>
&emsp;&emsp;即使网络架构非常简单，您也会看到它的执行情况。所以，让我们开始这个实现。<br>
&emsp;&emsp;我们导入此实现所需的所有包:<br>
### 使用包
```
#%matplotlib inline线导向魔法函数，绘制命令的输出将在前端显示，该命令激活为 IPython 提供支持的“内联（inline）后端”
#IPython “内联后端” 也可以使用 IPython的 %config 命令进行微调。
#对于 Mac OS X 用户 %config InlineBackend.figure_format='retina' 是另一个有用的选项，它能提升 Matplotlib 图形在 Retina 屏上的质量
#urlretrieve()方法直接将远程数据下载到本地
#os.path.isfile()函数判断某一路径是否为文件
#os.path.isdir()函数判断某一路径是否为目录
#Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
#tarfile解压缩一个tar包
#Numpy支持大量的维度数组和矩阵运算，对数组运算提供了大量的数学函数库
#random() 方法返回随机生成的一个实数，它在[0,1)范围内
#matplotlib.pyplot绘图库
#LabelBinarizer标签二值化
#OneHotEncoder one-hot编码可以使分类更加准确
#pickle用于python特有的类型和python的数据类型间进行转换
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import pickle
import tensorflow as tf
```
### 加载CIFAR-10数据集
&emsp;&emsp;在此实施中，我们将使用CIFAR - 10，这是使用最广泛的数据集之一用于对象检测。因此，让我们首先定义一个辅助类来下载和提取 CIFAR - 10数据集(如果尚未下载) :<br>
```
cifar10_batches_dir_path = 'cifar-10-batches-py/'
tar_gz_filename = './data/cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_filename):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Python Images Batches') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_filename,
            pbar.hook)

if not isdir(cifar10_batches_dir_path):
    with tarfile.open(tar_gz_filename) as tar:
        tar.extractall()
        tar.close()
```
&emsp;&emsp;下载并提取CIFAR - 10数据集后，您会发现它已经存在分成五批。CIFAR - 10包含10个类别/类别的图像:<br>
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

&emsp;&emsp;在我们深入研究构建网络核心之前，让我们做一些数据分析预处理。<br>
### 数据分析和预处理
&emsp;&emsp;我们需要分析数据集并做一些基本的预处理。那么，让我们从定义一些辅助函数，使我们能够从这五个批次中加载特定的批次进行处理，并打印关于该批次及其样本的一些分析:<br>
```
# Defining a helper function for loading a batch of images
#定义用于加载一批图像的辅助函数
def load_batch(cifar10_dataset_dir_path, batch_num):
    with open(cifar10_dataset_dir_path + 'data_batch_' + str(batch_num), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    input_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    target_labels = batch['labels']

    return input_features, target_labels
```
&emsp;&emsp;然后，我们定义了一个函数，它可以帮助我们显示来自特定批次:<br>
```
# Defining a function to show the stats for batch ans specific sample
#定义一个函数来显示批次和特定样本的统计数据
def batch_image_stats(cifar10_dataset_dir_path, batch_num, sample_num):
    batch_nums = list(range(1, 6))

    # checking if the batch_num is a valid batch number
    #检查batch_num是否是有效的批号
    if batch_num not in batch_nums:
        print('Batch Num is out of Range. You can choose from these Batch nums: {}'.format(batch_nums))
        return None

    input_features, target_labels = load_batch(cifar10_dataset_dir_path, batch_num)

    # checking if the sample_num is a valid sample number
    #检查sample_num是否是有效的样本号
    if not (0 <= sample_num < len(input_features)):
        print('{} samples in batch {}.  {} is not a valid sample number.'.format(len(input_features), batch_num,
                                                                                 sample_num))
        return None

    print('\nStatistics of batch number {}:'.format(batch_num))
    print('Number of samples in this batch: {}'.format(len(input_features)))
    print('Per class counts of each Label: {}'.format(dict(zip(*np.unique(target_labels, return_counts=True)))))

    image = input_features[sample_num]
    label = target_labels[sample_num]
    cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print('\nSample Image Number {}:'.format(sample_num))
    print('Sample image - Minimum pixel value: {} Maximum pixel value: {}'.format(image.min(), image.max()))
    print('Samplei mage - Shape: {}'.format(image.shape))
    print('Sample Label - Label Id: {} Name: {}'.format(label, cifar10_class_names[label]))
    plt.axis('off')
    plt.imshow(image)
```
&emsp;&emsp;现在，我们可以使用这个函数来处理我们的数据集并可视化特定的图像:<br>
```
# Explore a specific batch and sample from the dataset
#从数据集中探索特定批次和样本
batch_num = 3
sample_num = 6
batch_image_stats(cifar10_batches_dir_path, batch_num, sample_num)
```
Statistics of batch number 3:<br>
Number of samples in this batch: 10000<br>
Per class counts of each Label: {0: 994, 1: 1042, 2: 965, 3: 997, 4: 990, 5: 1029, 6: 978, 7: 1015, 8: 961, 9: 1029}<br>

Sample Image Number 6:<br>
Sample image - Minimum pixel value: 30 Maximum pixel value: 242<br>
Samplei mage - Shape: (32, 32, 3)<br>
Sample Label - Label Id: 8 Name: ship<br>
![船](https://github.com/yanjiusheng2018/dlt/blob/master/image/8.2.jpg?raw=true)

在继续将数据集输入模型之前，我们需要将其标准化为0到1的范围。批量标准化优化了网络训练。 它已被证明有几个好处：<br>
更快的训练：由于在网络的前向传递期间的额外计算以及在网络的向后传播过程中训练的额外超参数，每个训练步骤将更慢。 但是，它应该更快地收敛，因此整体训练应该更快。<br>

更高的学习率：梯度下降算法通常需要较小的学习率才能使网络收敛到损失函数的最小值。 随着神经网络越来越深，它们的梯度值在反向传播过程中变得越来越小，因此它们通常需要更多的迭代。 使用批量标准化的想法允许我们使用更高的学习率，这进一步提高了网络训练的速度。<br>

容易初始化权重：权重初始化可能很困难，如果我们使用深度神经网络则会更加困难。 批量标准化似乎让我们在选择初始起始重量时要小心谨慎。<br>
