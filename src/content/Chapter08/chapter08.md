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
<br/>
更快的训练：<br>
&emsp;&emsp;由于在网络的前向传递期间的额外计算以及在网络的向后传播过程中训练的额外超参数，每个训练步骤将更慢。 但是，它应该更快地收敛，因此整体训练应该更快。<br>
<br/>
更高的学习率：<br>
&emsp;&emsp;梯度下降算法通常需要较小的学习率才能使网络收敛到损失函数的最小值。 随着神经网络越来越深，它们的梯度值在反向传播过程中变得越来越小，因此它们通常需要更多的迭代。 使用批量标准化的想法允许我们使用更高的学习率，这进一步提高了网络训练的速度。<br>
<br/>
容易初始化权重：<br>
&emsp;&emsp;权重初始化可能很困难，如果我们使用深度神经网络则会更加困难。 批量标准化似乎让我们在选择初始起始重量时要小心谨慎。<br>
<br/>
&emsp;&emsp;因此，让我们继续定义一个函数，该函数将负责规范输入图像，使这些图像的所有像素值介于0和1之间:<br>
```
# Normalize CIFAR-10 images to be in the range of [0,1]
#将CIFAR-10图像标准化为[0,1]范围
def normalize_images(images):
    # initial zero ndarray
    normalized_images = np.zeros_like(images.astype(float))

    # The first images index is number of images where the other indices indicates
    #第一图像索引是其他索引指示的图像的数量
    # hieight, width and depth of the image
    #图像的高度，宽度和深度
    num_images = images.shape[0]

    # Computing the minimum and maximum value of the input image to do the normalization based on them
    #计算输入图像的最小值和最大值，以便根据它们进行标准化
    maximum_value, minimum_value = images.max(), images.min()

    # Normalize all the pixel values of the images to be from 0 to 1
    #将图像的所有像素值标准化为0到1
    for img in range(num_images):
        normalized_images[img, ...] = (images[img, ...] - float(minimum_value)) / float(maximum_value - minimum_value)

    return normalized_images
```
&emsp;&emsp;接下来，我们需要实现另一个辅助函数来对输入图像的标签进行编码。在这个函数中，我们将使用sklearn的one-hot编码，其中每个图像标签都由一个0向量表示，除了这个向量表示的图像的类索引之外。输出向量的大小将取决于我们在数据集中拥有的类的数量，在CIFAR-10数据中是10个类:<br>
```
# encoding the input images. Each image will be represented by a vector of zeros except for the class index of the image
#编码输入图像。 除了图像的类索引之外，每个图像将由零向量表示
# that this vector represents. The length of this vector depends on number of classes that we have
#这个向量代表。这个向量的长度取决于我们拥有的类的数量
# the dataset which is 10 in CIFAR-10
#CIFAR-10中的数据集为10

def one_hot_encode(images):
    num_classes = 10

    # use sklearn helper function of OneHotEncoder() to do that
    #使用OneHotEncoder（）的sklearn辅助函数来做到这一点
    encoder = OneHotEncoder(num_classes)

    # resize the input images to be 2D
    #将输入图像的大小调整为2D
    input_images_resized_to_2d = np.array(images).reshape(-1, 1)
    one_hot_encoded_targets = encoder.fit_transform(input_images_resized_to_2d)

    return one_hot_encoded_targets.toarray()
```
&emsp;&emsp;现在，是时候调用前面的辅助函数来进行预处理和持久化数据集，以便我们以后可以使用它了:<br>
```
def preprocess_persist_data(cifar10_batches_dir_path, normalize_images, one_hot_encode):
    num_batches = 5
    valid_input_features = []
    valid_target_labels = []

    for batch_ind in range(1, num_batches + 1):
        # Loading batch
        #加载批次
        input_features, target_labels = load_batch(cifar10_batches_dir_path, batch_ind)
        num_validation_images = int(len(input_features) * 0.1)

        # Preprocess the current batch and perisist it for future use
        #预处理当前批次并对其进行保存以备将来使用
        input_features = normalize_images(input_features[:-num_validation_images])
        target_labels = one_hot_encode(target_labels[:-num_validation_images])

        # Persisting the preprocessed batch
        #保留预处理的批处理
        pickle.dump((input_features, target_labels), open('./preprocess/preprocess_train_batch_' + str(batch_ind) + '.p', 'wb'))

        # Define a subset of the training images to be used for validating our model
        #定义用于验证模型的训练图像的子集
        valid_input_features.extend(input_features[-num_validation_images:])
        valid_target_labels.extend(target_labels[-num_validation_images:])

    # Preprocessing and persisting the validationi subset
    #预处理并持久化validationi子集
    input_features = normalize_images(np.array(valid_input_features))
    target_labels = one_hot_encode(np.array(valid_target_labels))

    pickle.dump((input_features, target_labels), open('./preprocess/preprocess_valid.p', 'wb'))

    # Now it's time to preporcess and persist the test batche
    #现在是时候预处理并持久化测试批次了
    with open(cifar10_batches_dir_path + '/test_batch', mode='rb') as file:
        test_batch = pickle.load(file, encoding='latin1')

    test_input_features = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_input_labels = test_batch['labels']

    # Normalizing and encoding the test batch
    #对测试批处理进行规范化和编码
    input_features = normalize_images(np.array(test_input_features))
    target_labels = one_hot_encode(np.array(test_input_labels))

    pickle.dump((input_features, target_labels), open('./preprocess/preprocess_test.p', 'wb'))

# Calling the helper function above to preprocess and persist the training, validation, and testing set
#调用上面的辅助函数来预处理并持久化训练，验证和测试集
preprocess_persist_data(cifar10_batches_dir_path, normalize_images, one_hot_encode)
```
&emsp;&emsp;我们将预处理后的数据保存到磁盘。我们还需要在训练过程的不同时期加载用于在其上运行训练模型的验证集：<br>
```
# Load the Preprocessed Validation data
#加载预处理的验证数据
valid_input_features, valid_input_labels = pickle.load(open('./preprocess/preprocess_valid.p', mode='rb'))
```
### 建立网络
&emsp;&emsp;现在是时候构建我们的分类应用程序的核心，这是该CNN架构的计算图，但是为了最大化这种实现的好处，我们不会使用TensorFlow层API。 相反，我们将使用它的TensorFlow神经网络版本。<br>
<br\>
&emsp;&emsp;因此，让我们首先定义模型输入占位符，它将输入图像，目标类和丢失层的保持概率参数（这有助于我们通过删除一些连接来降低架构的复杂性，从而减少机会的可能性过拟合）：<br>
```
# Defining the model inputs
#定义模型输入
def images_input(img_shape):
    return tf.placeholder(tf.float32, (None,) + img_shape, name="input_images")


def target_input(num_classes):
    target_input = tf.placeholder(tf.int32, (None, num_classes), name="input_images_target")
    return target_input


# define a function for the dropout layer keep probability
#为dropout层定义一个保持概率的函数
def keep_prob_input():
    return tf.placeholder(tf.float32, name="keep_prob")

tf.reset_default_graph()
```
&emsp;&emsp;接下来，我们需要使用TensorFlow神经网络实现版本来构建具有最大池的卷积层：<br>
```
# Applying a convolution operation to the input tensor followed by max pooling
#将卷积运算应用于输入张量，然后进行最大池化
def conv2d_layer(input_tensor, conv_layer_num_outputs, conv_kernel_size, conv_layer_strides, pool_kernel_size,
                 pool_layer_strides):
    input_depth = input_tensor.get_shape()[3].value
    weight_shape = conv_kernel_size + (input_depth, conv_layer_num_outputs,)

    # Defining layer weights and biases
    #定义图层权重和偏差
    weights = tf.Variable(tf.random_normal(weight_shape))
    biases = tf.Variable(tf.random_normal((conv_layer_num_outputs,)))

    # Considering the biase variable
    #考虑偏差变量
    conv_strides = (1,) + conv_layer_strides + (1,)

    conv_layer = tf.nn.conv2d(input_tensor, weights, strides=conv_strides, padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, biases)

    conv_kernel_size = (1,) + conv_kernel_size + (1,)

    pool_strides = (1,) + pool_layer_strides + (1,)

    pool_layer = tf.nn.max_pool(conv_layer, ksize=conv_kernel_size, strides=pool_strides, padding='SAME')

    return pool_layer
```
&emsp;&emsp;正如您在上一章中可能看到的，最大池操作的输出是一个4D张量，它与完全连接层所需的输入格式不兼容。因此，我们需要实现一个平化层，将最大池化层的输出从4D转换为2D张量:<br>
```
#Flatten the output of max pooling layer to be fing to the fully connected layer which only accepts the output
# to be in 2D
#将最大池层的输出展平为仅接受输出的完全连接层
#在2D中
def flatten_layer(input_tensor):

    return tf.contrib.layers.flatten(input_tensor)
```
&emsp;&emsp;接下来，我们需要定义一个辅助函数，使我们能够在我们的架构中添加一个完全连接的层：<br>
```
#Define the fully connected layer that will use the flattened output of the stacked convolution layers
#to do the actuall classification
#定义将使用堆叠卷积层的展平输出的完全连接层。
#进行实际分类
def fully_connected_layer(input_tensor, num_outputs):
    return tf.layers.dense(input_tensor, num_outputs)
```
&emsp;&emsp;最后，在使用这些辅助函数创建整个体系结构之前，我们需要创建另一个函数来获取完全连接层的输出，并生成与我们在数据集中具有的类数相对应的10个实值：<br>
```
#Defining the output function
#定义输出函数
def output_layer(input_tensor, num_outputs):
    return  tf.layers.dense(input_tensor, num_outputs)
```
