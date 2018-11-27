# 目标检测-使用CNNs进行转移学习
&emsp;"个体如何从一个环境转移到另一个具有相似特征的环境"<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;——E.L. Thorndike,R.S. Woodworth (1991) <br>
&emsp;&emsp;**迁移学习(TL)** 是数据科学中的一个研究问题，它主要是将在解决某一特定任务时所获得的知识持久化，然后利用这些知识来解决另一个不同但又相似的任务。在这一章中，我们将展示使用TL在数据科学领域中使用的一种现代实践和常见主题。最后，我们将回顾CIFAR-10的对象检测示例，并尝试减少训练时间和性能错误。<br>
**本章将讨论以下主题:**<br>
&emsp;&emsp;**1. 迁移学习**<br>
&emsp;&emsp;**2. 再次回顾CIFAR-10的目标检测**<br>
## 迁移学习
&emsp;&emsp;深度学习架构是数据贪婪的，在一个训练集中有一些样本并不能让我们得到最好的结果。TL通过将从使用大数据集解决任务中学到的或获得的知识,表示形式转移到使用较小数据集解决任务的另一个不同但类似的任务来解决这个问题。<br>
&emsp;&emsp;TL不仅适用于小训练集，而且可以使训练过程更快。从头开始训练大型深度学习体系结构有时会非常缓慢，因为这些体系结构中有数百万个需要学习的权重。相反，人们可以使用TL，只需要对一个与他/她试图解决的问题类似的问题微调一个有学问的权重。<br>
## TL背后的直觉
&emsp;&emsp;让我们通过以下师生类比来建立TL背后的直观理解。一个老师在他所教的模块中有多年的经验。另一方面，学生们从这位老师的讲课中对主题有了一个紧凑的概述。所以你可以说，老师是以一种简明扼要的方式向学生传授知识的。<br>
&emsp;&emsp;老师和学生的类比同样适用于我们在深度学习或神经网络中传递知识的例子。因此，我们的模型从数据中学习了一些表示，这些数据由网络的权值表示。这些学习到的表征/特征(权重)可以转移到另一个不同但相似的任务中。将学习到的权重传递到另一个任务的过程，将减少深度学习体系结构收敛所需的海量数据集，也将减少模型适应新比较的数据集所需的时间。<br>
&emsp;&emsp;目前，深度学习被广泛使用，但通常大多数人在培训深度学习架构时使用TL;它们中很少有人从头开始培训深度学习架构，因为大多数情况下，很少有足够大的数据集来满足深度学习的收敛。因此，在一个大数据集上使用一个预先训练的模型是非常常见的，例如ImageNet，它有大约120万张图像，并将其应用到您的新任务中。我们可以使用预先训练的模型的权重作为特性提取器，或者我们可以用它初始化我们的架构，然后根据您的新任务对其进行微调:<br>
&emsp;&emsp;**使用卷积网络作为固定的特征提取器:** 在这个场景中，您在一个大型数据集(如ImageNet)上使用一个经过预处理的卷积模型，并将其用于解决您的问题。例如，在ImageNet上的预训练卷积模型将具有一个完整的连接层，该层具有ImageNet所具有的1,000个类别的输出分数。所以您需要删除这一层，因为您不再对ImageNet类感兴趣了。然后，您将所有其他层视为一个特征提取器。一旦您使用预先训练的模型提取了这些特性，您就可以将这些特性提供给任何一个。<br>
&emsp;&emsp;**微调卷积神经网络:** 第二个场景涉及第一个场景，但是需要额外的努力，使用反向传播来微调新任务上预先训练的权重。通常，人们保持大多数层固定，只微调网络的顶端。尝试微调整个网络，甚至大部分的层可能导致过拟合。因此，您可能只对那些与图像的语义级特性有关的层进行微调感兴趣。让早期的图层保持固定的直觉是，它们包含了一般的或者低级的专长。<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/one.png?raw=true)  <br>
&emsp;&emsp;**预先培训的模型:** 第三种广泛使用的场景是下载人们在internet上提供的检查点。如果您没有强大的计算能力来从头开始训练模型，那么您可以采用这种场景，因此您只需使用发布的检查点初始化模型，然后进行一些微调。<br>
## 传统机器学习与TL的区别
&emsp;&emsp;正如您在前一节中注意到的，我们应用机器学习的传统方法与涉及TL的机器学习方法之间有明显的区别(如下图所示)。在传统的机器学习中，你不需要将任何知识或表现形式转移到任何其他任务上，而在TL中却不是这样，有时人们使用TL的方式是错误的，所以我们将提到一些条件，在这些条件下，你只能使用TL来实现收益的最大化。<br>
用TL的条件如下:<br>
&emsp;&emsp;与传统机器学习不同，源和目标任务或领域不必来自相同的分布，但它们必须相似<br>
&emsp;&emsp;您还可以在训练样本较少或没有必要的计算能力的情况下使用TL<br>
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Two.png?raw=true)  <br>
## CIFAR-10目标检测“重访”
&emsp;&emsp;在上一章中，我们对CIFAR-10数据集训练了一个简单的卷积神经网络(CNN)模型。在这里，我们将演示使用预训练模型作为特征提取器，同时删除预训练模型的完全连接层的情况，然后我们将这些提取的特征或传输值提供给softmax层。<br>
&emsp;&emsp;这个实现中的预培训模型将是inception模型，它将在ImageNet上进行预培训。但是请记住，这个实现是在介绍CNN的前两章的基础上实现的。<br>
### 方案概要
&emsp;&emsp;同样，我们将替换预先训练的先启模型的最后一个完全连接的层，然后使用先启模型的其余部分作为特征提取器。因此，我们首先在inception模型中提供原始图像，它将从这些图像中提取特性，然后输出所谓的传输值。<br>
&emsp;&emsp;在获得从inception模型中提取的特性的传输值之后，您可能需要将它们保存到您的办公桌上，因为如果您是动态地执行这些操作，那么将会花费一些时间，因此将它们保存到您的办公桌上以节省时间是非常有用的。在TensorFlow教程中，他们使用术语瓶颈值而不是传输值，但它只是完全相同事物的不同名称。<br>
&emsp;&emsp;在获取传输值或从桌面加载它们之后，我们可以将它们提供给任何为新任务定制的线性分类器。在这里，我们将提取的传输值输入另一个神经网络，然后针对CIFAR-10的新类进行培训。<br>
如下图所示，为我们将要进行的通解大纲:<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Three.png?raw=true) <br>
## 装载和探索CIFAR-10
&emsp;&emsp;让我们从导入这个实现所需的包开始:<br>
```python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
# 为调用Inception模形先导入一个模块。
import inception
```
&emsp;&emsp;接下来,我们需要加载另一个辅助脚本,我们可以使用它来下载处理cifar 10数据集:<br>
```python
import cifar10
from cifar10 import num_classes
```
如果您还没有这样做，则需要为CIFAR-10设置路径。cifar-10.py脚本将使用此路径保存数据集:
```python
cifar10.data_path = "data/CIFAR-10/"
```
输出：
```python
The CIFAR-10 dataset is about 170MB, the next line checks if the dataset is alredy downloaded if not it downloads the dataset and store in the previous ata_path:
```
```python
cifar10.maybe_download_and_extract()
```
输出:
```python
- Download progress: 100.0%
Download finished.Extracting files.
Done.
```
让我们看看CIFAR-10数据集中的类别:
```python
#下载名为CIFAR-10的数据集
class_names = cifar10.load_class_names()
```
输出:
```python
Loading data: data/CIFAR-10/cifar-10-batches-py/batches.meta
['airplane',
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck']
Load the training-set.
```
这些返回的值里面，class-numbers 是整型的，那么我们接下来就要对其进行One-Hot编码处理形成lables：
```python
training_images, training_cls_integers, trainig_one_hot_labels = cifar10.load_training_data()
cifar10.load_training_data()
```
输出：
```python
Loading data: data/CIFAR-10/cifar-10-batches-py/data_batch_1
Loading data: data/CIFAR-10/cifar-10-batches-py/data_batch_2
Loading data: data/CIFAR-10/cifar-10-batches-py/data_batch_3
Loading data: data/CIFAR-10/cifar-10-batches-py/data_batch_4
Loading data: data/CIFAR-10/cifar-10-batches-py/data_batch_5
```
现在，让我们对测试集做同样的事情，加载目标类的图像及其对应的整数表示，使用One-Hot编码:
```python
testing_images, testing_cls_integers, testing_one_hot_labels = cifar10.load_test_data()
cifar10.load_test_data()
```
输出：
```python
Loading data: data/CIFAR-10/cifar-10-batches-py/test_batch
```
让我们来看看CIFAR-10中培训和测试集的分布:
```python
print("-Number of images in the training set:\t\t{}".format(len(training_images)))
print("-Number of images in the testing set:\t\t{}".format(len(testing_images)))
```
输出：
```python
-Number of images in the training set:		50000
-Number of images in the testing set:		10000
```
让我们定义一些帮助函数，使我们能够探索数据集。下面的辅助函数在网格中绘制了一组九幅图像:
```python
def plot_imgs(imgs, true_class, predicted_class=None):
    assert len(imgs) == len(true_class)

    # 为9个字节创建一个占位符
    fig, axes = plt.subplots(3, 3) 

    # 调整间距
    if predicted_class is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # 可能只有不到9张图片，确保不会崩溃
        if i < len(imgs):
            # 画图
            ax.imshow(imgs[i],
                      interpolation='nearest')

            # 从class_names数组中获取真实类的实际名称
            true_class_name = class_names[true_class[i]]

            #显示预测类和真实类的标签
            if predicted_class is None:
                xlabel = "True: {0}".format(true_class_name)
            else:
                # 预测类的名称
                predicted_class_name = class_names[predicted_class[i]]

                xlabel = "True: {0}\nPred: {1}".format(true_class_name, predicted_class_name)

            ax.set_xlabel(xlabel)

        # 从图画中移除ticks.
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
```
让我们继续，将来自测试集的一些图像及其对应的实际类可视化:
```python
# 获得测试集中的前9个图像
imgs = testing_images[0:9]

#获得真实类的整数表现形式
true_class = testing_cls_integers[0:9]

#画图
plot_imgs(imgs=imgs, true_class=true_class)
```
输出：

 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Four.png?raw=true)  






（这里是237页第8行，这句话用来标记已经翻译到的位置，下次再续写的话记得删除这句话）
 
 
 
 
 
 
 
 
 

