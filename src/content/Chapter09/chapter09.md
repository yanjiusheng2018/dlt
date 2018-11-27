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






（这里是240页第Incepthon Model上面行，这句话用来标记已经翻译到的位置，下次再续写的话记得删除这句话）
 
 
 （这里是240页第Incepthon Model上面行，这句话用来标记已经翻译到的位置，下次再续写的话记得删除这句话）
 ## 先启模型传输值
 &emsp;&emsp;如前所述，我们将在ImageNet数据集上使用预先训练好的初始化模型。所以，我们需要从互联网上下载这个预先训练好的模型:<br>
 &emsp;&emsp;让我们首先为inception模型定义<br>
 ```python
#下载预训练的inception v3模型
inception.maybe_download()
```
 &emsp;&emsp;预训练的先启模型的权重约为85MB，如果在之前定义的EBUB@EJS中不存在，下面的代码行将下载它:
  ```python
inception.maybe_download() 
Downloading Inception v3 Model ...
-download progress:100%
```
&emsp;&emsp;我们将加载inception模型，这样我们就可以把它作为CIFAR-10图像的特征提取器:
#我们将加载初始模型，以便将其用作CIFAR-10图像的特征提取器 
#这样我们就可以用预先训练过的权重将其最小化，并为我们的模型定制
 ```python
inception_model = inception.Inception()
```
&emsp;&emsp;正如我们前面提到的，计算CIFAR-10数据集的传输值需要一些时间，因此我们需要缓存它们以备将来使用。值得庆幸的是，JODFQUJPO模块中有一个帮助函数可以帮助我们做到这一点:
```python
from inception import transfer_values_cache
```
&emsp;&emsp;接下来，我们需要为缓存的培训和测试文件设置文件路径
```python
file_path_train = os.path.join(cifar10.data_path, 
                               'inception_cifar10_train.pkl')
file_path_test = os.path.join(cifar10.data_path, 
                              'inception_cifar10_test.pkl')
print(file_path_train,file_path_test,sep='\n')
print("Cifar-10训练图像的初始化处理转换值…")
#首先，我们需要缩放img来满足初始模型的要求，因为模型要求所有像素都在0到255之间，
#我们的CIFAR-10像素的训练示例在0.0到1.0之间
imgs_scaled = training_images * 255.0
#下面这个函数是检查我们的训练图像的传输值是否已经计算并加载，如果没有则计算并保存它们
transfer_values_training = transfer_values_cache(cache_path=file_path_train,
                                              images=imgs_scaled,
                                              model=inception_model)
#训练数据相同，首先，我们需要缩放img来满足初始模型的要求，要所有像素都在0到255之间，
#我们的CIFAR-10像素的训练示例在0.0到1.0之间
imgs_scaled = testing_images * 255.0
#检查我们的训练图像的传输值是否已经计算并加载，如果没有则计算并保存它们。
transfer_values_testing = transfer_values_cache(cache_path=file_path_test,
                                             images=imgs_scaled,
                                             model=inception_model)                                            
```
&emsp;&emsp;如前所述，我们在CIFAR-10数据集的训练集中有50,000张图像。我们来检查一下这些图像的传输值的形状。训练集中的每个图像应该是2048:
```python
transfer_values_training.shape
```
&emsp;&emsp;输出:
```python
(50000, 2048)
```
&emsp;&emsp;我们需要对测试集做同样的事情:
```python
transfer_values_training.shape
```
&emsp;&emsp;输出:
```python
(10000, 2048)
```
&emsp;&emsp;为了直观地理解转换值是什么样子的，我们将定义一个辅助函数，使我们能够使用来自训练或测试集的特定图像的转换值的图形:
```python
def plot_transferValues(ind):
    print("原始输入图像:")

    # 在测试集的索引处绘制图像.
    plt.imshow(testing_images[ind], interpolation='nearest')
    plt.show()

    print("使用Inception model传输值:")

    #将传输值可视化为图像.
    transferValues_img = transfer_values_testing[ind]
    transferValues_img = transferValues_img.reshape((32, 64))

    # 绘制传输值图像
    plt.imshow(transferValues_img, interpolation='nearest', cmap='Reds')
    plt.show()
    plot_transferValues(i=16)
```
&emsp;&emsp;输入图像:<br>
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Five.png?raw=true) <br>
&emsp;&emsp;使用先启模型传输图像的值:<br>
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Eight.png?raw=true) <br>  
```python 
plot_transferValues(i=17)
```
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Seven.png?raw=true) <br>
&emsp;&emsp;使用先启模型传输图像的值:<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Six.png?raw=true) <br>  

## 转让值分析
&emsp;&emsp;在这一节中，我们将对刚刚得到的训练图像的传输值进行一些分析。这个分析的目的是看看这些传输值是否足以对我们在CIFAR-10中的图像进行分类。

&emsp;&emsp;对于每个输入图像，我们有2048个传输值。为了绘制这些传递值并对其进行进一步分析，我们可以使用来自scikit-learn的主成分分析(PCA)等降维技术。我们将把传输值从2048减少到2，以便能够直观地看到它，并看看它们是否能够成为区分不同类别CIFAR-10的好特性:
```python
from sklearn.decomposition import PCA
```
&emsp;&emsp;接下来，我们需要创建一个PCA对象，其中组件的数量只有2个
```python
pca_obj = PCA(n_components=2)
```
&emsp;&emsp;将传输值从2048减少到2需要很多时间，所以我们将在5000张具有传输值的图像中只选取3000张作为子集:
```python
subset_transferValues = transfer_values_training[0:3000]
```
&emsp;&emsp;我们还需要得到这些图像的类号:
```python
cls_integers = testing_cls_integers[0:3000]
```
&emsp;&emsp;我们可以通过打印传输值的形状来再次检查我们的子设置:接下来，我们使用PCA对象将传输值从2048减少到2:
```python
subset_transferValues.shape
```
&emsp;&emsp;输出:
```python
(3000, 2048)
```
&emsp;&emsp;接下来，我们使用PCA对象将传输值从2048减少到2:
```python
reduced_transferValues = pca_obj.fit_transform(subset_transferValues)
```
&emsp;&emsp;现在，让我们看看PCA约简过程的输出:
```python
reduced_transferValues.shape
```
&emsp;&emsp;输出:
```python
(3000, 2)
```
&emsp;&emsp;将传递值的维数降为2后，将这些值作图:
#导入颜色映射图，以用不同的颜色绘制每个类。
```python
import matplotlib.cm as color_map
```
```python
def plot_reduced_transferValues(transferValues, cls_integers):
    #为每一个类创建一个颜色不同的颜色图
    c_map = color_map.rainbow(np.linspace(0.0, 1.0, num_classes))

    # 获得每个样本的颜色
    colors = c_map[cls_integers]

    # 获得X,y的值
    x_val = transferValues[:, 0]
    y_val = transferValues[:, 1]

    # 在散点图中绘制传输值
    plt.scatter(x_val, y_val, color=colors)
    plt.show()
```
&emsp;&emsp;这里，我们绘制的是训练集子集的约简传递值。在CIFAR-10中，我们有10个类，所以我们要用不同的颜色绘制它们对应的传递值。从下图可以看出，传输值是根据相应的类分组的。组与组之间的重叠是由于主成分分析的约简过程不能很好地分离转移值:
```python
plot_reduced_transferValues(reduced_transferValues, cls_integers)
```
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Ten.png?raw=true)   
&emsp;&emsp;我们可以使用不同的降维方法t-SNE对传递值进行进一步的分析:
```python
from sklearn.manifold import TSNE
```
&emsp;&emsp;再一次，我们将降低传递值的维度，也就是2048，但是这次是50个值，而不是2:
```python
pca_obj = PCA(n_components=50)
transferValues_50d = pca_obj.fit_transform(subset_transferValues)
```
&emsp;&emsp;接下来，我们将第二维度约简技术叠加起来，将主成分分析过程的输出反馈给它:
```python
tsne_obj = TSNE(n_components=2)
```
&emsp;&emsp;最后，我们使用PCA方法的约简值，并将t-SNE方法应用于它:
```python
reduced_transferValues = tsne_obj.fit_transform(transferValues_50d) 
```
&emsp;&emsp;输出:
```python
(3000, 2)
```
&emsp;&emsp;让我们用t-SNE方法画出减少的传输值。正如您在下一幅图像中看到的，t-SNE能够比PCA更好地分离分组传输值。
&emsp;&emsp;从这个分析中得出的结论是，我们通过将输入图像输入到预先训练的初始化模型中得到的提取的传输值可以用于将训练图像分离到10个类中。这种分离并不是100%准确的，因为在下面的图中有少量的重叠，但是我们可以通过对我们预先训练的模型做一些微调来消除这种重叠:
```python
plot_reduced_transferValues(reduced_transferValues, cls_integers)
```
 
 
 
 
# 金灵大大写的 <br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Ten.png?raw=true) <br>
&emsp;&emsp;现在我们有了从训练图像中提取的传输值，我们知道这些值在某种程度上能够区分CIFAR-10的不同类。接下来，我们需要构建一个线性分类器，并将这些传递值提供给它来执行实际的分类。<br>
## 建模和训练
&emsp;&emsp;因此，让我们从指定输入占位符变量开始，这些变量将被输入到我们的神经网络模型中。第一个输入变量的形状（将包含提取的传输值）将是[None,transfer_len]。第二占位符变量将以一个热向量格式保存训练集的实际类标签。<br>
```python
transferValues_arrLength = inception_model.transfer_len
input_values = tf.placeholder(tf.float32,shape=[None,transferValues_arrLength],name=’input_values’)
y_actual = tf.placeholder(tf.float32,shape=[None,num_classes],name=’y_actual’)
```
第249页
&emsp;&emsp;我们还可以通过定义另一个占位符变量，得到每个类的对应整数值，从1到10: <br>
```python
y_actual_cal = tf.argmax(y_actual,axis=1)
```
&emsp;&emsp;接下来，我们需要构建一个实际的分类神经网络来获取这些输入占位符并生成预测类: <br>
```python
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))
def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):
#参数1：前一层；参数2：Num.inputs from prev.layer；参数3：Num.outputs;
#参数4:采用整流线形单元
    #创建新权重和偏差值
    weights = new_weights(shape=[num_inputs,num_outputs])
    biases = new_biases(length=num_outputs)
    #将层计算为输入和权重的矩阵乘法，并添加偏值
    layer = tf.matmul(input,weights)+biases
    #使用relu?
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
#第一个全连接层：
layer+fc1 = new_fc_layer(input=input_values,num_inputs=2048,
num_outputs=1024,use_relu=True)
#第二个全连接层：
layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=1024,
num_outputs=num_classes,use_relu=False)
#预测类标签
y_predicted = tf.nn.softmax(layer_fc2)
第250页
#为了每个图像分类的交叉熵
cross_entropy = \
tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_actual)
#损失 aka. 成本计量
#这是必须最小化的标量值
loss = tf.reduce_mean(cross_entropy)
```

&emsp;&emsp;然后，我们需要定义一个优化准则，用于分类器的训练。在这个实现中，我们将使用AdamOptimizer。这个分类器的输出将是一个10个概率得分的数组，对应于CIFAR-10数据集中的类数。然后，我们将对这个数组应用argmax操作，将最大得分的类分配给这个输入示例: <br>
```python
step = tf.Variable(initial_value=0,name=’step’,trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss,step)
y_predicted_cls = tf.argmax(y_predicted,axis=1)
#比较预测类和真实类
correct_prediction = tf.equal(y_predicted_cls,y_actual_cls)
#将boolearn值转换为float
model_accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```
&emsp;&emsp;接下来，我们需要定义一个将实际执行图形的TensorFlow会话，然后初始化我们在此实现中早先定义的变量: <br>
```python
session = tf.Session()
session.run(tf.global_variables_initializer())
```
&emsp;&emsp;在这个实现中，我们将使用随机梯度下降(SGD)，因此我们需要定义一个函数来从50,000张图像的训练集中随机生成特定大小的批。<br>
&emsp;&emsp;因此，我们将定义一个助手函数，用于从传输值的输入训练集生成一个随机批处理:<br>
```python
#定义训练批次的大小
train_batch_size = 64
#定义一个从数据集中随机选择一批图像的函数
def select_random_batch():
    #训练集中的图像数(传输值)
    num_imgs = len(transfer_values_training)
    #创建一个随机8索引

第251页
ind = np.random.choice(num_imgs,size=training_batch_size,replace=False) 
#使用随机索引去选择一些随机的x和y的值
#我们用transfer-values 代替 图像作为 x-values
x_batch = transfer_values_training[ind]
y_batch = training_one_hot_labels[ind]
    return x_batch,y_batch
```
&emsp;&emsp;接下来，我们需要定义一个助手函数来执行实际的优化过程，这将细化网络的权重。它将在每次迭代中生成一个批处理，并根据该批处理优化网络: <br>
```python
def optimize(num_iterations):
    for i in range(num_iterations):
        #随机的选一批图像去训练
        #在input_batch中存储图像传输值
        #其他图像的真实值将会储存在y_actual_batch
        input_batch,y_actual_batch = select_random_batch()
        #存储这批数据在一个字典里再给它取一个适当的名字
        #像我们在上面定义且输入的占位符变量
        feed_dict = {input_values:input_batch,y_actual:y_actual_batch}
        #现在我们调用这批图像的优化器
        #TensorFlow 将自动提供我们上面创建的dict的值
        #在我们上面定义的模型输入占位符变量
        i_global,_=session.run([step,optimizer],feed_dict=feed_dict)
        #打印每100步的精度
        if (i_global % 100 == 0) or (I == num_iterations - 1):
            #计算在training_batch中的精确值
            batch_accuracy = session.run(model_accuracy,feed_dict=feed_dict)
            msg = “step:{0:>6},Training Accuracy:{1:>6.1%}”
            print(mag.format(i_global,batch_accuracy))
```

第252页
&emsp;&emsp;我们将定义一些辅助函数来显示前一个神经网络的结果，并显示预测结果的混淆矩阵: <br>
```python
def plot_errors(cls_predicted,cls_correct):
    #cls_predicted 是一个数组它里面是测试集中所有图像的预测类
    #cls_correct 是一个具有布尔值的数组，用于指示模型是否预测了正确的类
    
    #否定布尔数组
    incorrect = (cls_correct == False)
    #从测试集中获得图片这是错误的分类
    incorrectly_classified_images = testing_images[incorrect]
    #从那些图片中获得预测类
    cls_predicted = cls_predicted[incorrect]
    
    #从那些图像获得真实的类
    true_class = testing_cls_integers[incorrect]

    n = min(9,len(incorrectly_classified_images))
    #画第一个 n 图像
    plot_imgs(imgs=incorrectly_classified_images[0:n],true_class=true_class[0:n],
predicted_class=cls_predicted[0:n])
```

&emsp;&emsp;接下来，我们需要定义辅助函数来绘制混淆矩阵<br>
```python
from sklearn.metrics import confusion_matrix
def plot_confusionMatrix(cls_predicted):
    #所有预测值的cls_predicted 数组
    #测试中的类编号

    #从sklearn 调用混淆矩阵
    cm =confusion_matrix(y_true=testing_cls_integers,y_pred=cls_predicted)
    #打印混淆矩阵
    for I in range(num_classes):
        #将类名加入每一排
        class_name = “({}) {}”.format(i,class_names[i])
        print(cm[I,:],class_name)
    #用类号标记混淆矩阵的每一列
    cls_numbers = [“ ({0})”.format(i) for i in range(num_classes)]
    print(“”.join(cls_numbers))
```
    
    第253页
&emsp;&emsp;另外，我们将定义另一个辅助函数来运行经过训练的分类器在测试集之上，并测量经过训练的模型在测试集之上的准确性: <br>
```python
#将数据集分批分类，以限制RAM的使用
batch_size = 128
def predict_class(transferValues,labels,cls_true):
    #图片的编号
    num_imgs = len(transferValues)
    #为预测类分配一个数组，这些类将分批计算并填充到这个数组中
    cls_predicted = np.zeros(shape=num_imgs,dtype=np.int)
    #现在计算批次的预测类，我们将遍历所有批次，也许有一种更聪明、更毕达哥拉斯
#的方法来解决这个问题
    
    #下一批的起始指标记为i
    i = 0
    while i < num_imgs:
        #下一批结束指标记为j
        j = min(i + batch_size,num_imgs)
        #用索引i 和j之间的图像和标签创建一个feed-dict
        feed_dict = {input_values:transferValues[i:j],y_actual:labels[i:j]}
        #使用TensorFlow计算预测类
        cls_predicted[i:j] = session.run(y_predicted_cls,feed_dict = feed_dict)
        #将下一批的开始索引设置为当前批的结束索引
        i = j
        #创建一个布尔数组，每一个图像是否正确分类
    correct = [a == p for a,p in zip(cls_true,cls_predicted)]
    return correct,cls_predicted
#调用之前的函数对test做预测
def predict_cls_test():
    return predict_class(transferValues = transfer_values_test,labels = labels_test,
cls_true = cls_test)
第254页
def classification_accuracy(correct):
    #定平均一个布尔数列时，False代表0，True代表1。所以我们在计算：True/len(correct) 的个数，这和分类精度是一样
的

    #返回这个分类精度，和正确分类数量
    return np.mean(correct),np.sum(correct)

def test_accuracy(show_example_errors=False,show_confusion_matrix=False)：
    #再测试集中的所有数据，计算预测类和是否他们正确
    correct,cls_pred = predict_class_test()
    #分类 accuracy predict_class_test 和正确分类数量
    accuracy,num_correct = classification_accuracy(correct)
    #正确分类图像的数量
    num_images = len(correct)

     #打印accuracy
    mag = “Test set accuracy:{0:.1%} ({1}/{2})”
    print(msg.format(accuracy,num_correct,num_images))
    #画一些mis-classifications的例子，若需要的话
    if show_example_errors:
        print(‘Example errors:’)
        plot_erroes(cls_predicted=cls_pred,cls_correct=correct)
    #画混淆矩阵，如果需要的话
    if show_confusion_matrix:
        print(“confusion matrix:”)
        plot_confusionMatrix(cls_predicted=cls_pred)
```
&emsp；&emsp；在进行任何优化之前，让我们看看前面的神经网络模型的性能: <br>
```python
test_accuracy(show_example_errors=True,show_confusion_matrix=True)
Accuracy on Test-Set:9.4%(939/10000)
```
第255页
&emsp;&emsp;正如您所看到的，网络的性能非常低，但是在基于我们已经定义的优化标准进行一些优化之后，它会变得更好。因此，我们将运行优化器进行10000次迭代，然后测试模型的准确性: <br>
```python
optimize(num_iterations=10000)
test_accuracy(show_example_errors=True,show_confusion_matrix=True)
Accuracy on Test-Set:90.7%(9069/10000)
```
 ![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Four.png?raw=true)  

confusion Matrix:
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter09/chapter09_image/Twelve.png?raw=true)  

第256页
&emsp;&emsp;最后，我们将结束开放的会议: <br>

```python
 model.close()
session.close()
```


## 摘要
&emsp;&emsp;在本章中，我们介绍了最广泛使用的深度学习最佳实践之一。TL是一个非常令人兴奋的工具，您可以使用它来获得深度学习体系结构，从您的小数据集中学习，但要确保以正确的方式使用它<br>
&emsp;&emsp;接下来，我们将介绍一种用于自然语言处理的广泛使用的深度学习体系结构。这些递归式架构在大多数NLP领域都取得了突破:机器翻译、语音识别、语言建模和情绪分析。<br>
 
 
 
 

