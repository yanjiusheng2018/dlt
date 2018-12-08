# &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;13、自动编码器-特征提取和降噪

&emsp;&emsp;自动编码器网络是当前广泛应用的深度学习体系结构之一。 主要用于高效解码任务的无监督学习。它还可以通过学习编码或特定数据集的表示来进行降维。在本章中，我们将使用autoencoders，通过构造另一个具有相同尺寸但噪音更小的数据集来减少数据集的噪音。为了在实践中使用这个概念，我们将从MNIST数据集中提取重要的特性，并尝试了解如何这将显著提高性能。

本章将讨论以下主题:

- **自动编码器介绍**
- **自动编码器的示例**
- **自动编码器架构**
- **压缩MNIST数据集**
- **卷积式自动编码器**
- **去噪自动编码器**
- **自动编码器的应用**

## 自动编码器的介绍
&emsp;&emsp;自动编码器是另一种可用于许多有趣任务的深度学习体系结构，但它也可以看作是普通前馈神经网络的变体，其中输出具有与输入相同的维度。如图1所示，自动编码器的工作方式是将数据样本(x1，…，x6)发送到网络。它将尝试学习L2层中该数据的较低表示形式，您可以将其称为以较低表示形式编码数据集的方法。然后，网络的第二部分(您可能称之为解码器)负责从该表示构造输出。您可以将网络从输入数据中学到的中间低层表示形式看作是它的压缩版本。
与我们目前看到的所有其他深度学习架构没有太大区别，自动编码器使用反向传播算法。

&emsp;&emsp;与我们目前看到的所有其他深度学习架构没有太大区别，自动编码器使用反向传播算法。

&emsp;&emsp;自编码器神经网络是一种应用反向传播的无监督学习算法，其目标值等于输入:
![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/src/content/Chapter13/chapter13_image/%E5%9B%BE1.jpg)

## 自编码的例子
&emsp;&emsp;在本章中，我们将展示一些使用MNIST数据集的自动编码器的不同变体的例子。例如，假设输入x为28×28图像(784像素)的像素强度值;输入数据样本的个数是n=784。在L2层有s2=392个隐藏单元。由于输出和输入数据样本的尺寸相同，y∈R784。输入层神经元数量为784个，中间层L2神经元数量为392个;所以网络将是一个较低的表示，这是一个压缩版本的输出。然后，网络将把输入a(L2) ∈R392压缩后的较低表示形式提供给网络的第二部分，后者将努力从这个压缩版本重构输入像素784。

&emsp;&emsp;自动编码器依赖于这样一个事实，即由图像像素表示的输入样本将以某种方式相互关联，然后它将利用这个事实来重构它们。因此，自动编码器有点类似于降维技术，因为它们还可以学习输入数据的更低表示形式。

&emsp;&emsp;综上所述，一个典型的自动编码器由三部分组成: 1.编码器部分，负责将输入压缩成较低的表示形式 2.代码，它是编码器的中间结果 3.解码器，它负责用这个代码重建原始输入

&emsp;&emsp;下图显示了一个典型的自动编码器的三个主要组件:
![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/src/content/Chapter13/chapter13_image/%E5%9B%BE%E4%BA%8C.jpg)
&emsp;&emsp;正如我们所提到的，自动编码器部分学习输入的压缩表示，然后将其输入到第三部分，第三部分试图重构输入。重构后的输入与输出类似，但与原始输出并不完全相同，因此自动编码器不能用于压缩任务。

## 自编码器的架构
&emsp;&emsp;正如我们所提到的，一个典型的自动编码器由三个部分组成。让我们更详细地探讨这三个部分。为了激励你们，我们不打算在这一章里重复发明轮子。编码器-解码器部分是一个完全连接的神经网络，而代码部分是另一个神经网络，但它没有完全连接。该代码部分的维度是可控的，我们可以把它当作一个超参数:
![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/src/content/Chapter13/chapter13_image/%E5%9B%BE%E4%B8%89.jpg)

&emsp;&emsp;在深入使用自动编码器压缩MNIST数据集之前，我们将列出一组超参数，我们可以使用它们来微调自动编码器模型。超参数主要有四个:<br>
**1.代码部分大小:** 这是中间层的单元数。这一层的单元数越少，我们得到的输入表示就越压缩。<br>
**2.编码器和解码器的层数:** 正如我们所提到的，编码器和解码器只是一个完全连接的神经网络，我们可以通过添加更多的层来尽可能深入地构建它。<br>
**3.每个层的单元数:** 我们也可以在每个层中使用不同数量的单元数。编码器和解码器的形状非常类似于反解码器，当我们接近代码部分时，编码器中的层数会减少，然后在接近解码器的最后一层时开始增加。<br>
**4.模型损失函数:** 我们可以使用不同的损失函数，如MSE或交叉熵。<br>
在定义这些超参数并给出初始值之后，我们可以使用反向传播算法对网络进行训练。<br>
## 压缩MNIST数据集<br>
&emsp;&emsp;在这一部分中，我们将构建一个简单的自编码器，用于压缩MNIST数据集。因此，我们将把这个数据集的图像输入到编码器部分，编码器将尝试为它们学习一个较低的压缩表示;然后在解码器部分尝试重新构造输入图像。<br>
## MNIST数据集
&emsp;&emsp;我们将通过获取MNIST数据集，使用TensorFlow的辅助函数开始实现。
&emsp;&emsp;让我们导入这个实现所需的包:
```%matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist_dataset = input_data.read_data_sets("MNIST_data/",validation_size=0)
```
Output:
```Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
```
&emsp;&emsp;让我们从MNIST数据集的一些例子开始:
```#从训练集中选取一个图像画出来
image = mnist_dataset.train.images[2]
plt.imshow(image.reshape((28,28)),cmap = 'Greys_r')
```
Outpout:

![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%9C1.png)
```#从训练集中选取一个图像画出来
image = mnist_dataset.test.images[2]
plt.imshow(image.reshape((28,28)), cmap= 'Greys_r')
```
Output:

![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%9C2.png)
## 建立模型
&emsp;&emsp;为了构建编码器，我们需要计算出每个MNIST图像将有多少像素，这样我们就可以计算出编码器的输入层的大小。来自MNIST数据集的每张图像都是28×28像素，因此我们将把这个矩阵重塑为一个28×28 = 784像素值的向量。我们不需要标准化MNIST的图像因为它们已经标准化了。<br>
&emsp;&emsp;让我们开始构建模型的三个组件。在这个实现中，我们将使用一个非常简单的架构，即一个隐藏层，后面是ReLU激活 如下图所示:
![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/src/content/Chapter13/chapter13_image/%E5%9B%BE%E5%9B%9B.jpg)
&emsp;&emsp;让我们按照前面的解释来实现这个简单的编码器-解码器架构:
```#编码层或者隐藏层的大小
encoding_layer_dim = 32
img_size =  mnist_dataset.train.images.shape[1]
```
```#定义输入和目标值的占位符变量,
#img_size就是图像的大小，是28x28的
inputs_values = tf.placeholder(tf.float32,(None,img_size),name = "inputs_values")
targets_values = tf.placeholder(tf.float32,(None,img_size),name = "targets_values")
```
```#定义一个接受输入值并对其进行编码的编码层
encoding_layer = tf.layers.dense(inputs_values, encoding_layer_dim,activation = tf.nn.relu)
```
```#定义logit层，这是一个完全连接的层，但是它的输出没有任何激活函数
logits_layer = tf.layers.dense(encoding_layer, img_size, activation = None)
```
```#在logit层后面添加一个sigmoid层
decoding_layer = tf.sigmoid(logits_layer,name = "decoding_layer")
```
```#用sigmoid函数交叉熵作为一个损失函数
model_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_layer,labels=targets_values)
```
```#平均输入数据的损失值
model_cost = tf.reduce_mean(model_loss)
```
```#现在我们有了一个损失函数，我们需要使用Adam优化器来优化它
model_optimizier = tf.train.AdamOptimizer().minimize(model_cost)
```
&emsp;&emsp;现在我们已经定义了我们的模型并且使用了二进制交叉熵作为损失函数，并且图像值已经标准化了。<br>
## 模型训练
&emsp;&emsp;在本节中，我们将开始训练过程。我们将使用mnist_dataset对象的辅助函数，以便从具有特定大小的数据集中获得随机批处理;然后我们将对这批图像运行优化器。<br>
&emsp;&emsp;让我们从创建会话变量开始这一节，它将负责执行我们前面定义的计算图:<br>
```#创建会话
sess = tf.Session()
```
&emsp;&emsp;接下来，让我们开始训练过程:
```#现在让我们开始训练过程
num_epochs = 20 #执行20个周期
train_batch_size = 200 #每一批次200数据

sess.run(tf.global_variables_initializer())#初始化所有tensorflow全局变量
for e in range(num_epochs):  #使e循环遍历num_epcohs 即20个周期
    for ii in range(mnist_dataset.train.num_examples//train_batch_size): #使用60000项数据进行训练，分为每一批次200项数据，所以有300个批次
        input_batch = mnist_dataset.train.next_batch(train_batch_size)  #调用.next_batch函数，每次拿取200个数据进行训练
        feed_dict = {inputs_values: input_batch[0], targets_values:input_batch[0]} 
        input_batch_cost, _ = sess.run([model_cost, model_optimizier],feed_dict=feed_dict)# 运行这些代码    
        
        print("Epoch:{}/{}...".format(e+1, num_epochs),"Training loss: {:.3f}".format(input_batch_cost)) #保留小数点后三位
 ```
  Outpout:
  ```
  .
  .
  .
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.089
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.089
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.098
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.099
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.098
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.098
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.089
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.089
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.098
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.089
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.089
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.088
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.088
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.089
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.096
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.093
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.097
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.095
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.094
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.091
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.092
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.090
Epoch:20/20... Training loss: 0.091
```
&emsp;&emsp;在运行前面的代码片段20个epoch之后，我们将得到一个经过训练的模型，它能够从MNIST数据的测试集生成或重构图像。 请记住，如果我们输入的图像与模型所训练的图像不相似，那么重构过程就无法工作，因为自动编码器是特定于数据的<br>
&emsp;&emsp;让我们从测试集中获取一些图像，对训练好的模型进行测试，看看模型在解码器部分是如何进行重构的:
```fig, axes = plt.subplots(nrows=2, ncols = 10, sharex=True, sharey=True, figsize=(20,4))
input_images = mnist_dataset.test.images[:10]#先取出mnist训练图像的前十个值   
reconstructed_images, compressed_images = sess.run([decoding_layer, encoding_layer], 
feed_dict={inputs_values:input_images})
for imgs, row in zip([input_images, reconstructed_images], axes):
    for img, ax in zip(imgs, row):  
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(True) #True代表显示坐标轴，False代表隐藏坐标轴
        ax.get_yaxis().set_visible(True)

    fig.tight_layout(pad=0.1)#调整多个图之间的间隔来减少堆叠，pad值越大，间隔越大
 ```
 Output:
 ![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/src/content/Chapter13/chapter13_image/%E5%9B%BE%E4%BA%94.jpg)
 &emsp;&emsp;正如你所看到的，重构的图像与输入的图像非常接近，但是我们可以在编码器-解码器部分使用卷积层得到更好的图像<br>
 ## 卷积神经网络自动编码器
 &emsp;&emsp;前面的简单实现在尝试从MNIST数据集重构输入图像时做得很好，但是我们可以通过编码器中的卷积层和自动编码器的解码器部分获得更好的性能。这种替换的结果网络称为卷积自动编码器(CAE)。能够替换层的这种灵活性是自动编码器的一大优点，使它们适用于不同的领域。<br>
 &emsp;&emsp;我们将用于CAE的架构将在网络的解码器部分包含上采样层，以获得图像的重构版本。<br>
 
 ## 数据集
  &emsp;&emsp;在这个代码实现中，我们可以使用任何类型的成像数据集，看看如何自动编码器的卷积版本会有所不同。我们仍然会使用MNIST为此数据集，所以让我们从使用TensorFlow助手获取数据集开始:
  ```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```
 ```
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist_dataset = input_data.read_data_sets("MNIST_data/",validation_size=0)
```
Output：
```
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
```
&emsp;&emsp;让我们从一个数据集中选择一个数字
```
#从训练集中选取一个图像画出来
image = mnist_dataset.train.images[2]
plt.imshow(image.reshape((28,28)),cmap = 'Greys_r')
```
Output:
![image.png](https://raw.githubusercontent.com/yanjiusheng2018/dlt/master/%E4%BB%A3%E7%A0%81%E7%BB%93%E6%9E%9C1.png)
## 建立模型
&emsp;&emsp;在这个实现中，我们将使用步长为1的卷积层，pading = same。这样，我们就不会改变图像的高度或宽度。另外，我们使用一组最大池化层来减少图像的宽度和高度，从而构建图像的压缩较低的表示形式。<br>
&emsp;&emsp;所以让我们继续构建我们网络的核心:
```learning_rate = 0.001 #设置学习率为0.001
```
```#为输入和目标值定义占位符变量，1为黑白色
inputs_values = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_values')
targets_values = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_values')
```
```
#定义netowrk的编码器部分
##在编码器parrt中定义第一个卷积层
#输出的张量将会是28x28x16 
#让卷积运算产生的图像大小不变 激活函数为relu
conv_layer_1 = tf.layers.conv2d(inputs=inputs_values, filters=16, kernel_size=(3,3), padding='same', 
                                activation=tf.nn.relu)
```
```
#输出张量的形状为14x14x16   
maxpool_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=(2,2), strides=(2,2), padding='same')
```
```
#输出张量的形状为14x14x8
conv_layer_2 = tf.layers.conv2d(inputs=maxpool_layer_1, filters=8, kernel_size=(3,3),
                                padding='same', activation=tf.nn.relu)
```
```
#输出张量的形状为7x7x8 
maxpool_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=(2,2), strides=(2,2), padding='same')
```
```
#输出张量的形状为7x7x8
conv_layer_3 = tf.layers.conv2d(inputs = maxpool_layer_2, filters=8,kernel_size=(3,3),
                               padding='same',activation=tf.nn.relu)
```
```
#输出张量的形状为7x7x8
conv_layer_3 = tf.layers.conv2d(inputs = maxpool_layer_2, filters=8,kernel_size=(3,3),
                               padding='same',activation=tf.nn.relu)
```
```
#输出张量的形状为4x4x8，池化大小为2x2
encoded_layer = tf.layers.max_pooling2d(conv_layer_3, pool_size=(2,2), strides=(2,2),padding='same')
```
```
#定义第一个上采样层
#输出张量的形状为7x7x8
upsample_layer_1 = tf.image.resize_images(encoded_layer, size=(7,7), 
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
```
```
#输出张量的结果是7x7x8 
conv_layer_4 = tf.layers.conv2d(inputs=upsample_layer_1, filters=16, kernel_size=(3,3), 
                                padding='same', activation=tf.nn.relu)
```
```
#输出张量的形状为14x14x8 定义第二个上采样层
upsample_layer_2 = tf.image.resize_images(conv_layer_4, size=(14,14),
                                         method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
```
```
#输出张量的结果是14x14x8  定义第二个卷积层
conv_layer_5 = tf.layers.conv2d(inputs=upsample_layer_2, filters=8,
                               kernel_size = (3,3), padding='same', activation=tf.nn.relu)
```
```
#输出张量的形状为28x28x8 定义第三个上采样层
upsample_layer_3 =  tf.image.resize_images(conv_layer_4, size=(28,28),
                                         method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
```
```
#输出张量的形状为28x28x16 定义第三个卷积层
conv6 = tf.layers.conv2d(inputs=upsample_layer_3, filters=16, kernel_size=(3,3), padding='same', 
                        activation=tf.nn.relu)
```
```
#输出张量的形状为28x28x1
logits_layer = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3),padding='same', activation=None)
```
```
#将logits值赋给sigmoid激活函数函数，得到重建的图像
decoded_layer = tf.nn.sigmoid(logits_layer)
```
```
#把目标值和逻辑层喂给交叉熵，得到损失函数
model_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_values,logits=logits_layer)
```
```
#得到模型的损失并且定义优化器来优化
model_cost = tf.reduce_mean(model_loss)
model_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model_cost)
```
&emsp;&emsp;现在我们已经做得很好了，我们构建了卷积神经网络的解码器-解码器部分，同时展示了如何在解码器部分重建输入图像的尺寸。
## 模型训练
&emsp;&emsp;现在我们已经构建了模型，那么我们可以通过生成启动学习过程 随机批量形成MNIST数据集，并将它们提供给前面所定义的优化器。<br>
&emsp;&emsp;让我们从创建会话变量开始吧，它将负责执行我们之前所定义的计算图:<br>
```
sess = tf.Session()
num_epochs = 20 
train_batch_size = 200 
sess.run(tf.global_variables_initializer()) 

for e in range(num_epochs): #e遍历20个周期
    for ii in range(mnist_dataset.train.num_examples//train_batch_size):
        input_batch = mnist_dataset.train.next_batch(train_batch_size) 
        input_images = input_batch[0].reshape((-1,28,28,1))
        input_batch_cost, _ =sess.run([model_cost,model_optimizer],
        feed_dict={inputs_values:input_images,targets_values:input_images}) 
        
        print("Epoch: {}/{}...".format(e+1, num_epochs),
               "Training loss: {:.3f}".format(input_batch_cost))
```
Output:
```
.
.
.
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.103
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.103
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.093
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.104
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.103
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.103
Epoch: 20/20... Training loss: 0.094
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.105
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.103
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.096
Epoch: 20/20... Training loss: 0.102
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.095
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.099
Epoch: 20/20... Training loss: 0.100
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.103
Epoch: 20/20... Training loss: 0.098
Epoch: 20/20... Training loss: 0.101
Epoch: 20/20... Training loss: 0.097
Epoch: 20/20... Training loss: 0.099
```
&emsp;&emsp;在运行前面的代码片段20个阶段后，我们可以训练得到CAE， 所以让我们继续测试这个模型，从MNIST数据集中输入类似的图像<br>
```
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
input_images = mnist_dataset.test.images[:10] #先取出mnist训练图像的前十个值  
reconstructed_images = sess.run(decoded_layer, 
                                feed_dict={inputs_values:input_images.reshape((10,28,28,1))})
 
for imgs, row in zip([input_images, reconstructed_images], axes): 
    for img, ax in zip(imgs, row):#然后用img遍历imgs
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)#True代表显示坐标轴，False代表隐藏坐标轴
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
```
Output:
![image.png]()
