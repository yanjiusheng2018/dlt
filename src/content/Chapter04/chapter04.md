## 计算图
&emsp;&emsp;关于TensorFlow的所有想法中最大的想法是，数值计算被表示为计算图，如下图所示。因此，任何TensorFlow程序的主干将是一个计算图，下面的说法是正确的：<br>
&emsp;&emsp;1.图中的节点是具有任意数量的输入和输出的操作。<br>
&emsp;&emsp;2.节点之间的图边将是在这些操作之间流动的张量，并且认为张量在实践中的最佳方式是n维数组。<br>
&emsp;&emsp;使用流程图作为深度学习框架骨干的优点在于，它允许您根据小而简单的操作构建复杂模型。此外，这将使梯度计算非常简单，我们会在后面的部分中提到这一点。<br>
&emsp;&emsp;另外一种考虑TensorFlow计算图的方法是每一个操作是一个可以在那个点进行评估的函数。<br>
## TensorFlow的数据类型，变量和占位符
&emsp;&emsp;对计算图的理解将帮助我们从子图和操作的角度考虑复杂模型。让我们来看一个只有一个隐藏层的神经网络的例子，以及它的计算图在TensorFlow中的样子：<br>
&emsp;&emsp;因此，我们有一些试图计算的隐藏层，一些参数矩阵W乘以一些输入值x加上偏倚项b作为激活函数。 Relu函数取输入值和零中最大的数值。下面的图表展示了TensorFlow计算图：<br>
&emsp;&emsp;在这个图中，我们有变量b和W以及一个叫做占位符的x；我们也让计算图图中的每个操作都有节点。因此，让我们更加详细地了解那些节点类型。<br>
## 变量
&emsp;&emsp;变量将是状态节点，这些节点输出它们的当前值。在这个例子中，它只是b和W。我们所说的变量是有状态的，是指它们在多个执行中保持当前值，并且很容易将保存的值还原成变量：<br>
&emsp;&emsp;此外，变量还有其他有用的特性；例如，它们可以在训练期间和之后保存到磁盘上，这就提供了我们之前提到的设备使用，它允许来自不同公司和小组的人保存、存储和发送他们的模型参数给其他人。此外，变量是为了使损失最小你想要调整的，我们将看到如何尽快做到这一点。<br>
&emsp;&emsp;计算图中变量的操作是很重要的，比如b和W。因为根据定义，图中的所有节点都是操作。因此，当您评估这些持有值的操作时，b和W在运行时，您将得到这些变量的值。<br>
&emsp;&emsp;我们可以使TensorFlow的Variable()函数去定义变量并赋予它一些初始值：<br>
&emsp;&emsp;`var = tf.Variable(tf.random_normal((0,1)),name='random_values')`<br>
&emsp;&emsp;这一行代码将定义2行2列的变量，并用标准正态分布初始化它。你也可以给变量赋一个名称。<br>
## 占位符
&emsp;&emsp;第二种类型的节点是占位符。占位符的值在执行时被给予的节点：<br>
&emsp;&emsp;如果您在计算图中依赖于一些外部数据的输入，那么这些值的占位符，我们将在训练期间添加到计算中。因此，对于占位符，我们不提供任何初始值。我们只是赋值一个张量的数据类型和形状，所以图形仍然知道要计算什么，即使它还没有任何存储值。我们可以使用TensorFlow的占位符函数来创建占位符：<br>
```
ph_var1 = tf.placeholder(tf.float32,shape=(2,3))  
ph_var2 = tf.placeholder(tf.float32,shape=(3,2))
result = tf.matmul(ph_var1,ph_var2)
```
&emsp;&emsp;这些代码行定义特定形状的两个占位符变量，然后定义将这两个值相乘的操作（参见下一节）。<br>
## 数学运算
&emsp;&emsp;第三种类型的节点是数学运算，它们是矩阵乘法（MatMul）、加法（Add）和ReLU激活函数。所有这些都是TensorFlow图中的节点，并且非常类似于NumPy操作：<br>
&emsp;&emsp;让我们看看计算图用代码如何实现。我们执行以下步骤来生成计算图：<br>
&emsp;&emsp;1.创建权重W和b，并将其初始化。我们可以利用均匀分布W~U(-1,1)初始化权重矩阵W ，并将b的值初始化为0。<br>
&emsp;&emsp;2.创建并输入占位符x，这将是一个m行784列的矩阵。<br>
&emsp;&emsp;3.建立流程图。<br>
&emsp;&emsp;让我们按照下面的步骤来构建流程图：<br>
```
b = tf.Variable(tf.zeros((100,)))  #100行的0矩阵
W = tf.Variable(tf.random_uniform((784,100),-1,1))  #生成1个784*100的矩阵，生成的值在（-1,1）内服从均匀分布
x = tf.placeholder(tf.float32,(100,784))  
h = tf.nn.relu(tf.matmul(x,W) + b)  #数据运算 计算激活函数relu
```
&emsp;&emsp;从前面的代码中可以看出，我们实际上并没有用这个代码片段操做任何数据。我们只是在图形内部建立符号，在运行这个图表之前不能打印出来h和看到他的值。所以，这个代码片段只是用于构建一个我们模型的主干。如果尝试打印值W或b在前面的代码中，您应该在Python中获得以下输出：<br>
&emsp;&emsp;到目前为止，我们已经定义了我们的计算图。现在，我们需要实际运行它。<br>
## 从TensorFlow中获取输出
&emsp;&emsp;在前一节中，我们知道如何构建一个计算图，但是我们需要实际运行它并获得它的值。我们可以使用“会话”来部署/运行计算图，该会话只是对特定执行上下文（如CPU或GPU）的绑定。因此，我们将采取的计算图是建立和部署到CPU或GPU上下文。为了运行该图，我们需要定义一个叫做sess的会话对象，我们将调用并运行这个函数，它包含两个参数：<br>
`sess.run(fetches,feeds)`
&emsp;&emsp;1.fetches是返回节点输出的图形节点列表。这些是我们感兴趣的节点在计算的值。<br>
&emsp;&emsp;2.feeds将是我们想在模型中运行的从图节点到实际值的字典映射。这就是我们之前填写占位符的地方。<br>
&emsp;&emsp;那么，让我们继续运行我们的计算图：<br>
```
#从tensorflow中获取输出  建立计算图之后，可建立session对话函数执行计算图。
import numpy as np  #导入numpy
sess = tf.Session()  
sess.run(tf.global_variables_initializer())  #初始化tensorflow gloal变量
sess.run(h,{x:np.random.random((100,784))})  #生成100*784的0-1之间的随机浮点数。
```
&emsp;&emsp;运行我们的计算图之后通过sess对象，我们应该得到和下面类似的输出：<br>
&emsp;&emsp;如你所见，在上面的代码片段的第二行中，我们初始化了变量，这是TensorFlow中的一个概念称为延后计算。这意味着你的计算图的计算只在运
行时发生，而在TensorFlow中运行时意味着会话。所以，调用这个函数global_variables_initializer()实际上会初始化图表中的变量，如在我们的例子中的W和b。<br>
&emsp;&emsp;我们还可以使用会话变量，以确保在执行计算图之后它将被关闭：<br>
#使用会话变量确保在执行完流程图后将其关闭.<br>
```
ph_var1 = tf.placeholder(tf.float32,shape=(2,3)) 
ph_var2 = tf.placeholder(tf.float32,shape=(3,2)) 
result = tf.matmul(ph_var1,ph_var2) 
with tf.Session() as sess: 
    print(sess.run([result],feed_dict={ph_var1:[[1.,3.,4.],[1.,3.,4.]],ph_var2:[[1., 3.],[3.,1.],[.1,4.]]})) 
```
## TensorBoard —— 可视化学习
&emsp;&emsp;使用TensorFlow进行大规模深层神经网络之类的计算可能很复杂和令人困惑，并且其相应的计算图也会很复杂。为了便于理解、调试和优化TensorFlow程序，TensorFlow团队已经包括了一套名为TensorBoard的可视化工具，这是一套可以通过浏览器运行的Web应用程序。TensorBoard可用于可视化您的TensorFlow图，绘制关于计算图执行的定量度量，并显示其他数据，例如通过它的图像。当TensorBoard被完全安装时，看起来是这样的：<br>
&emsp;&emsp;为了理解TensorBoard的工作原理，我们将构建一个计算图，该图将充当MNIST数据集（手写图像的数据集）的分类器。您不必理解这个模型的所有细节，但是它将向您展示在TensorFlow中实现的机器学习模型的一般流程。<br>
&emsp;&emsp;因此，让我们首先导入TensorFlow并使用TensorFlow帮助函数加载所需的数据集；这些帮助函数将检查您是否已经下载了数据集，否则它将为您下载：<br>
```
#MNIST数据集共有训练数据6000项，测试数据1000项。MNIST数据都有images(数字图像)与labels(真实的数字)组成。
#下载MNIST数据集 
import tensorflow as tf  #导入tensorflow模块
from tensorflow.examples.tutorials.mnist import input_data  #tensorflow中已经提供现成模块可用于下载并读取数据
mnist_dataset = input_data.read_data_sets("/tmp/data/", one_hot=True)  #one_hot编码，1个one_hot向量只有1位数是1，其他维数全都是0.
```
output:<br>
Extracting /tmp/data/train-images-idx3-ubyte.gz<br>
Extracting /tmp/data/train-labels-idx1-ubyte.gz<br>
Extracting /tmp/data/t10k-images-idx3-ubyte.gz<br>
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz<br>
&emsp;&emsp;接下来，我们需要定义超参数（可用于微调模型性能的参数）和模型的输入：<br>
```
定义训练参数并输入模型
learning_rate = 0.01  #学习率
num_training_epochs = 25  #迭代25次
train_batch_size = 100  #训练样本的样本个数
display_epoch = 1      
logs_path = '/tmp/tensorflow_tensorboard/'  #数据下载的路径
input_values = tf.placeholder(tf.float32,[None,784],name='input_values')  #输入值，形状第一维项数不定，第二维784个项数。
target_values = tf.placeholder(tf.float32,[None,10],name='target_values')  #目标值
weights =tf.Variable(tf.zeros([784,10]),name='weights')  #权重
biases =tf.Variable(tf.zeros([10]),name='biases')  #偏差值  生成1行10列全0矩阵
```
&emsp;&emsp;现在我们需要建立模型并定义一个我们将要优化的损失函数：<br>
```
定义损失函数
with tf.name_scope('Model'):  # 它的主要目的是为了更加方便地管理参数命名。
    predicted_values = tf.nn.softmax(tf.matmul(input_values,weights)) + biases  #预测值
with tf.name_scope('Loss'):
    model_cost = tf.reduce_mean(-tf.reduce_sum(target_values*tf.log(predicted_values),reduction_indices=1))  #损失模型，使用交叉熵训练法
with tf.name_scope('SGD'):
    model_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model_cost)  #使用梯度下降法优化模型
with tf.name_scope('Accuracy'):
    model_accuracy = tf.equal(tf.argmax(predicted_values,1),tf.argmax(target_values,1))  #判断预测值和真实值是否相等，相等返回1，不等返回0.
    model_accuracy = tf.reduce_mean(tf.cast(model_accuracy,tf.float32))  #tf.cast转换数据类型
init = tf.global_variables_initializer() #初始化tensorflow gloal变量
```
&emsp;&emsp;我们将定义汇总变量，用于监视将发生在特定的变量上的更改，比如损失函数，以及它如何通过训练过程：<br>
```
tf.summary.scalar('model loss',model_cost)  #对标量数据汇总和记录使用
tf.summary.scalar('model accuracy',model_accuracy)
merged_summary_operation = tf.summary.merge_all()
```
&emsp;&emsp;最后，我们将通过定义会话变量来运行模型，会话变量将用于执行我们建立的计算图：<br>
```
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph()) #使用程序代码将要显示在tensorboard的计算图写入log文件
    for train_epoch in range(num_training_epochs):
        average_cost = 0
        total_num_batch = int(mnist_dataset.train.num_examples/train_batch_size) #计算每个训练周期，所需执行的批次=训练数据项数6000/每一批次项数1000
        for i in range(total_num_batch):
            batch_xs,batch_ys = mnist_dataset.train.next_batch(train_batch_size)  #读取批次数据
            _, c,summary = sess.run([model_optimizer,model_cost,merged_summary_operation],
            feed_dict = {input_values:batch_xs,target_values:batch_ys})   #see.run计算准确率，并通过feed_dict把数据传给两个占位符。
            summary_writer.add_summary(summary,train_epoch * total_num_batch + i)
            average_cost  += c / total_num_batch  #相加返回前一个变量
        if (train_epoch+1) % display_epoch == 0:  #求余运算，为了将25次的迭代结果全部显示
            print("Epoch:", '%03d' % (train_epoch+1), "cost=", "{:.9f}".format(average_cost)) #定义打印出来的格式。
    print("Optimization Finished!")
    print("Accuracy:",model_accuracy.eval({input_values:mnist_dataset.test.images,target_values:mnist_dataset.test.labels}))
    print("To view summaries in the Tensorboard,run the command line:\n"\
              "--> tensorboard.exe --logdir=/tmp/tensorflow_tensorboard"\
              "\nThen open http://localhost:6006/ into your web browser")
```
output:<br>
Epoch: 001 cost= 2.250371679<br>
Epoch: 002 cost= 2.091732949<br>
Epoch: 003 cost= 1.960212840<br>
Epoch: 004 cost= 1.882646561<br>
Epoch: 005 cost= 1.833850788<br>
Epoch: 006 cost= 1.801827587<br>
Epoch: 007 cost= 1.779734955<br>
Epoch: 008 cost= 1.763468241<br>
Epoch: 009 cost= 1.750940601<br>
Epoch: 010 cost= 1.740937850<br>
Epoch: 011 cost= 1.732734323<br>
Epoch: 012 cost= 1.725847443<br>
Epoch: 013 cost= 1.719974984<br>
Epoch: 014 cost= 1.714881604<br>
Epoch: 015 cost= 1.710406746<br>
Epoch: 016 cost= 1.706438520<br>
Epoch: 017 cost= 1.702880976<br>
Epoch: 018 cost= 1.699663615<br>
Epoch: 019 cost= 1.696721135<br>
Epoch: 020 cost= 1.694013688<br>
Epoch: 021 cost= 1.691496699<br>
Epoch: 022 cost= 1.689108889<br>
Epoch: 023 cost= 1.686783923<br>
Epoch: 024 cost= 1.684409949<br>
Epoch: 025 cost= 1.681657583<br>
Optimization Finished!<br>
Accuracy: 0.8322<br>
To view summaries in the Tensorboard,run the command line:<br>
--> tensorboard.exe --logdir=/tmp/tensorflow_tensorboard<br>
Then open http://localhost:6006/ into your web browser<br>
