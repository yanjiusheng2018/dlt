
#  tensorBoard--可视化学习，以MNIST手写数字识别数据集为例

### 1.下载MNIST数据集 


```python
import tensorflow as tf  #导入tensorflow模块                                                                                                                                                                                                               
from tensorflow.examples.tutorials.mnist import input_data  #tensorflow中已经提供现成模块可用于下载并读取数据
mnist_dataset = input_data.read_data_sets("/tmp/data/", one_hot=True)  #one_hot编码，1个one_hot向量只有1位数是1，其他维数全都是0.
```

    Extracting /tmp/data/train-images-idx3-ubyte.gz
    Extracting /tmp/data/train-labels-idx1-ubyte.gz
    Extracting /tmp/data/t10k-images-idx3-ubyte.gz
    Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
    

###  2.定义训练参数并输入模型


```python
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

### 3.定义损失函数


```python
#3.定义损失函数
with tf.name_scope('Model'):  # 它的主要目的是为了更加方便地管理参数命名。
    predicted_values = tf.nn.softmax(tf.matmul(input_values,weights)) + biases  #预测值
with tf.name_scope('Loss'):
    model_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_values,labels=target_values))  #损失模型,直接调用使用交叉熵训练法
with tf.name_scope('SGD'):
    model_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model_cost)  #使用梯度下降法优化模型
with tf.name_scope('Accuracy'):
    model_accuracy = tf.equal(tf.argmax(predicted_values,1),tf.argmax(target_values,1))  #判断预测值和真实值是否相等，相等返回1，不等返回0.
    model_accuracy = tf.reduce_mean(tf.cast(model_accuracy,tf.float32))  #tf.cast转换数据类型
init = tf.global_variables_initializer() #初始化tensorflow gloal变量
```


```python
tf.summary.scalar('model loss',model_cost)  #对标量数据汇总和记录使用
tf.summary.scalar('model accuracy',model_accuracy)
merged_summary_operation = tf.summary.merge_all()
```

    INFO:tensorflow:Summary name model loss is illegal; using model_loss instead.
    INFO:tensorflow:Summary name model accuracy is illegal; using model_accuracy instead.
    

### 4.定义会话并输出结果


```python
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

    Epoch: 001 cost= 2.250371679
    Epoch: 002 cost= 2.091732949
    Epoch: 003 cost= 1.960212840
    Epoch: 004 cost= 1.882646561
    Epoch: 005 cost= 1.833850788
    Epoch: 006 cost= 1.801827587
    Epoch: 007 cost= 1.779734955
    Epoch: 008 cost= 1.763468241
    Epoch: 009 cost= 1.750940601
    Epoch: 010 cost= 1.740937850
    Epoch: 011 cost= 1.732734323
    Epoch: 012 cost= 1.725847443
    Epoch: 013 cost= 1.719974984
    Epoch: 014 cost= 1.714881604
    Epoch: 015 cost= 1.710406746
    Epoch: 016 cost= 1.706438520
    Epoch: 017 cost= 1.702880976
    Epoch: 018 cost= 1.699663615
    Epoch: 019 cost= 1.696721135
    Epoch: 020 cost= 1.694013688
    Epoch: 021 cost= 1.691496699
    Epoch: 022 cost= 1.689108889
    Epoch: 023 cost= 1.686783923
    Epoch: 024 cost= 1.684409949
    Epoch: 025 cost= 1.681657583
    Optimization Finished!
    Accuracy: 0.8322
    To view summaries in the Tensorboard,run the command line:
    --> tensorboard.exe --logdir=/tmp/tensorflow_tensorboard
    Then open http://localhost:6006/ into your web browser
    
