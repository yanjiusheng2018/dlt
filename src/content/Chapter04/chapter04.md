## 计算图
&emsp;&emsp;关于TensorFlow的所有想法中最大的想法是，数值计算被表示为计算图，如下图所示。因此，任何TensorFlow程序的主干将是一个计算图，下面的说法是正确的：<br>
&emsp;&emsp;&emsp;&emsp;1.图中的节点是具有任意数量的输入和输出的操作。<br>
&emsp;&emsp;&emsp;&emsp;2.节点之间的图边将是在这些操作之间流动的张量，并且认为张量在实践中的最佳方式是n维数组。<br>
&emsp;&emsp;使用流程图作为深度学习框架骨干的优点在于，它允许您根据小而简单的操作构建复杂模型。此外，这将使梯度计算非常简单，我们会在后面的部分中提到这一点。<br>
&emsp;&emsp;另外一种考虑TensorFlow计算图的方法是每一个操作是一个可以在那个点进行评估的函数。<br>
## TensorFlow的数据类型，变量和占位符
&emsp;&emsp;对计算图的理解将帮助我们从子图和操作的角度考虑复杂模型。让我们来看一个只有一个隐藏层的神经网络的例子，以及它的计算图在TensorFlow中的样子：<br>
![](https://github.com/yanjiusheng2018/dlt/blob/guiabbey-patch-1/src/content/Chapter04/chapter04_images/tu2.jpg)
