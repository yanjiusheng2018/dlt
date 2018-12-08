# Chapter 10    循环神经网络RNN——语言建模
&emsp;&emsp;循环神经网络（RNNs）是一类广泛用于自然语言处理（NLP）的深度学习体系结构，这套体系结构使我们能够为当前预测提供语境信息，并且具有特定的体系结构能够处理任一输入序列中的长时依赖关系。在本章中，我们将介绍如何建立序列到序列（seq2seq）的模型，seq2seq在自然语言处理（NLP）的应用中将会非常有用。我们将通过构建字符级语言模型来介绍RNN概念，并了解我们的模型是如何生成类似于原始输入序列的句子。<br>
#### 本章将包含以下三个方面：
*   RNNs的本质
*   LSTM（Long Short Term Memory长短期记忆网络）网络
*   语言模型的实现
## RNNs的本质
&emsp;&emsp;到目前为止，我们学习并运用的深度学习体系结构都没有任何机制能够记忆之前输入的序列或字符的信息。例如，前馈神经网络（FNN），如果输入一串字符序列HELLO，当神经网络接收到E，你会发现它忘记了刚读取过H，更没有保存任何关于H的信息，。这对基于序列的学习而言是非常严重的问题。因为它对以前读取的字符没有记忆，因此运用这类神经网络去训练预测下一个字符是非常困难的。同时这对于语言模型、机器翻译、语音识别等的应用都没有意义。<br>
&emsp;&emsp;由于这个特殊的原因，我们引入了RNNs，这是一组深度学习体系结构，它能够保存信息并记忆刚接收到的信息。<br>
&emsp;&emsp;接下来演示RNNs应该如何处理相同的输入的字符序列，HELLO。当RNNs记忆单元接收到E作为输入字符，它同时也会接收到字符H，字符H比字符E接收的更早。这种把当前的字符以及之前的字符作为对RNN细胞元的输入时，对这种体系结构提供了很大的优势，即短期记忆；在这个特定的字符序列中，它还能使这些体系结构可用于预测或猜测H之后最可能出现的字符，即E。<br>
&emsp;&emsp;我们已经了解到，以前的体系结构为其输入分配权重；RNNs遵循相同的优化过程，为其多个输入分配权重，即当前的输入和过去的输入。所以在这个案例中，神经网络将会对当输入前和上一时刻的输出作为这一时刻的输入分配两个不同的权重矩阵。为了做到这一点，我们将使用梯度下降和重配比的反向传播，即基于时间的反向传播算法（BPTT）。<br>
## 循环神经网络体系结构
&emsp;&emsp;基于我们之前使用的深度学习体系结构的背景，你会发现RNNs的特别之处。我们之前学习的结构体系在输入或训练方面并不灵活。这些结构体系接收固定大小的序列、向量、图像作为输入并产生另一个固定大小的序列、向量、图像作为输出。RNN体系结构在某种程度上是不同的，因为它可以输入一个序列但输出另一个序列，或者如图一所示，输入序列但是单输出，或单输入但是输出为序列。这种灵活性对于如语言建模和情绪分析的多种应用程序非常有用:<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE1.jpg )<br>
&emsp;&emsp;图1：在输入或输出形状方面RNNs的灵活性"<br>
&emsp;&emsp;这类体系结构的本质是模仿人类处理信息的方式。在任何谈话过程中，你对对方话语的理解完全取决于他之前所讲的话，你甚至可以根据对方刚才讲的话预测他接下来会将什么。<br>
&emsp;&emsp;RNN在运用过程中也应该遵循完全相同的过程。例如，假设你想要翻译某一个句子中的一个特定的单词。你不会使用传统的前馈神经网络，因为传统的神经网络没有将之前接收到的单词的翻译的输出作为我们想要翻译的当前单词的输入的能力，并且也会因为缺少单词的上下文的信息而导致翻译错误。<br>
&emsp;&emsp;RNNs保留过去的信息，并具有某种循环方式，允许在任何给定的点上使用之前学习到的信息进行当前预测：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE2.jpg)<br>
&emsp;&emsp;图2：RNNs体系结构具有保留过去步骤的信息的循环<br>
&emsp;&emsp;在图2中，A是接收X(t)作为输入的一些神经网络，并产生和输出h(t)。此外，在这个循环的辅助下接收前一个步骤的信息。<br>
&emsp;&emsp;这个循环看上去不是那么清晰，但是如果我们把循环展开，如图2所示，你会发现循环非常简单和直观，RNN只不过是同一个网络（可能是普通FNN）的重复，如图3所示：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE3.jpg)图3：RNN体系结构展开图<br>
&emsp;&emsp;RNNs这种直观的结构及其在输入输出形状方面的灵活性使其非常适用于基于序列的学习任务，例如机器翻译、语言建模、情绪分析、图像字幕等。<br>
## RNNs的案例
&emsp;&emsp;现在，我们对RNNs的工作原理以及它在不同的基于序列的示例中的有用性有了直观的了解。让我们进一步了解其中一些有趣的例子。<br>
### 字符级语言模型
&emsp;&emsp;语言建模是语音识别、机器翻译等许多应用的一项必要且重要的任务。在本节中，我们尝试模拟RNN的训练过程，并深入了解这些网络的工作原理。我们将建立一个对字符进行操作的语言模型。因此，我们将一堆文本信息作为对神经网络的输入，目的是试图建立下一个字符的概率分布，因为前面输出的字符将允许我们生成与在培训过程中作为输入提供的文本相似的文本。<br>
&emsp;&emsp;例如，假设我们有一种语言，它只有四个字母作为其词汇，helo。<br>
&emsp;&emsp;我们要做的任务是输入特定的字符序列（如HELLO）来训练循环神经网络。在这个特殊示例中，我们有四个训练样本：<br>
&emsp;&emsp;1.根据输入的第一个字符h的上下文计算字符e的概率，<br>
&emsp;&emsp;2.根据给定的he的上下文计算字符l的概率，<br>
&emsp;&emsp;3.根据给定的hel的上下文计算字符l的概率，<br>
&emsp;&emsp;4.最后根据给定的hell的上下文计算字符0的概率。<br>
&emsp;&emsp;正如我们前几章学到的，机器学习技术通常是深度学习的一部分，只接受实值数字作为输入。所以，我们需要以某种方式转换或编码或输入字符为数字形式。为此，我们使用独热（one-hot）向量编码，这是一种通过具有零向量的方法对文本进行编码, 向量中除了词汇中字符的索引是1，其余位置均是0（本例词汇helo）。在对训练样本进行编码后，我们将一次性把编码后的训练样本输入到RNN类型的模型中。对每个给定的字符，RNN类型的模型的输出结果都是一个四维的向量（向量的维度对应于词汇数量），它表示词汇中每个字符作为给定输入字符之后的下一个字符的概率。图4表明了该过程：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE4.jpg)图4：以独热编码字符作为RNN类型的网络的输入以及输出是基于当前输入之后最有可能出现的字符的概率分布示例<br>
&emsp;&emsp;正如图4所示，你会发现我们将输入序列中的第一个字符h输入模型，输出的四维向量表示下一个字符的置信度。因此，在输入h之后，出现的下一个字符是h的置信度是1，出现下一个字符是e的置信度是2.2，出现下一个字符是l的置信度是-3.0，出现下一个字符是o的置信度4.1。在这个特殊示例中，基于我们得训练样本的序列是hello，我们知道正确的下一个字符是e。因此，我们在训练这个RNN型网络的同时，主要目标是增加e作为下一个字符的可信度，降低其他字符的可信度。为了达到优化目的，我们将使用梯度下降和反向传播算法进行权重的更新并影响网络，以便对下个出现的准确的字符e生成更高的可信度，以此类推，对其他三个训练样本也进行权重更新以降低损失。<br>
&emsp;&emsp;正如你所看到的，RNN型网络的输出会对词汇中的所有字符作为下一个字符出现生成置信分布。我们可以将这种置信分布转化为概率分布，这样某一个字符作为下一个字符出现的概率增加会导致其他字符出现的概率，因为概率和恒为1。对于这种特殊优化，我们可以对每个输出向量进行标准softmax函数的转换。<br>
&emsp;&emsp;为了从这类网络中生成文本，我们可以对这个模型输入一个初始字符并得到接下来有可能出现的字符的概率分布，然后可以从这些字符中进行采样，并将其作为输入字符返回输入到模型中。通过一遍又一遍的重复这个过程我们可以得到一系列字符，也就是我们想要生成的固定长度的文本。<br>
### 使用莎士比亚数据建立语言模型
&emsp;&emsp;从上述例子中，我们可以通过模型生成文本。但另我们惊奇的是，神经网络不仅会生成文本，还会学习训练数据的风格和结构。我们可以通过对某一具有结构和风格的特定文本进行RNN型模型的训练来阐明这个有趣的过程，比如下面莎士比亚的作品:<br>
&emsp;&emsp;让我们来看看通过训练网络生成输出的作品：<br>
&emsp;&emsp;Second Senator: 　<br>
&emsp;&emsp;They are away this miseries, produced upon my soul,<br>
&emsp;&emsp;Breaking and strongly should be buried, when I perish<br>
&emsp;&emsp;The earth and thoughts of many states.<br>
&emsp;&emsp;尽管神经网络一次只生成一个字符，但它能生成一些有意义的文本和名字，这些文本和名字实际上都具有莎士比亚作品的结构和风格。<br>
## 梯度消失问题
&emsp;&emsp;在训练这类RNN型结构体系时，我们使用梯度下降和基于时间的反向传播算法，这为许多基于序列的学习工作带来了一些成功。但是因为梯度的性质和使用快速训练策略的原因，可以证明梯度价值会变小甚至会消失。这个过程说明了许多学习者陷入的梯度消失的问题。<br>
&emsp;&emsp;在本章的后面部分，我们将讨论研究者是如何处理这类问题并产生一般RNN（vanilla RNNs）算法的变体来克服这些问题：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE5.jpg)
&emsp;&emsp;图5：梯度消失问题 <br>
## 长时依赖问题
&emsp;&emsp;研究者们面对的另一个具有挑战性的问题是人们可以在文本中找到的长时依赖问题。例如，假设某一句子，“I used to live in France and I learned how to speak...”，很显然该句子的后一个单词是French。<br>
&emsp;&emsp;在这种情况下，具有短期依赖性的一般RNN模型便可以处理，如图6显示：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE6.jpg)
&emsp;&emsp;图6：展示文本中的短期依赖<br>
&emsp;&emsp;进一步举例，如果一个人一开始就说“I used to live in France..”，然后他/她开始描述在法国的美好生活，最后以“I learned to speak French”结尾。因此，要想用模型预测他/她在句子结束时说的所学到的语言，模型就需要用到前面说的关于live 和French的信息。如果模型无法追踪文本中的长时依赖关系，那么该模型则无法处理此类情况：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE7.jpg)<br>
&emsp;&emsp;图7：文本中长时依赖问题的挑战<br>
&emsp;&emsp;为了处理文本中的梯度消失和长时依赖问题，研究者引入了一般RNN模型的变体，即长短期记忆网络（LSTM）。<br>
## LSTM网络
&emsp;&emsp;LSTM，RNN模型的变体，用于帮助学习文本中的长时依赖关系。早在1997年，Hochreiter & Schmidhuber就介绍了LSTM网络（http://www.bioinf.jku.at/publications/older/2604.pdf） ，很多研究者也对此进行了研究并在很多领域取得了有趣的成果。<br>
&emsp;&emsp;因为LSTM网络的内部结构，因此这类型的体系结构能够处理文本中的长时依赖关系的问题。<br>
&emsp;&emsp;LSTM网络与常规的RNN模型类似，因为随着时间的推移，它有一个重复的模块，但这个重复部件的内部结构不同于常规的RNN。它包括很多用于遗忘和更新信息的图层：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE8.jpg)<br>
&emsp;&emsp;图8：标准RNN的重复模块包含一个层<br>
&emsp;&emsp;正如上文提到的，常规RNN只有一个非常简单的结构，例如一个tanh层，但是LSTMs神经网络有四个不同的层并以某种特殊的方式进行相互作用。这种特殊的相互作用的方式使得LSTM在很多领域能很好地工作，我们将在建立语言模型的示例中看到这一点：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE9.jpg)
&emsp;&emsp;图9：LSTM的重复模块包含四个相互作用的层<br>
&emsp;&emsp;有关数学细节以及四个层之间实际相互作用方式的更多详细信息，可以查阅以下网址：http://colah.github.io/posts/2015-08-Understanding-LSTMs/<br>
##  LSTM工作原理
&emsp;&emsp;常规的LSTM体系结构的第一步是决定哪些信息是不必要的，它将通过扔掉不必要的信息来为更重要的信息留下更多的空间来发挥作用。为此，模型中有一个层叫做遗忘门(forget gate layer)，它会根据上一时刻的输出h(t-1)和当前时刻的输入x(t)来产生一个0-1的值，来决定哪些信息是要丢弃的。<br>
&emsp;&emsp;1．第一步是经过输入门(input gate layer)，它用来决定神经元的上一个状态的哪些值需要更新<br>
&emsp;&emsp;2．第二步是生成一组新的候选值，这些值将添加到神经元中，对细胞状态进行更新<br>
&emsp;&emsp;最后，需要决定LSTM的神经元需要输出的内容。这些输出的内容将基于我们的细胞状态，但将是一个筛选过的版本。<br>
## 语言模型的实现
&emsp;&emsp;在这一部分中，我们将建立对字符进行操作的语言模型。为了演示模型的实现，我们将使用安娜•卡列尼娜的一本小说，并了解神经网络是如何学习文本的结构和风格并实现生成类似的文本：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE10.jpg)
&emsp;&emsp;图10：字符级RNN的一般结构<br>
&emsp;&emsp;*这个神经网络是根据Andrej Karpathy的关于RNNs的帖子( http://karpathy.github.io/2015/05/21/rnn-effectiveness/ )和Troch (http://github.com/karpathy/char-rnn) 的实现为基础的。*<br>
&emsp;&emsp;*同样，在r2rt(http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html) 以及在github上Sherjil Ozairp(http://github.com/sherjilozair/char-rnn-tensorflow) 写的关于神经网络的信息。下面是字符级RNN的一般体系结构。*<br>
&emsp;&emsp;我们将建立一个字符级的RNN模型用来训练安娜•卡列尼娜的小说(链接：https://en.wikipedia.org/wiki/Anna_Karenina) 。它将根据书中的文本生成一个新的文本。链接上包含txt文本和实现的代码。<br>
&emsp;&emsp;让我们从导入此字符级模型实现所需的库开始：<br>
```import numpy as np
import tensorflow as tf

from collections import namedtuple
```


&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
