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
<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE1.jpg">
</div><br> 
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE1.jpg )<br>          
&emsp;&emsp;图1：在输入或输出形状方面RNNs的灵活性<br>
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
```
import numpy as np
import tensorflow as tf

from collections import namedtuple
```
&emsp;&emsp;首先，我们需要通过加载数据集并将其转换为整数来为模型的实现做准备。因此，我们将把字符转换为整数，然后将他们编码成整数。这使得数据显得直观并可作为模型的输入量：<br>
```
#reading the Anna Karenina novel text file
with open('Anna_Karenina.txt', 'r') as f:
    textlines=f.read()	

#Building the vocan and encoding the characters as integers
language_vocab = set(textlines)
vocab_to_integer = {char: j for j, char in enumerate(language_vocab)}
integer_to_vocab = dict(enumerate(language_vocab))
encoded_vocab = np.array([vocab_to_integer[char] for char in textlines], dtype=np.int32)
```
&emsp;&emsp;让我们先来看安娜•卡列尼娜文本的前200个字符:<br>
```
textlines[:200]
Output:
"Chapter 1\n\n\nHappy families are all alike; every unhappy family is unhappy in its own\nway.\n\nEverything was in confusion in the Oblonskys' house. The wife had\ndiscovered that the husband was carrying on"
```
&emsp;&emsp;我们需把字符转换成整数形式，这种形式对神经网络而言直观且方便。因此，让我们来看看编码后的字符形式：<br>
```
encoded_vocab[:200]
Output:
array([79,  6, 58, 52, 77, 67,  9, 35, 13, 56, 56, 56, 12, 58, 52, 52, 61,
       35, 42, 58, 41, 43, 40, 43, 67,  1, 35, 58,  9, 67, 35, 58, 40, 40,
       35, 58, 40, 43, 54, 67, 37, 35, 67, 74, 67,  9, 61, 35, 76, 70,  6,
       58, 52, 52, 61, 35, 42, 58, 41, 43, 40, 61, 35, 43,  1, 35, 76, 70,
        6, 58, 52, 52, 61, 35, 43, 70, 35, 43, 77,  1, 35, 44, 14, 70, 56,
       14, 58, 61,  0, 56, 56, 39, 74, 67,  9, 61, 77,  6, 43, 70, 63, 35,
       14, 58,  1, 35, 43, 70, 35,  5, 44, 70, 42, 76,  1, 43, 44, 70, 35,
       43, 70, 35, 77,  6, 67, 35, 38,  4, 40, 44, 70,  1, 54, 61,  1, 60,
       35,  6, 44, 76,  1, 67,  0, 35, 73,  6, 67, 35, 14, 43, 42, 67, 35,
        6, 58, 10, 56, 10, 43,  1,  5, 44, 74, 67,  9, 67, 10, 35, 77,  6,
       58, 77, 35, 77,  6, 67, 35,  6, 76,  1,  4, 58, 70, 10, 35, 14, 58,
        1, 35,  5, 58,  9,  9, 61, 43, 70, 63, 35, 44, 70])
```
&emsp;&emsp;因为神经网络使用的是单个字符，因此它类似于分类问题，在该问题中，我们试图从上一个文本中预测下一个字符。这就是我们的神经网络要将文本分成多少类的问题。<br>
&emsp;&emsp;因此，每一次我们将为模型提供一个字符，并且模型将通过接下来可能出现的字符数生成概率分布来预测下一个字符，相当于神经网络需要从许多类中进行挑选。<br>
```
len(language_vocab)
Output:
83
```
&emsp;&emsp;由于我们将使用随机梯度下降来训练我们的模型，我们需要将我们的整个数据转换成训练批次。<br>
## 生成小批次训练                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
&emsp;&emsp;在本节中，我们将把我们的数据分成小的批次用于训练。因此，批由所需序列步骤数的许多序列组成。图11是一个可视化示例：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE11.jpg)
&emsp;&emsp;图11：批次和序列的外观示例<br>
&emsp;&emsp;因此，我们现在需要定义一个函数，该函数将遍历编码的文本并生成各个批。在这个函数中，我们将使用python机制，yield(http://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained )。<br>
&emsp;&emsp;典型的批由N×M个字符组成，N是序列数，M是序列阶数。为了得到数据集中可能的批数（the number of batches），我们只需要将数据长度除以希望的批规模（batch size），在得到可能的批的数量后，我们可以得到每个批中应该有多少字符。<br>
&emsp;&emsp;之后，我们需要将数据集拆分成所需数量（N）的序列。我们可以使用arr.reshape(size)。我们知道我们需要N个序列(代码中可以使用num_seqs)，将其作为第一维度的大小。对于第二维度，可以使用-1作为规模占位符，系统会为数组填充适当的数据。之后，会得到一个N×（M×K）的数组，其中K是批次数。<br>
&emsp;&emsp;进行上述运算后，我们得到了这个数组，我们可以通过循环访问来获得训练批次，每个批都含有N×M个字符。对于每一个后续批次，窗口以num_step设置的参数为步长移动。最后，我们还需要对我们的输入和输出创建数组，以便用作模型输入。创建输出值的步骤非常简单，但是记住我们需要的目标是在一个字符上移动的输入。你会发现第一个输入的字符被用来作为最后一个目标字符，类似于以下内容：<br>
&emsp;&emsp;y[：，：-1]，y[：，-1] = x[：，1，：]，x[：，0]<br>
&emsp;&emsp;在这里x是输入的批，y是目标批。<br>
&emsp;&emsp;在这一部分我常使用num_steps来控制批的步骤长度，从0开始到arr.shape[1]，即每个序列的总阶长数。这样，从范围中得到的整数总是指向批处理的开头，每部分都是num_steps设置的宽度。<br>
```
def generate_character_batches(data, num_seq, num_steps):
    '''Create a function that returns batches of size
       num_seq x num_steps from data.
    '''
    # Get the number of characters per batch and number of batches
    num_char_per_batch = num_seq * num_steps
    num_batches = len(data) // num_char_per_batch
    # Keep only enough characters to make full batches
    data = data[:num_batches * num_char_per_batch]
    # Reshape the array into n_seqs rows
    data = data.reshape((num_seq, -1))
    for i in range(0, data.shape[1], num_steps):
        # The input variables
        input_x = data[:, i:i + num_steps]
        # The output variables which are shifted by one
        output_y = np.zeros_like(input_x)
        output_y[:, :-1], output_y[:, -1] = input_x[:, 1:], input_x[:, 0]
        yield input_x, output_y
```
&emsp;&emsp;因此，让我们通过生成一个含有15个序列和50序列阶长的批来演示这个函数：<br>
```
generated_batches = generate_character_batches(encoded_vocab, 15, 50)
input_x, output_y = next(generated_batches)
print('input\n', input_x[:10, :10])
print('\ntarget\n', output_y[:10, :10])
Output:

input
 [[79  6 58 52 77 67  9 35 13 56]
 [35 67 40 43 63 43  4 40 67 35]
 [44 35 40 43 63  6 77 64  6 67]
 [35  6 58 10 35  4 67 67 70 35]
 [35 44 70 67 35 54 70 67 14 56]
 [61 67 58  9  1 35 40 58  1 77]
 [35  2 67 35 63 44 35 43 70 77]
 [44 14 35 77 44 35 41 58 54 67]
 [ 4 67 42 44  9 67 35  7 67 70]
 [ 1 58 63 67 35 40 43 54 67 35]]

target
 [[ 6 58 52 77 67  9 35 13 56 56]
 [67 40 43 63 43  4 40 67 35 63]
 [35 40 43 63  6 77 64  6 67 58]
 [ 6 58 10 35  4 67 67 70 35  4]
 [44 70 67 35 54 70 67 14 56 14]
 [67 58  9  1 35 40 58  1 77 35]
 [ 2 67 35 63 44 35 43 70 77 44]
 [14 35 77 44 35 41 58 54 67 47]
 [67 42 44  9 67 35  7 67 70 77]
 [58 63 67 35 40 43 54 67 35 77]]
```
&emsp;&emsp;接下来，我们将建立这个例子的核心，即LSTM记忆单元的建模。<br>
## 建立模型
&emsp;&emsp;在使用LSTMs深入构建字符级模型之前，有必要提及栈式长短期记忆神经网络(Stacked LSTM)。<br>
&emsp;&emsp;栈式LSTM可用于查看不同时间尺度下的信息。<br>
### 栈式LSTMs（Stacked LSTMs）
&emsp;&emsp;*“通过将多个循环的隐藏层堆叠起来构建深度RNN。这种方法在潜意识里允许不同隐藏层下的隐藏状态在不同的时间尺度上操作。”《如何构建深度RNN》一书中说到(http://arxiv.org/abs/1312.6026 )，2013。*<br>
&emsp;&emsp;*“因为他们的隐藏状态是之前所有隐藏状态的函数，RNN在本质上是在时间深处进行模型运作。从这篇论文得到启发的一个问题是是否RNN也可以从空间的深度中受益；这就是堆叠多个循环隐藏层，就像在传统的深层网络中堆叠前馈层一样”《深度RNNs中的语音识别》(https://arxiv.org/abs/1303.5778 )，2013。*<br>
&emsp;&emsp;很多研究者尝试用栈式LSTMs解决序列预测问题。栈式LSTM的结构被定义为有多个LSTM层构成的LSTM模型。之前介绍的LSTM层提供序列的输出，而不是单值输出到下一个LSTM层，如下所示。<br>
&emsp;&emsp;具体而言，它是每一个时间节点上的单一输出，而不是所有输入时间步长上的输出。<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE12.jpg)
&emsp;&emsp;图12：栈式LSTMs <br>
&emsp;&emsp;因此在本例中，我们将使用这种多层LSTM结构，以获得更好的效果。<br>
### 模型结构
&emsp;&emsp;我们将在此建立模型，我们会把它分成几个部分，这样就更容易对每个点进行推理。然后，我们可以将它们连接到整个神经网络：<br>
![image](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter10/chapter10_image/%E5%9B%BE13.jpg)
&emsp;&emsp;图13：字符级模型体系结构 <br>
## 输入
&emsp;&emsp;我们从对模型的初始输入定义一个占位符开始建模。模型的输入将是训练数据和目标。在dropout层使用keep_probability的参数，避免模型数据的过度拟合：<br>
```
def build_model_inputs(batch_size, num_steps):
    # Declare placeholders for the input and output variables
    inputs_x = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets_y = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    # define the keep_probability for the dropout layer
    keep_probability = tf.placeholder(tf.float32, name='keep_prob')
    return inputs_x, targets_y, keep_probability
```
## 建立LSTM细胞单元
&emsp;&emsp;在本部分中，我们将编写一个用于创建LSTM细胞单元的函数并作用于模型的隐藏层。这个细胞单元是模型的组成部分。以此，我们将用Tensorflow构建记忆单元。让我们了解我们是如何用Tensorflow构建基本的LSTM细胞单元。<br>
&emsp;&emsp;我们调用下面的代码创建LSTM细胞单元，参数num_units代表隐藏层的细胞单元数：<br>  
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units)<br>  
&emsp;&emsp;为了防止过度拟合，我们可以使用dropout层，它可以通过降低模型的复杂性防止数据过度拟合：<br>  
tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_probability)<br>  
&emsp;&emsp;正如我们前面所提及的，我们将使用栈式LSTM结构；它将帮助我们从不同角度理解数据，实践证明，它也比单层LSTM表现的更好。为了在Tensorflow中定义多层LSTM，可以使用tf.contrib.rnn.MultiRNNCell函数(http://www.tensorflow.org/versions/r1.0/api_docs/python/tf/contrib/rnn/MultiRNNCell):<br>  
tf.contrib.rnn.MultiRNNCell([cell] * num_layers)<br>
&emsp;&emsp;对于第一个细胞单元的初始状态，没有先前的信息，所以需要将第一个细胞单元初始为零。可以用以下函数来执行此操作：<br>  
initial_state = cell.zero_state(batch_size, tf.float32)<br>  
&emsp;&emsp;最后，把所有的模块都放在一起构建LSTM记忆单元：<br>
```
def build_lstm_cell(size, num_layers, batch_size, keep_probability):
    ### Building the LSTM Cell using the tensorflow function
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(size)
    # Adding dropout to the layer to prevent overfitting
    drop_layer = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_probability)
    # Add muliple cells together and stack them up to oprovide a level of more understanding
    stakced_cell = tf.contrib.rnn.MultiRNNCell([drop_layer] * num_layers)
    initial_cell_state = lstm_cell.zero_state(batch_size, tf.float32)
    return lstm_cell, initial_cell_state
```
## RNN输出
&emsp;&emsp;接下来，我们需要创建输出层，该层负责读取单个LSTM细胞单元的输出并通过与其完全连接的层输出。这一层使用softmax函数用于生成在输出一个字符后可能出现的字符的概率分布，进而输出结果。<br>
&emsp;&emsp;我们已经了解到对神经网络输入的批（batch）含有N×M 个字符，N是批包含的序列数，M是序列阶数。在创建模型时，我们还在隐藏层使用L个隐藏单元。由于批的规模和隐藏单元的数量，网络的输出是一个N×M×L 的三维张量，对于每一个序列，我们调用M次LSTM的细胞单元。每一次调用LSTM细胞单元都产生规模为L的输出。最后，由于批含有N个序列，因此需要重复操作N次。<br>
&emsp;&emsp;我们将NML的输出传递到与LSTM细胞单元完全连接的层（对于所有的输出具有相同的权重）。但是在输出之前，我们将该输出结果重组成2维的张量，维度为（NM）L。因为重组的2维对我们而言更简单，也会使模型在执行输出时更容易。每一行的值代表LSTM细胞单元的L个输出，因此行代表一个序列和一个步骤。<br>
&emsp;&emsp;对数组重组后，可以通过将矩阵与权重相乘并运用softmax函数将结果通过神经网络传到输出层。默认情况下，在LSTM细胞单元中创建的权重和我们要在此创建的权重具有相同的名称，在这种情况下，Tensorflow会产生错误。为了避免此类错误，可以使用Tensorflow中的tf.variable_scope()函数使创建的权重和偏置项在一个可变范围内变动。<br>
&emsp;&emsp;再解释了输出形状和如何重组数组之后，模型运作就会变得更加容易，让我们进一步用build_model_output函数编写输出层:<br>
```
def build_model_output(output, input_size, output_size):
    # Reshaping output of the model to become a bunch of rows, where each row correspond for each step in the seq
    sequence_output = tf.concat(output, axis=1)
    reshaped_output = tf.reshape(sequence_output, [-1, input_size])
    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((input_size, output_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(output_size))
    # the output is a set of rows of LSTM cell outputs, so the logits will be a set
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(reshaped_output, softmax_w) + softmax_b
    # Use softmax to get the probabilities for predicted characters
    model_out = tf.nn.softmax(logits, name='predictions')
    return model_out, logits
```
## 训练损失
&emsp;&emsp;接下来是训练损失。通过模型的运算，我们已经得到了逻辑运算和目标函数，并计算softmax的交叉熵损失。首先，对需要的目标进行独热编码，得到已经编码好的字符。其次，重组热编码之后的目标，得到规格为（MN）C的2维张量，其中C是类或字符数。由于我们重组了LSTM输出结果并通过具有C个单元的完全连接层运行。因此，我们的逻辑运算也是（MN）C。<br>
&emsp;&emsp;之后，通过tf.nn.softmax_cross_entropy_with_logits运行logits和targets并得到损失的平均值：<br>
```
def model_loss(logits, targets, lstm_size, num_classes):
    # convert the targets to one-hot encoded and reshape them to match the logits, one row per batch_size per step
    output_y_one_hot = tf.one_hot(targets, num_classes)
    output_y_reshaped = tf.reshape(output_y_one_hot, logits.get_shape())
    # Use the cross entropy loss
    model_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_y_reshaped)
    model_loss = tf.reduce_mean(model_loss)
    return model_loss
```
## 优化
&emsp;&emsp;最后，我们需要用某种优化方法以帮助我们从数据集中学习新的东西。众所周知，一般RNN存在梯度爆炸和梯度消失的问题。LSTM仅仅解决了梯度消失的问题，但即使在使用LSTM之后，有些梯度值会在没有边界的情况下爆炸和增长。为了解决这个问题，我们可以引入梯度裁剪（gradient clipping）,这是一种将梯度裁剪到特定阈值的技术。<br>
&emsp;&emsp;因此，我们使用Adam优化器定义我们学习过程要用的优化器：<br>
```
def build_model_optimizer(model_loss, learning_rate, grad_clip):
    # define optimizer for training, using gradient clipping to avoid the exploding of the gradients
    trainable_variables = tf.trainable_variables()
    gradients, _ = tf.clip_by_global_norm(tf.gradients(model_loss, trainable_variables), grad_clip)
    # Use Adam Optimizer
    train_operation = tf.train.AdamOptimizer(learning_rate)
    model_optimizer = train_operation.apply_gradients(zip(gradients, trainable_variables))
    return model_optimizer
```
## 构建网络
&emsp;&emsp;现在我们将所有的部分放在一起构建神经网络。通过LSTM细胞单元单元实际运行数据时我们使用tf.nn.drnamic_rnn (http://www.tensorflow.org/version/r1.0/api_docs/python/tf/nn/dynamic_rnn )。这个函数会为我们适当的传递LSTM细胞中的隐藏状态和细胞状态。它对批（mini-batch）中的每个序列进行操作，并在每一个步骤中返回每个LSTM细胞单元的输出并提供最终的LSTM状态。我们把这个状态存为final_state，因此在下一个批运行时，可以将最终状态传给第一个LSTM细胞单元。我们将从build_lstm得到的初始状态以及输入的序列传递到细胞单元，并通过tf.nn.dynamic_rnn运行。同样，将数据输入到RNN模型中时也需要进行独热编码：<br>
```
class CharLSTM:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):
        # When we're using this network for generating text by sampling, we'll be providing the network with
        # one character at a time, so providing an option for it.
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()
        # Build the model inputs placeholders of the input and target variables
        self.inputs, self.targets, self.keep_prob = build_model_inputs(batch_size, num_steps)
        # Building the LSTM cell
        lstm_cell, self.initial_state = build_lstm_cell(lstm_size, num_layers, batch_size, self.keep_prob)
        ### Run the data through the LSTM layers
        # one_hot encode the input
        input_x_one_hot = tf.one_hot(self.inputs, num_classes)
        # Runing each sequence step through the LSTM architecture and finally collectting the outputs
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, input_x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        # Get softmax predictions and logits
        self.prediction, self.logits = build_model_output(outputs, lstm_size, num_classes)
        # Loss and optimizer (with gradient clipping)
        self.loss = model_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_model_optimizer(self.loss, learning_rate, grad_clip)
```
## 模型超参数
&emsp;&emsp;正如其他深度学习的体系结构，RNN也有一些超参数可以用来控制模型并对其进行微调。下面是此结构体系中使用的一组超参数：<br>
*   批的大小是在一次传递中通过网络运行的序列数。
*   步骤数是网络训练的序列所含的字符数。步骤数多通常更好，网络会学习更多的长远依赖关系，但需要更长的时间进行网络训练，100通常较合理。
*   LSTM的规模是隐藏层的单元数。
*   步骤数是网络训练的序列所含的字符数。步骤数多通常更好，网络会学习更多的长远依赖关系，但需要更长的时间进行网络训练，100通常较合理。
*   LSTM的规模是隐藏层的单元数。
*   结构体系编号图层是要使用的隐藏LSTM层的数量。
*   学习率是训练的标准学习率。
*   最后，dropout层可以保持概率，以避免网络的数据过度拟合。因此如果神经网络运行过度拟合，尝试降低隐藏层的神经元。
## 训练模型
&emsp;&emsp;现在，通过向建立的模型提供输入和输出开始训练网络，然后优化训练网络。在开始预测当前状态时切记使用上一个状态的输出。因此，我们需要将输出状态重新返回到神经网络中，以便在预测下一个状态时是使用。<br>
&emsp;&emsp;我们需要对超参数设置初始值（之后可以根据用于训练这个体系结构的数据集对其进行调整）：<br>
```
batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_probability = 0.5         # Dropout keep probability
print('Starting the training process...')
epochs = 5

# Save a checkpoint N iterations
save_every_n = 100
LSTM_model = CharLSTM(len(language_vocab), batch_size=batch_size, num_steps=num_steps,
                      lstm_size=lstm_size, num_layers=num_layers,
                      learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Use the line below to load a checkpoint and resume training
    # saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(LSTM_model.initial_state)
        loss = 0
        for x, y in generate_character_batches(encoded_vocab, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {LSTM_model.inputs: x,
                    LSTM_model.targets: y,
                    LSTM_model.keep_prob: keep_probability,
                    LSTM_model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([LSTM_model.loss,
                                                 LSTM_model.final_state,
                                                 LSTM_model.optimizer],
                                                feed_dict=feed)
            end = time.time()
            print('Epoch number: {}/{}... '.format(e + 1, epochs),
                  'Step: {}... '.format(counter),
                  'loss: {:.4f}... '.format(batch_loss),
                  '{:.3f} sec/batch'.format((end - start)))
            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
```
&emsp;&emsp;训练过程的最后，会得到接近以下内容的错误：<br>
```
.
.
.
Epoch number: 5/5...  Step: 978...  loss: 1.7151...  0.050 sec/batch
Epoch number: 5/5...  Step: 979...  loss: 1.7428...  0.051 sec/batch
Epoch number: 5/5...  Step: 980...  loss: 1.7151...  0.050 sec/batch
Epoch number: 5/5...  Step: 981...  loss: 1.7236...  0.050 sec/batch
Epoch number: 5/5...  Step: 982...  loss: 1.7314...  0.051 sec/batch
Epoch number: 5/5...  Step: 983...  loss: 1.7369...  0.051 sec/batch
Epoch number: 5/5...  Step: 984...  loss: 1.7075...  0.065 sec/batch
Epoch number: 5/5...  Step: 985...  loss: 1.7304...  0.051 sec/batch
Epoch number: 5/5...  Step: 986...  loss: 1.7128...  0.049 sec/batch
Epoch number: 5/5...  Step: 987...  loss: 1.7107...  0.051 sec/batch
Epoch number: 5/5...  Step: 988...  loss: 1.7351...  0.051 sec/batch
Epoch number: 5/5...  Step: 989...  loss: 1.7260...  0.049 sec/batch
Epoch number: 5/5...  Step: 990...  loss: 1.7144...  0.051 sec/batch
```
## 检查点
&emsp;&emsp;现在，加载checkpoints。为了更好的了解存储和加载checkpoints，可以查看Tensorflow文档(http://www.tensorflow.org/programmers_guide/variables):<br>  
tf.train.get_checkpoint_state('checkpoints')<br>
## 生成文本
&emsp;&emsp;我们的训练模型是基于输入的数据集的。下一步是使用这个训练模型生成文本并了解这个模型是如何学习输入文本的风格和结构。为此，我们可以从一些初始字符开始，然后将新的预测字符作为下一步的输入。我们会重复这个步骤直到得到具有特定长度的文本。<br>
&emsp;&emsp;通过以下代码，我们还可以对函数添加额外的语句，以便使用一些初始文本为网络提供支持，并从初始文本开始运行。<br>
&emsp;&emsp;神经网络为我们提供词汇中每个字符的预测或概率。为了减少干扰并且只使用置信度更有可能的字符，我们将只从最有可能输出的前N个字符中选择一个新字符：<br>
```
def pick_top_n_characters(preds, vocab_size, top_n_chars=4):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n_chars]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample_from_LSTM_output(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    LSTM_model = CharLSTM(len(language_vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(LSTM_model.initial_state)
        for char in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_integer[char]
            feed = {LSTM_model.inputs: x,
                    LSTM_model.keep_prob: 1.,
                    LSTM_model.initial_state: new_state}
            preds, new_state = sess.run([LSTM_model.prediction, LSTM_model.final_state],
                                        feed_dict=feed)
        c = pick_top_n_characters(preds, len(language_vocab))
        samples.append(integer_to_vocab[c])
        for i in range(n_samples):
            x[0, 0] = c
            feed = {LSTM_model.inputs: x,
                    LSTM_model.keep_prob: 1.,
                    LSTM_model.initial_state: new_state}
            preds, new_state = sess.run([LSTM_model.prediction, LSTM_model.final_state],
                                        feed_dict=feed)
            c = pick_top_n_characters(preds, len(language_vocab))
            samples.append(integer_to_vocab[c])
    return ''.join(samples)
```
&emsp;&emsp;我们使用最新的存储的checkpoint开始取样过程：<br>  
tf.train.latest_checkpoint('checkpoints')<br> 
&emsp;&emsp;现在，是时候使用最新的checkpoints进行采样：<br>  
checkpoint = tf.train.latest_checkpoint('checkpoints') print('Sampling text frm the trained model....') sampled_text = sample_from_LSTM_output(checkpoint, 2000, lstm_size, len(language_vocab), prime="Far") print(sampled_text)<br> 
&emsp;&emsp;你可以看到，我们会产生一些有意义的词和一些毫无意义的词。为了获得更多的结果，你可以多次运行模型，并尝试使用超参数。<br>
## 总结
&emsp;&emsp;我们已经学习了RNNs，它的工作原理，以及RNNs的广泛运用。我们用一部有趣的小说作为数据集，训练了基于字符水平的RNN模型，以及RNN的应用走向。我们足够期待RNN领域的巨大创新，我相信它将成为智能系统的一个普遍却关键的组成部分。<br>
