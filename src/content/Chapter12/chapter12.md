# Chapter12  神经情感分析
&emsp;&emsp;在本章中，我们将讨论自然语言处理中的热点和流行应用之一，即情感分析。如今，大多数人都通过社交媒体平台来表达自己对某件事的看法，而利用这一庞大的文本来追踪顾客对某件事的满意程度，对于公司甚至政府来说都是至关重要的。<br>
&emsp;&emsp;在本章中，我们将使用循环神经网络来建立一个情感分析的解决方案。本章将讨论以下主题：<br>
&emsp;&emsp;●一般情感分析体系结构<br>
&emsp;&emsp;●情感分析——模型实现<br>
## 一、一般情感分析体系结构
&emsp;&emsp;在本节中，我们将重点介绍可用于情感分析的一般深度学习体系结构。下图显示了构建情感分析模型所需的处理步骤。<br>
&emsp;&emsp;因此，首先，我们将讨论自然人类语言：<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter12/chapter12_image/picture1.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图1：情感分析解决方案的一般传递途径，甚至是基于序列的自然语言解决方案<br>
&emsp;&emsp;我们将使用电影评论来构建这个情感分析应用程序。此应用程序的目标是根据输入的原始文本生成正面和负面的评论。例如，如果原始文本是这样的，“this movie is good”，那么我们需要模型来产生一个积极的情感。如果原始文本是“this is not a good movie”，那么我们就需要模型来产生一个消极的情感。情感分析应用程序将带领我们完成许多处理步骤，这些步骤是在神经网络(如嵌入)中使用自然人类语言所需要的。在这类应用中有几个困难：<br>
&emsp;&emsp;其中之一是序列可能有不同的长度。<br>
&emsp;&emsp;另一个问题是，如果我们只看单个单词(例如，good)，就会显示出一种积极的情感。然而，它的前面是“not”一词，那么它就是一种负面情感。这会使得我们的分析变得更加复杂，稍后我们会看到一个例子。<br>
&emsp;&emsp;正如我们在上一章中所了解的，神经网络不能工作在原始文本上，所以我们首先需要将它转换为所谓的数字。这些基本上都是整数值，我们需要遍历整个数据集，计算每个单词被使用的次数；然后，我们制作一个词汇表，每个单词都在这个词汇表中得到一个索引。如图1，单词this就转化为整数11，单词is转化为6，依此类推。
&emsp;&emsp;现在，我们已经将原始文本转换为一个名为token的整数列表。但神经网络仍然不能对这些数据进行操作，因为如果我们有10000个单词的词汇表，这些数字可以取0到9999之间的值，它们可能根本不相关。因此，与数字999相比，数字998可能有一个完全不同的语义。因此，我们将使用我们在上一章中学习到的表示学习或嵌入的思想。这个嵌入层将数字列表转换为实值向量，因此数字11转换为向量[0.67，0.36，…，0.39]，如图1所示。同样的情况也适用于下一个数字6。<br>
&emsp;&emsp;快速回顾我们在前一章中研究的内容：图1中所示的嵌入层会学习数字列表与它们对应的实值向量之间的映射。此外，嵌入层还会学习单词的语义，以便在这个嵌入空间中具有相似意义的单词之间有某种程度的接近。<br>
&emsp;&emsp;从输入的原始文本中，我们得到一个二维矩阵，即张量，它现在可以输入到循环神经网络(RNN)。它可以处理任意长度的序列，然后使该网络的输出具有Sigmoid激活函数的完全连接或dense层。因此，输出介于0到1之间，其中0值被认为是消极情感，1值被认为是积极情感。但是，如果Sigmoid函数的值既不是0也不是1呢？我们就需要在中间引入一个截断值或一个阈值，这样如果这个值低于0.5，那么相应的输入就被认为是一种消极情感，而高于这个阈值的值就会被认为是一种积极情感。<br>
### 1、RNNs——情感分析背景
&emsp;&emsp;让我们回顾一下RNNs的基本概念，并在情感分析应用中讨论它们。正如我们在RNN章中提到的，RNN的基本构造块是一个循环单元，如下图所示：<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter12/chapter12_image/picture2.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图2：RNN单位的抽象概念<br>
&emsp;&emsp;这个图是循环单元内部所发生的事情的抽象。我们在这里得到的是输入，因此这将是一个词，例如，good；当然，它必须转换为嵌入向量。然而，我们现在将忽视这一点。此外，这个单元有一种记忆状态，根据该state（状态）的内容和input（输入），我们将更新此状态并将新数据写入状态。例如，假设我们以前在输入中看到了单词not；我们将它写入状态，这样当我们在后面的输入中看到good一词时，由于我们从状态中知道我们刚刚看到了not这个词，我们就必须把它们写在一起，看到not good一词的状态，这样才可能表明整个输入文本可能有一种消极情感。<br>
&emsp;&emsp;从旧的状态和输入到状态的新内容的映射是通过一个所谓的gate（闸门）来完成的，不同版本的循环单元实现这些映射的方式不同。它基本上是一个带有激活函数的矩阵运算，但是我们稍后会看到，反向传播梯度有一个问题。因此，RNN必须以一种特殊的方式来设计，这样梯度就不会被太大的扭曲。<br>
&emsp;&emsp;在一个循环单元中，我们有一个类似的gate（闸门）来产生输出，而循环单元的输出又一次依赖于状态的当前内容和我们所看到的输入。因此，我们可以尝试将一个循环单元内部发生的处理进行展开，如下图所示：<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter12/chapter12_image/picture3.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图3：循环神经网络的展开图<br>
&emsp;&emsp;现在，我们这里只有一个循环单元，但是流程图显示了它在不同的时间步骤中发生了什么。因此：<br>
&emsp;&emsp;在时间步骤一中，我们将单词this输入到循环单元，它的内部记忆状态首先初始化为0。每当我们开始处理新的数据序列时，TensorFlow就会这样做。所以，我们看到单词this，循环单位状态是0。因此，我们使用内部gate（闸门）更新记忆状态，在时间步骤二中输入单词is，然后this在这个时间步骤二中使用，记忆状态就有了一些内容。然而This个词没有太多的意义，所以状态可能仍然是在0左右。而且is也没有太多的意义，所以也许这个状态仍然是0。<br>
&emsp;&emsp;在下一个时间步骤中，我们看到单词not，这是我们最终想要预测的整个输入文本的情感。它是我们需要存储在记忆中的东西以便循环单元内的gate（闸门）可以看到状态可能已经包含了接近于零的值。但是现在它想存储我们刚刚看到的单词not，所以它在这种状态下保存了一些非零值。<br>
&emsp;&emsp;然后，我们进入下一个时间步骤，在这里我们有一个单词a；这个词也没有太多的信息，所以它可能只是被忽略了。它只是复制到状态。<br>
&emsp;&emsp;现在，我们有了very一词，这表明，无论存在什么情绪，都可能是一种强烈的情绪，因此，现在知道我们已经看到的循环单位，not和very。它以某种方式将它存储在它的记忆状态中。<br>
&emsp;&emsp;在接下来的时间步骤里，我们看到了good这个词，所以现在网络知道not very good，它想，哦，这可能是一种消极情感！因此，它将该值存储在记忆状态中。<br>
&emsp;&emsp;然后，在最后的时间步骤里，我们看到movie，而这并不是真正相关的，所以它可能只是被忽略了。<br>
&emsp;&emsp;接下来，我们使用循环单元内的另一个gate（闸门）输出记忆状态的内容，然后用Sigmoid函数(这里不显示)处理它。我们得到0到1之间的输出值。<br>
&emsp;&emsp;我们的想法是，我们想通过网络电影数据库中成千上万的电影评论来训练这个网络，在这里，对于每一个输入文本，我们都给出了正反两方面的真实情感价值。然后，我们希望TensorFlow找出循环单元内部的gate（闸门）应该是什么，以便它们能够准确地将这个输入文本映射到正确的情感：<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter12/chapter12_image/picture4.png)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;图4：本章实现使用的架构<br>
&emsp;&emsp;我们将在这个实现中使用的RNN的体系结构是一个RNN类型的三层体系结构。在第一层，我们刚才解释的那部分已经发生了，只是现在我们需要在每个时间步骤输出循环单元的值。然后，我们收集了一个新的数据序列，即第一个循环层的输出。接下来，我们可以将它输入到第二层，因为循环单元需要输入数据的序列(我们从第一层得到的输出和我们想要输入到第二层的输出是一些浮点值，我们并不真正理解这些浮点值的含义)。这在RNN中有意义，但这不是我们人类所能理解的。然后，在第二层进行类似的处理。<br>
&emsp;&emsp;因此，首先，我们将这个循环单元的内部记忆状态初始化为0，然后，我们从第一个循环层获取第一个输出，并输入它。我们用这个循环单元内的gate（闸门）处理它，更新状态，将第一层的循环单元的输出作为第二个单词is，并使用它作为输入和内部记忆状态。我们继续这样做，直到我们处理完整个序列，然后我们收集第二个循环层的所有输出。我们使用它们作为第三个循环层的输入，在那里我们进行类似的处理。但是在这里，我们只需要最后一步的输出，这是对到目前为止所提供的所有内容的一种总结。然后，我们把它输出到一个完全连接的层，我们在这里不显示。最后，我们有Sigmoid激活函数，因此我们得到了一个介于0和1之间的值，它分别代表消极情感和积极情感。<br>
### 2、梯度爆炸和消失——回顾
&emsp;&emsp;我们在前一章中提到，有一种现象叫做梯度值的爆炸和消失，这在RNNs中是非常重要的。让我们回过头来看图1；这个流程图解释了这个现象是什么。<br> 
&emsp;&emsp;假设我们在这个数据集中有一个包含500个单词的文本，我们将使用它来实现我们的情感分析。在每一时间步骤中，我们以循环的方式在循环单元中应用内部gate（闸门）；因此，如果有500个单词，我们将应用这些gates（闸门）500次来更新循环单元的内部记忆状态。<br>
&emsp;&emsp;我们知道，神经网络的训练方式是使用梯度的反向传播，所以我们有一些损失函数，它得到神经网络的输出，然后得到我们对给定输入文本的真正输出。我们希望最小化这个损失值，以便神经网络的实际输出与这个特定输入文本的期望输出相对应。因此，我们需要取这个损失函数的梯度，相对于这些循环单元内的权重，这些权重用于更新内部状态并最终输出值的gates（闸门）。<br>
&emsp;&emsp;现在，gate（闸门）被应用了大约500次，如果其中有一个乘法，我们得到的实质是一个指数函数。所以，你用它自己乘以500倍，如果这个值略小于1，那么它就会很快消失或者失去。同样，如果一个略大于1的值与其自身相乘500倍，它就会爆炸。<br>
&emsp;&emsp;唯一能活过500次乘法的值是0和1。它们将保持不变，因此循环单位实际上比你在这里看到的要复杂得多。这是一个抽象的想法，我们想要以某种方式映射内部记忆状态和输入，以更新内部记忆状态并输出一些值——但实际上，我们需要非常小心地通过这些gates（闸门）向后传播梯度，这样我们就不会在许多时间步骤中出现指数乘法。我们也鼓励你们看一些关于循环单位的数学定义的教程。<br>
## 二、情感分析——模型实现
&emsp;&emsp;我们已经看到了如何实现RNNs的LSTM变体的所有细节和部分。为了让事情变得更令人兴奋，我们将使用一个名为Keras的更高级别的API。<br>
&emsp;&emsp;"Keras"是一个高级的神经网络API（应用程序编程接口），用Python编写，能够运行在TensorFlow、CNTK或Theano之上。它是以快速试验为重点开发的。能够在尽可能短的时间内从一个想法到另一个结果是做好研究的关键。”—Keras网站<br>
&emsp;&emsp;所以，Keras只是TensorFlow和其他深度学习框架的包装器。它非常适合于原型开发和快速构建，但另一方面，它使您对代码的控制更少。我们将借此机会在Keras中实现这个情感分析模型，这样您就可以在TensorFlow和Keras中得到一个手动实现。您可以使用keras进行快速原型开发，而tensorFlow用于生产准备系统。<br>
&emsp;&emsp;对你来说，更有趣的消息是，你不必切换到一个完全不同的环境。现在可以将Keras作为TensorFlow中的一个模块访问，并导入包，如下所示：<br>
&emsp;&emsp;from tensorflow.python.keras.models import sequential<br>
&emsp;&emsp;from tensorflow.python.keras.layers import dense GRU Embedding<br>
&emsp;&emsp;from tensorflow.python.keras.optimizers import Adam<br>
&emsp;&emsp;from tensorflow.python.keras.preprocession.sequence import pad_sequences<br>
&emsp;&emsp;因此，让我们继续使用我们现在所称的TensorFlow中一个更抽象的模块，它将帮助我们非常快地创建深度学习解决方案。这是因为我们将用几行代码编写完整的深度学习解决方案。<br>
### 1、数据分析和预处理
#### （1）导入所需模块
```
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
```
#### （2）读取数据
```
import imdb
imdb.maybe_download_and_extract()     #下载并解压imdb数据集
```
```
input_text_train, target_train = imdb.load_data(train=True)
input_text_test,  target_test  = imdb.load_data(train=False)

print("Size of the trainig set: ", len(input_text_train))
print("Size of the testing set:  ", len(input_text_test))
```
**output**<br>
```
Size of the trainig set:  25000
Size of the testing set:   25000
```
从这个结果可以看到，这里训练数据与测试数据各有25000项。<br>
下面我们将举一个例子来看看数据集的输入以及输出外观。<br>
```
text_data = input_text_train + input_text_test
input_text_train[1]
```
**output**<br>
```
'Homelessness (or Houselessness as George Carlin stated) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. Most people think of the homeless as just a lost cause while worrying about things such as racism, the war on Iraq, pressuring kids to succeed, technology, the elections, inflation, or worrying if they\'ll be next to end up on the streets.<br /><br />But what if you were given a bet to live on the streets for a month without the luxuries you once had from a home, the entertainment sets, a bathroom, pictures on the wall, a computer, and everything you once treasure to see what it\'s like to be homeless? That is Goddard Bolt\'s lesson.<br /><br />Mel Brooks (who directs) who stars as Bolt plays a rich man who has everything in the world until deciding to make a bet with a sissy rival (Jeffery Tambor) to see if he can live in the streets for thirty days without the luxuries; if Bolt succeeds, he can do what he wants with a future project of making more buildings. The bet\'s on where Bolt is thrown on the street with a bracelet on his leg to monitor his every move where he can\'t step off the sidewalk. He\'s given the nickname Pepto by a vagrant after it\'s written on his forehead where Bolt meets other characters including a woman by the name of Molly (Lesley Ann Warren) an ex-dancer who got divorce before losing her home, and her pals Sailor (Howard Morris) and Fumes (Teddy Wilson) who are already used to the streets. They\'re survivors. Bolt isn\'t. He\'s not used to reaching mutual agreements like he once did when being rich where it\'s fight or flight, kill or be killed.<br /><br />While the love connection between Molly and Bolt wasn\'t necessary to plot, I found "Life Stinks" to be one of Mel Brooks\' observant films where prior to being a comedy, it shows a tender side compared to his slapstick work such as Blazing Saddles, Young Frankenstein, or Spaceballs for the matter, to show what it\'s like having something valuable before losing it the next day or on the other hand making a stupid bet like all rich people do when they don\'t know what to do with their money. Maybe they should give it to the homeless instead of using it like Monopoly money.<br /><br />Or maybe this film will inspire you to help others.'
```
```
target_train[1]
```
**output**:1.0<br>
这里的输出值为1，这意味着它是一个积极的情感。所以，无论是什么电影，这是一个积极的评论。<br>
#### （3）建立字典
&emsp;&emsp;现在，我们讨论tokenizer，这是处理原始数据的第一步，因为神经网络不能处理文本数据。Keras实现了所谓的tokenizer，用于构建词汇表并从单词映射到整数。<br>
```
num_top_words = 10000
tokenizer_obj = Tokenizer(num_words=num_top_words)     #使用Tokenizer建立单词数为10000的字典。
```
&emsp;&emsp;现在，我们从数据集中获取所有文本，并在文本上调用函数fit，按照每一个单词在影评中出现的次数进行排序，前10000名的单词会列入字典中。<br>
```
tokenizer_obj.fit_on_texts(text_data)
tokenizer_obj.word_index  #字典数据类型，显示每一个单词在所有文章中出现的次数的排名。
```
**output**
```
{'the': 1,
 'and': 2,
 'a': 3,
 'of': 4,
 'to': 5,
 'is': 6,
 'br': 7,
 'in': 8,
 'it': 9,
 'i': 10,
 'this': 11,
 'that': 12,
 'was': 13,
 'as': 14,
 'for': 15,
 'with': 16,
 'movie': 17,
 'but': 18,
 'film': 19,
 'on': 20,
 'not': 21,
 'you': 22,
 'are': 23,
 'his': 24,
 'have': 25,
 'be': 26,
 'one': 27,
 'he': 28,
 'all': 29,
 'at': 30,
 'by': 31,
 'an': 32,
 'they': 33,
 'so': 34,
 'who': 35,
 'from': 36,
 'like': 37,
 'or': 38,
 'just': 39,
 'her': 40,
 'out': 41,
 'about': 42,
 'if': 43,
 "it's": 44,
 'has': 45,
 'there': 46,
 'some': 47,
 'what': 48,
 'good': 49,
 'when': 50,
 'more': 51,
 'very': 52,
 'up': 53,
 'no': 54,
 'time': 55,
 'my': 56,
 'even': 57,
 'would': 58,
 'she': 59,
 'which': 60,
 'only': 61,
 'really': 62,
 'see': 63,
 ...
 'straight': 722,
 '8': 723,
 'whether': 724,
 'die': 725,
 'add': 726,
 'dialog': 727,
 'entertainment': 728,
 'above': 729,
 'sets': 730,
 'future': 731,
 'enjoyable': 732,
 'appears': 733,
 'near': 734,
 'space': 735,
 'easily': 736,
 'hate': 737,
 'soundtrack': 738,
 'bring': 739,
 'giving': 740,
 'lots': 741,
 'similar': 742,
 'romantic': 743,
 'george': 744,
 'supporting': 745,
 'release': 746,
 'mention': 747,
 'within': 748,
 ...
 'political': 992,
 'leading': 993,
 'reasons': 994,
 'portrayed': 995,
 'spent': 996,
 'telling': 997,
 'cover': 998,
 'outside': 999,
 'fighting': 1000,
 ...}
```
现在，每个单词都与一个整数相关联<br>
这里，the单词是数字1：<br>
`tokenizer_obj.word_index['the']`<br>
**output**:1<br>
这里，and是数字2：<br>
`tokenizer_obj.word_index['and']`<br>
**output**:2<br>
单词a是数字3：<br>
`tokenizer_obj.word_index['a']`<br>
**output**:3<br>
我们看到movie是数字17：<br>
`tokenizer_obj.word_index['movie']`<br>
**output**:17<br>
Film是数字19<br>
`tokenizer_obj.word_index['film']`<br>
**output**:19<br>
&emsp;&emsp;这意味着the是数据集中使用最多的词，而and是数据集中使用第二多的词。因此，每当我们想要将单词映射到整数tokens时，我们就会得到这些数字。
让我们试着以数字743为例，这是单词romantic：<br>
`tokenizer_obj.word_index['romantic']`<br>
**output**:743<br>
因此，每当我们在输入文本中看到单词romantic时，我们就将它映射到token整数743。<br>
下面我们再次使用tokenizer将训练集中第一个文本中的所有单词转换为整数tokens，指令及结果如下：<br>
`input_text_train[1]   #输入训练集中第一个文本`<br>
```
'Homelessness (or Houselessness as George Carlin stated) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. Most people think of the homeless as just a lost cause while worrying about things such as racism, the war on Iraq, pressuring kids to succeed, technology, the elections, inflation, or worrying if they\'ll be next to end up on the streets.<br /><br />But what if you were given a bet to live on the streets for a month without the luxuries you once had from a home, the entertainment sets, a bathroom, pictures on the wall, a computer, and everything you once treasure to see what it\'s like to be homeless? That is Goddard Bolt\'s lesson.<br /><br />Mel Brooks (who directs) who stars as Bolt plays a rich man who has everything in the world until deciding to make a bet with a sissy rival (Jeffery Tambor) to see if he can live in the streets for thirty days without the luxuries; if Bolt succeeds, he can do what he wants with a future project of making more buildings. The bet\'s on where Bolt is thrown on the street with a bracelet on his leg to monitor his every move where he can\'t step off the sidewalk. He\'s given the nickname Pepto by a vagrant after it\'s written on his forehead where Bolt meets other characters including a woman by the name of Molly (Lesley Ann Warren) an ex-dancer who got divorce before losing her home, and her pals Sailor (Howard Morris) and Fumes (Teddy Wilson) who are already used to the streets. They\'re survivors. Bolt isn\'t. He\'s not used to reaching mutual agreements like he once did when being rich where it\'s fight or flight, kill or be killed.<br /><br />While the love connection between Molly and Bolt wasn\'t necessary to plot, I found "Life Stinks" to be one of Mel Brooks\' observant films where prior to being a comedy, it shows a tender side compared to his slapstick work such as Blazing Saddles, Young Frankenstein, or Spaceballs for the matter, to show what it\'s like having something valuable before losing it the next day or on the other hand making a stupid bet like all rich people do when they don\'t know what to do with their money. Maybe they should give it to the homeless instead of using it like Monopoly money.<br /><br />Or maybe this film will inspire you to help others.'
```
```
input_train_tokens = tokenizer_obj.texts_to_sequences(input_text_train)  
                                            #将文本转换为整数tokens时，它将变成一个整数数组
np.array(input_train_tokens)
```
**output**
```
array([list([299, 6, 3, 1059, 202, 9, 2119, 30, 1, 167, 55, 14, 47, 79, 6274, 42, 368, 114, 138, 14, 5103, 56, 4515, 153, 8, 1, 4233, 5799, 469, 68, 5, 262, 12, 2072, 6, 72, 2556, 5, 614, 71, 6, 5103, 1, 5, 1897, 1, 5540, 1469, 35, 67, 63, 203, 140, 65, 1151, 1, 4, 1, 223, 871, 29, 3195, 68, 4, 1, 5510, 10, 677, 2, 65, 1469, 50, 10, 210, 1, 398, 8, 60, 3, 1425, 3345, 762, 5, 3491, 175, 1, 368, 10, 1220, 30, 299, 3, 360, 347, 3471, 145, 133, 5, 8306, 27, 4, 125, 5103, 1425, 2563, 5, 299, 10, 525, 12, 106, 1540, 4, 56, 599, 101, 12, 299, 6, 225, 3994, 48, 3, 2244, 12, 9, 213]),
       list([38, 14, 744, 3506, 45, 75, 32, 1771, 15, 153, 18, 110, 3, 1344, 5, 343, 143, 20, 1, 920, 12, 70, 281, 1228, 395, 35, 115, 267, 36, 166, 5, 368, 158, 38, 2058, 15, 1, 504, 88, 83, 101, 4, 1, 4339, 14, 39, 3, 432, 1148, 136, 8697, 42, 177, 138, 14, 2791, 1, 295, 20, 5276, 351, 5, 3029, 2310, 1, 38, 8697, 43, 3611, 26, 365, 5, 127, 53, 20, 1, 2032, 7, 7, 18, 48, 43, 22, 70, 358, 3, 2343, 5, 420, 20, 1, 2032, 15, 3, 3346, 208, 1, 22, 281, 66, 36, 3, 344, 1, 728, 730, 3, 3864, 1320, 20, 1, 1543, 3, 1293, 2, 267, 22, 281, 2734, 5, 63, 48, 44, 37, 5, 26, 4339, 12, 6, 2079, 7, 7, 3425, 2891, 35, 4446, 35, 405, 14, 297, 3, 986, 128, 35, 45, 267, 8, 1, 181, 366, 6951, 5, 94, 3, 2343, 16, 3, 7017, 3090, 5, 63, 43, 28, 67, 420, 8, 1, 2032, 15, 3082, 483, 208, 1, 43, 2802, 28, 67, 77, 48, 28, 487, 16, 3, 731, 1146, 4, 232, 51, 4161, 1, 20, 117, 6, 1334, 20, 1, 920, 16, 3, 20, 24, 4086, 5, 24, 170, 831, 117, 28, 185, 1562, 122, 1, 7951, 237, 358, 1, 31, 3, 100, 44, 407, 20, 24, 9597, 117, 911, 79, 102, 585, 3, 257, 31, 1, 389, 4, 5176, 2137, 4636, 32, 1222, 3303, 35, 189, 4287, 159, 2320, 40, 344, 2, 40, 8527, 6229, 1955, 4910, 2, 7720, 2618, 35, 23, 472, 328, 5, 1, 2032, 501, 4392, 213, 237, 21, 328, 5, 4805, 6768, 37, 28, 281, 115, 50, 109, 986, 117, 44, 557, 38, 2574, 505, 38, 26, 531, 7, 7, 136, 1, 112, 1906, 201, 5176, 2, 292, 1731, 5, 111, 10, 255, 114, 4541, 5, 26, 27, 4, 3425, 104, 117, 2557, 5, 109, 3, 202, 9, 276, 3, 4317, 486, 1107, 5, 24, 2347, 158, 138, 14, 8161, 186, 3889, 38, 15, 1, 504, 5, 119, 48, 44, 37, 263, 137, 4737, 159, 2320, 9, 1, 365, 254, 38, 20, 1, 79, 524, 232, 3, 364, 2343, 37, 29, 986, 83, 77, 50, 33, 89, 118, 48, 5, 77, 16, 65, 290, 273, 33, 142, 197, 9, 5, 1, 4339, 298, 4, 783, 9, 37, 290, 7, 7, 38, 273, 11, 19, 80, 5541, 22, 5, 343, 400]),
       list([513, 121, 113, 31, 2137, 4636, 116, 967, 824, 10, 25, 123, 107, 2, 112, 134, 8, 1688, 7418, 23, 336, 5, 619, 1, 5398, 20, 391, 6, 3, 360, 14, 49, 14, 231, 8, 8161, 1, 187, 20, 9106, 6, 81, 916, 100, 109, 3478, 4, 109, 3, 3157, 41, 24, 1488, 2, 109, 1, 2792, 4, 145, 3, 2792, 28, 546, 277, 152, 675, 4423, 3, 521, 36, 1, 305, 2354, 6639, 119, 6, 799, 133, 96, 14, 3, 1121, 6056, 35, 487, 5, 4709, 1, 6769, 24, 108, 6, 51, 71, 667, 1, 1447, 129, 2, 1, 129, 117, 1, 4339, 3, 2060, 23, 29, 55, 2112, 163, 15, 1, 3127, 129, 2, 1, 105, 191, 1000, 27, 11, 17, 217, 126, 252, 55, 10, 63, 9, 60, 6, 179, 403]),
       ...,
       list([10, 210, 238, 316, 30, 1, 19, 1398, 2, 9, 13, 27, 643, 1415, 1415, 84, 1, 773, 13, 4346, 962, 1, 9432, 4, 314, 6094, 8, 3, 4180, 4508, 17, 13, 1141, 2, 109, 3, 326, 934, 145, 21, 4, 1473, 4, 1, 1733, 10, 13, 3132, 5, 131, 52, 2120, 5, 808, 11, 17, 41, 56, 1237, 912, 1240, 5, 1, 2112, 422, 1, 773, 45, 1050, 18, 1, 3312, 23, 1054, 1, 150, 2760, 57, 6094, 6, 1957, 47, 58, 131, 12, 44, 3, 203, 2708, 4, 1, 1123, 181, 8, 1, 176, 12, 1, 9961, 4, 1, 102, 2755, 3, 1272, 2, 29, 12, 3708, 18, 9, 39, 162, 1, 223, 17, 39, 37, 1, 129, 117, 6094, 217, 3433, 9, 612, 1514, 3061, 10, 292, 1012, 231, 396, 18, 130, 709, 73, 1343, 5, 229, 5174, 15, 40, 10, 154, 21, 15, 3, 780, 59, 13, 34, 868, 29, 1, 95, 2, 10, 339, 12, 1, 1304, 13, 1186, 69, 9, 6, 49, 864, 18, 161, 1502, 2171, 346, 10, 206, 987, 12, 1, 2701, 30, 1, 1398, 292, 34, 49, 34, 273, 10, 1055, 41, 137, 133, 18, 30, 1, 127, 4, 1, 17, 10, 413, 343, 533, 7210, 229, 37, 147, 9324, 172, 3910, 51, 612, 1, 86, 129, 8, 1, 1393, 6, 1278, 319, 2, 29, 18, 10, 66, 1, 3037, 7561, 4, 147, 3, 1150, 2, 51, 348, 318, 4, 1, 129, 8, 1, 6738, 1484, 734, 1, 283, 2368, 4109, 8, 7358, 9071, 957, 1194, 16, 1, 2850, 830, 2, 1205, 1509, 1, 1269, 207, 1, 825, 851, 54, 10, 39, 413, 76, 82, 11, 17, 96, 74]),
       list([47, 104, 12, 22, 1185, 53, 15, 3, 6677, 477, 41, 5, 26, 248, 49, 1064, 104, 613, 4245, 4, 3805, 1049, 2, 283, 17, 12, 70, 78, 18, 259, 613, 47, 3418, 104, 2829, 400, 1, 6627, 4, 65, 9097, 6, 590, 37, 1, 1025, 6712, 7, 7, 1, 61, 1163, 148, 10, 67, 131, 42, 11, 19, 6, 12, 44, 1273, 734, 14, 637, 14, 1, 7943, 4, 2094, 79, 71, 12, 92, 3, 52, 4046, 7710, 19, 16, 1, 1890, 4, 3, 322, 2146, 31, 3, 7, 7, 44, 75, 3145, 3320, 18, 22, 62, 77, 25, 5, 3100, 42, 3, 19, 12, 5568, 4, 1, 205, 82, 92, 1098, 708, 34, 31, 1, 55, 9, 217, 5, 1, 1025, 2904, 841, 107, 9, 29, 472, 7, 7, 74, 17, 455, 80, 25, 3, 1964, 147, 1, 3079, 1726, 2448, 2649, 2, 1, 9769, 113, 4, 1, 174, 261, 1, 35, 13, 34, 74, 28, 115, 94, 68, 423, 3, 226, 2, 3495, 5, 15, 385, 8, 5, 1, 1174, 4, 177, 31, 7327, 3, 2549, 34, 74, 9, 436, 37, 10, 13, 147, 47, 1487, 53, 344, 373, 4, 3, 7333, 1231, 7, 7, 372, 372, 517, 2019, 29, 90, 517, 37, 11, 50, 2019, 1944, 3, 1075, 4, 3, 355, 18, 1114, 282, 2, 132, 104, 8, 260, 1219, 793]),
       list([11, 6, 27, 4, 1, 7047, 104, 198, 123, 107, 9, 7576, 122, 801, 123, 543, 4, 704, 2, 1019, 5, 94, 3, 954, 4, 93, 29, 7, 7, 222, 21, 3, 689, 49, 347, 38, 108, 8, 1, 223, 954, 43, 46, 13, 3, 111, 9, 13, 32, 2, 14, 225, 14, 113, 269, 222, 161, 49, 5, 131, 34, 1811, 131, 161, 10, 1230, 2192, 386, 85, 11, 543, 4, 1832, 217, 1034, 2, 160, 613, 124, 1851, 1219, 21, 30, 47, 927, 101, 56, 545, 11, 62, 6, 3, 3609, 4, 2, 656, 9, 3, 254, 92, 590, 37, 11, 12, 45, 83, 1, 1519, 285, 37, 3, 335, 279, 19, 30, 224, 43, 22, 25, 9, 22, 755, 1002, 125, 55, 38, 290, 89, 451, 125, 55, 11, 6, 1377])],
      dtype=object)
```
这里，单词homelessness变成了数字299，单词or变成了数字6，依此类推。<br>
同样，我们还需要转换文本的其余部分，代码如下:<br>
`input_test_tokens = tokenizer_obj.texts_to_sequences(input_text_test)  #将文本转换为数字列表`<br>
#### （4）数字列表截长补短
&emsp;&emsp;现在有另一个问题，因为tokens序列的长度取决于原始文本的长度，即使循环单元可以处理任意长度的序列。但是TensorFlow的工作方式是，批处理中的所有数据都需要具有相同的长度。<br>
&emsp;&emsp;因此，我们需要确保整个数据集中的所有序列都具有相同的长度，或者编写一个自定义数据生成器，以确保单个批处理中的序列具有相同的长度。现在，要确保数据集中的所有序列都具有相同的长度也比较简单，但问题是存在一些异常值。假定我们认为超过2200个单词的句子太长，如果我们有超过2200个单词的句字，那么我们的记忆就会受到很大的伤害。因此，我们必须要做出妥协。<br>
&emsp;&emsp;首先，我们需要计算每个输入序列中的所有单词或tokens。从下列结果我们可以看到，一个序列中的平均单词数大约是221个：<br>
```
total_num_tokens = [len(tokens) for tokens in input_train_tokens + input_test_tokens]
total_num_tokens = np.array(total_num_tokens)
np.mean(total_num_tokens)     #计算所有数字序列的平均单词数
```
**output**:221.27716<br>
从下列结果，我们可以看到这些序列中最大的单词数超过2200<br>
`np.max(total_num_tokens)`<br>
**output**:2209<br>
&emsp;&emsp;平均值221和最大值2209之间有巨大的差别，如果我们只是在数据集中填充所有的句子，以便它们会有2209个tokens，那么我们就会浪费大量的内存。如果说我们有一个包含数百万个文本序列的数据集，这将会是一个很大的问题。<br>
&emsp;&emsp;所以我们要做出一个妥协。我们将填充所有序列，并截断那些太长的序列，这样它们就有544个单词了。我们的计算方法是：取数据集中所有序列的平均单词数，并添加两个标准差，代码如下：<br>
```
max_num_tokens = np.mean(total_num_tokens) + 2 * np.std(total_num_tokens)   #均值加两个标准差
max_num_tokens = int(max_num_tokens)
max_num_tokens
```
**output**:544<br>
添加标准差后，我们每一个序列的单词数将保留为544个。<br>
`np.sum(total_num_tokens < max_num_tokens)/len(total_num_tokens)  #小于544个单词的序列个数占所有序列个数的比例`<br>
**output**:0.9453
从这里我们可以看到，大约有95%的文本长度均为544，只有5%的文本比544个单词长。<br>

&emsp;&emsp;现在我们知道，在Keras中称这些为函数。它们要么填充太短的序列(所以它们只添加零)，要么截断太长的序列(如果文本太长，基本上只需要切断一些单词)。 然而，需要注意的是：我们到底是在序列前还是在序列后模式下进行填充和截断呢？<br>
&emsp;&emsp;因此，假设我们有一个整数tokens序列，因为它太短了，我们想要填充它。我们可以：要么在开头放置这些零，以便在结尾处有实际的整数tokens。或者用相反的方式来做，这样我们所有的数据都在开始，所有的零在结尾。但是，如果我们回到前面的RNN流程图，我们知道它是一步步地处理序列，所以如果我们开始处理零，它可能没有任何意义，内部状态可能只是保持为零。因此，每当它看到一个特定单词的整数token时，它就会知道，好的，现在我们开始处理数据。然而，如果所有的零都在末尾，我们就会开始处理所有的数据；那么我们就会在循环单元中有一些内部状态。现在，我们看到了大量的零，这可能会破坏我们刚刚计算出来的内部状态。这就是为什么在开始时填充零可能是个好主意。<br>
&emsp;&emsp;另一个问题是关于截断文本。如果文本很长，我们将截断它，以使它适合于文字，或任何数字。现在，想象一下，我们在中间的某个地方抓住了一个句子，它写的是this very good movie，或者this is not。当然，我们只在很长的序列中这样做，但是我们有可能失去正确分类这篇文章所必需的信息。因此，这是我们在截断输入文本时需要做出妥协。一个比较好的方法是创建一个批处理并在批处理中填充文本。因此，当我们看到一个很长的序列时，我们会把其他序列放置在相同的长度上。但我们不需要将所有这些数据存储在内存中，因为大部分数据都是浪费的。<br>
&emsp;&emsp;接下来让我们返回并转换整个数据集，使其被截断和填充；它是一个大的数据矩阵：<br>
```
seq_pad = 'pre'        #pre表示从起始填充或截断

input_train_pad = pad_sequences(input_train_tokens, maxlen=max_num_tokens,
                            padding=seq_pad, truncating=seq_pad)           #padding表示填充，truncating表示截断

input_test_pad = pad_sequences(input_test_tokens, maxlen=max_num_tokens,
                           padding=seq_pad, truncating=seq_pad)
```
我们检查这个矩阵的形状：<br>
`input_train_pad.shape`<br>
**output**:(25000,544)<br>
`input_test_pad.shape`<br>
**output**:(25000,544)<br>
下面，让我们看看填充前后的特定示例tokens：<br>
填充前的数字矩阵如下：<br>
`np.array(input_train_tokens[1])`<br>
**output**<br>
```array([  38,   14,  744, 3506,   45,   75,   32, 1771,   15,  153,   18,
        110,    3, 1344,    5,  343,  143,   20,    1,  920,   12,   70,
        281, 1228,  395,   35,  115,  267,   36,  166,    5,  368,  158,
         38, 2058,   15,    1,  504,   88,   83,  101,    4,    1, 4339,
         14,   39,    3,  432, 1148,  136, 8697,   42,  177,  138,   14,
       2791,    1,  295,   20, 5276,  351,    5, 3029, 2310,    1,   38,
       8697,   43, 3611,   26,  365,    5,  127,   53,   20,    1, 2032,
          7,    7,   18,   48,   43,   22,   70,  358,    3, 2343,    5,
        420,   20,    1, 2032,   15,    3, 3346,  208,    1,   22,  281,
         66,   36,    3,  344,    1,  728,  730,    3, 3864, 1320,   20,
          1, 1543,    3, 1293,    2,  267,   22,  281, 2734,    5,   63,
         48,   44,   37,    5,   26, 4339,   12,    6, 2079,    7,    7,
       3425, 2891,   35, 4446,   35,  405,   14,  297,    3,  986,  128,
         35,   45,  267,    8,    1,  181,  366, 6951,    5,   94,    3,
       2343,   16,    3, 7017, 3090,    5,   63,   43,   28,   67,  420,
          8,    1, 2032,   15, 3082,  483,  208,    1,   43, 2802,   28,
         67,   77,   48,   28,  487,   16,    3,  731, 1146,    4,  232,
         51, 4161,    1,   20,  117,    6, 1334,   20,    1,  920,   16,
          3,   20,   24, 4086,    5,   24,  170,  831,  117,   28,  185,
       1562,  122,    1, 7951,  237,  358,    1,   31,    3,  100,   44,
        407,   20,   24, 9597,  117,  911,   79,  102,  585,    3,  257,
         31,    1,  389,    4, 5176, 2137, 4636,   32, 1222, 3303,   35,
        189, 4287,  159, 2320,   40,  344,    2,   40, 8527, 6229, 1955,
       4910,    2, 7720, 2618,   35,   23,  472,  328,    5,    1, 2032,
        501, 4392,  213,  237,   21,  328,    5, 4805, 6768,   37,   28,
        281,  115,   50,  109,  986,  117,   44,  557,   38, 2574,  505,
         38,   26,  531,    7,    7,  136,    1,  112, 1906,  201, 5176,
          2,  292, 1731,    5,  111,   10,  255,  114, 4541,    5,   26,
         27,    4, 3425,  104,  117, 2557,    5,  109,    3,  202,    9,
        276,    3, 4317,  486, 1107,    5,   24, 2347,  158,  138,   14,
       8161,  186, 3889,   38,   15,    1,  504,    5,  119,   48,   44,
         37,  263,  137, 4737,  159, 2320,    9,    1,  365,  254,   38,
         20,    1,   79,  524,  232,    3,  364, 2343,   37,   29,  986,
         83,   77,   50,   33,   89,  118,   48,    5,   77,   16,   65,
        290,  273,   33,  142,  197,    9,    5,    1, 4339,  298,    4,
        783,    9,   37,  290,    7,    7,   38,  273,   11,   19,   80,
       5541,   22,    5,  343,  400])
```
填充之后，这个示例如下所示：<br>
`input_train_pad[1]`<br>
**output**<br>
```
array([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
         38,   14,  744, 3506,   45,   75,   32, 1771,   15,  153,   18,
        110,    3, 1344,    5,  343,  143,   20,    1,  920,   12,   70,
        281, 1228,  395,   35,  115,  267,   36,  166,    5,  368,  158,
         38, 2058,   15,    1,  504,   88,   83,  101,    4,    1, 4339,
         14,   39,    3,  432, 1148,  136, 8697,   42,  177,  138,   14,
       2791,    1,  295,   20, 5276,  351,    5, 3029, 2310,    1,   38,
       8697,   43, 3611,   26,  365,    5,  127,   53,   20,    1, 2032,
          7,    7,   18,   48,   43,   22,   70,  358,    3, 2343,    5,
        420,   20,    1, 2032,   15,    3, 3346,  208,    1,   22,  281,
         66,   36,    3,  344,    1,  728,  730,    3, 3864, 1320,   20,
          1, 1543,    3, 1293,    2,  267,   22,  281, 2734,    5,   63,
         48,   44,   37,    5,   26, 4339,   12,    6, 2079,    7,    7,
       3425, 2891,   35, 4446,   35,  405,   14,  297,    3,  986,  128,
         35,   45,  267,    8,    1,  181,  366, 6951,    5,   94,    3,
       2343,   16,    3, 7017, 3090,    5,   63,   43,   28,   67,  420,
          8,    1, 2032,   15, 3082,  483,  208,    1,   43, 2802,   28,
         67,   77,   48,   28,  487,   16,    3,  731, 1146,    4,  232,
         51, 4161,    1,   20,  117,    6, 1334,   20,    1,  920,   16,
          3,   20,   24, 4086,    5,   24,  170,  831,  117,   28,  185,
       1562,  122,    1, 7951,  237,  358,    1,   31,    3,  100,   44,
        407,   20,   24, 9597,  117,  911,   79,  102,  585,    3,  257,
         31,    1,  389,    4, 5176, 2137, 4636,   32, 1222, 3303,   35,
        189, 4287,  159, 2320,   40,  344,    2,   40, 8527, 6229, 1955,
       4910,    2, 7720, 2618,   35,   23,  472,  328,    5,    1, 2032,
        501, 4392,  213,  237,   21,  328,    5, 4805, 6768,   37,   28,
        281,  115,   50,  109,  986,  117,   44,  557,   38, 2574,  505,
         38,   26,  531,    7,    7,  136,    1,  112, 1906,  201, 5176,
          2,  292, 1731,    5,  111,   10,  255,  114, 4541,    5,   26,
         27,    4, 3425,  104,  117, 2557,    5,  109,    3,  202,    9,
        276,    3, 4317,  486, 1107,    5,   24, 2347,  158,  138,   14,
       8161,  186, 3889,   38,   15,    1,  504,    5,  119,   48,   44,
         37,  263,  137, 4737,  159, 2320,    9,    1,  365,  254,   38,
         20,    1,   79,  524,  232,    3,  364, 2343,   37,   29,  986,
         83,   77,   50,   33,   89,  118,   48,    5,   77,   16,   65,
        290,  273,   33,  142,  197,    9,    5,    1, 4339,  298,    4,
        783,    9,   37,  290,    7,    7,   38,  273,   11,   19,   80,
       5541,   22,    5,  343,  400])
```
&emsp;&emsp;了解了文本转换为数字列表之后，接下来，我们来看一个向后映射的功能，即从整数tokens映射回文本单词。我们只需要用一个非常简单的助手函数即可，代码如下:<br>
```
index = tokenizer_obj.word_index      #数字列表
index_inverse_map = dict(zip(index.values(), index.keys()))    #zip函数将键和值反过来

def convert_tokens_to_string(input_tokens):          
    input_words = [index_inverse_map[token] for token in input_tokens if token != 0]   #将token整数转换为单词
    combined_text = " ".join(input_words)  #加入所有的单词

    return combined_text
 ```
例如，数据集中的原始文本如下：<br>
`input_text_train[1]`<br>
**output**<br>
```
'Homelessness (or Houselessness as George Carlin stated) has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school, work, or vote for the matter. Most people think of the homeless as just a lost cause while worrying about things such as racism, the war on Iraq, pressuring kids to succeed, technology, the elections, inflation, or worrying if they\'ll be next to end up on the streets.<br /><br />But what if you were given a bet to live on the streets for a month without the luxuries you once had from a home, the entertainment sets, a bathroom, pictures on the wall, a computer, and everything you once treasure to see what it\'s like to be homeless? That is Goddard Bolt\'s lesson.<br /><br />Mel Brooks (who directs) who stars as Bolt plays a rich man who has everything in the world until deciding to make a bet with a sissy rival (Jeffery Tambor) to see if he can live in the streets for thirty days without the luxuries; if Bolt succeeds, he can do what he wants with a future project of making more buildings. The bet\'s on where Bolt is thrown on the street with a bracelet on his leg to monitor his every move where he can\'t step off the sidewalk. He\'s given the nickname Pepto by a vagrant after it\'s written on his forehead where Bolt meets other characters including a woman by the name of Molly (Lesley Ann Warren) an ex-dancer who got divorce before losing her home, and her pals Sailor (Howard Morris) and Fumes (Teddy Wilson) who are already used to the streets. They\'re survivors. Bolt isn\'t. He\'s not used to reaching mutual agreements like he once did when being rich where it\'s fight or flight, kill or be killed.<br /><br />While the love connection between Molly and Bolt wasn\'t necessary to plot, I found "Life Stinks" to be one of Mel Brooks\' observant films where prior to being a comedy, it shows a tender side compared to his slapstick work such as Blazing Saddles, Young Frankenstein, or Spaceballs for the matter, to show what it\'s like having something valuable before losing it the next day or on the other hand making a stupid bet like all rich people do when they don\'t know what to do with their money. Maybe they should give it to the homeless instead of using it like Monopoly money.<br /><br />Or maybe this film will inspire you to help others.'
```
如果我们使用一个帮助函数将tokens转换回文本单词，我们将得到以下文本：<br>
`convert_tokens_to_string(input_train_tokens[1])     #将数字列表转化为字符串`<br>
**output**<br>
```
"or as george stated has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school work or vote for the matter most people think of the homeless as just a lost cause while worrying about things such as racism the war on iraq kids to succeed technology the or worrying if they'll be next to end up on the streets br br but what if you were given a bet to live on the streets for a month without the you once had from a home the entertainment sets a bathroom pictures on the wall a computer and everything you once treasure to see what it's like to be homeless that is lesson br br mel brooks who directs who stars as plays a rich man who has everything in the world until deciding to make a bet with a sissy rival to see if he can live in the streets for thirty days without the if succeeds he can do what he wants with a future project of making more buildings the on where is thrown on the street with a on his leg to his every move where he can't step off the sidewalk he's given the by a after it's written on his forehead where meets other characters including a woman by the name of molly ann warren an ex dancer who got divorce before losing her home and her pals sailor howard morris and teddy wilson who are already used to the streets they're survivors isn't he's not used to reaching mutual like he once did when being rich where it's fight or flight kill or be killed br br while the love connection between molly and wasn't necessary to plot i found life stinks to be one of mel films where prior to being a comedy it shows a tender side compared to his slapstick work such as blazing young frankenstein or for the matter to show what it's like having something valuable before losing it the next day or on the other hand making a stupid bet like all rich people do when they don't know what to do with their money maybe they should give it to the homeless instead of using it like money br br or maybe this film will inspire you to help others"
```
可以看到，除了标点符号和其他符号，其他基本一样。<br>
### 2、构建模型
&emsp;&emsp;现在，我们需要创建RNN，我们将在Keras中用所谓的sequential模型来实现。<br>
&emsp;&emsp;这个体系结构的第一层是所谓的嵌入层。如果我们回顾一下图1中的流程图，我们刚才所做的就是将原始输入文本转换为整数tokens。但是我们仍然不能将它输入到RNN，因此我们必须将其转换为嵌入向量，即介于-1和1之间的值。它们可以在一定程度上超过这个范围，但通常在-1到1之间，这是我们可以在神经网络中处理的数据。<br>
&emsp;&emsp;我们需要决定每个向量的长度，例如，token11被转换成一个实值向量，我们可以将长度设置为10（这个长度实际上是非常短的，通常，它在100到300之间)。<br>
#### （1）加入嵌入层
 这里，我们将嵌入大小设置为8，然后使用Keras将该嵌入层添加到RNN中。这必须是网络的第一层：<br>
```
rnn_type_model = Sequential()
 embedding_layer_size = 8 #typical value for this should be between 200 and 300
rnn_type_model.add(Embedding(input_dim=num_top_words,
                    output_dim=embedding_layer_size,
                    input_length=max_num_tokens,
                    name='embedding_layer'))
```
#### （2）建立RNN模型
&emsp;&emsp;然后，我们可以添加第一个循环层，我们将使用所谓的gated recurrent unit(GRU)。通常情况下，我们看到人们会使用所谓的LSTM，但其他人似乎认为GRU更好，因为LSTM内部有多余的gates。实际上，更简单的代码在更少的gates上工作更好。因此，我们这里采用GRU，让我们定义我们的GRU架构，我们希望输出维数为16，我们需要返回序列：<br>
```
rnn_type_model.add(GRU(units=16, return_sequences=True))   #添加第一个循环层

rnn_type_model.add(GRU(units=8, return_sequences=True))   #添加第二个循环层

rnn_type_model.add(GRU(units=4))              #添加第三个循环层

rnn_type_model.add(Dense(1, activation='sigmoid'))     #添加输出层，用Sigmoid激活函数处理，得到0~1之间的值

model_optimizer = Adam(lr=1e-3)

rnn_type_model.compile(loss='binary_crossentropy',    #设置损失函数
              optimizer=model_optimizer,           #使用Adam优化器
              metrics=['accuracy'])         #设置评估模型的方式是准确率
```
&emsp;&emsp;这里我们添加了三个循环层，最后一个dense层只给出GRU的最终输出，而不是一个完整的输出序列。这里的输出将被输入到一个完全连接或dense层中，该层应该为每个输入序列输出一个值。因为使用Sigmoid激活函数处理，所以它会输出一个介于0到1之间的值。我们在这里使用的是ADAM优化器，并且损失函数是RNN的输出和训练集的实际类值之间的二进制交叉熵，这个值要么是0，要么是1：<br>
&emsp;&emsp;现在，我们查看模型的外观，如下：<br>
`rnn_type_model.summary()     #查看模型的外观`<br>
**output**<br>
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_layer (Embedding)  (None, 544, 8)            80000     
_________________________________________________________________
gru (GRU)                    (None, 544, 16)           1200      
_________________________________________________________________
gru_1 (GRU)                  (None, 544, 8)            600       
_________________________________________________________________
gru_2 (GRU)                  (None, 4)                 156       
_________________________________________________________________
dense (Dense)                (None, 1)                 5         
=================================================================
Total params: 81,961
Trainable params: 81,961
Non-trainable params: 0
_________________________________________________________________
```
从该模型我们可以知道，我们有一个嵌入层，三个循环单元和一个dense层。注意，这没有太多的参数。<br>
### 3、模型训练和结果分析
#### （1）训练模型
现在我们开始对模型进行训练，代码如下：<br>
```
rnn_type_model.fit(input_train_pad, target_train,
          validation_split=0.05, epochs=3, batch_size=64)
 #validation_split=0.05表示在训练数据中分出5%作为验证数据，其余95%作为训练数据
 #epochs=3表示执行3个训练周期
 #batch_size=64表示每一批次64项数据
```
**output**<br>
```
Train on 23750 samples, validate on 1250 samples
Epoch 1/3
23750/23750 [==============================] - 194s 8ms/step - loss: 0.5021 - acc: 0.7364 - val_loss: 0.5293 - val_acc: 0.7648
Epoch 2/3
23750/23750 [==============================] - 188s 8ms/step - loss: 0.2781 - acc: 0.8945 - val_loss: 0.2628 - val_acc: 0.8880
Epoch 3/3
23750/23750 [==============================] - 186s 8ms/step - loss: 0.2111 - acc: 0.9265 - val_loss: 0.4714 - val_acc: 0.8032
```
从这里我们可以看到，共执行了3个训练周期，其误差越来越小，准确率越来越高。<br>
#### （2）评估模型准确率
```
model_result=rnn_type_model.evaluate(input_test_pad,target_test)
25000/25000 [==============================] - 71s 3ms/step
accuracy:85.26%
```
#### （3）进行预测
现在，让我们看一个错误分类文本的例子<br>
 首先，我们计算测试集中前1000个序列的预测类，然后取实际的类值。我们将它们进行比较，并得到存在这种不匹配的索引列表：<br>
```
target_predicted = rnn_type_model.predict(x=input_test_pad[0:1000])
target_predicted = target_predicted.T[0]
```
使用截止阈值表示上述所有值都为正值，其他值将被认为负值：<br>
`class_predicted = np.array([1.0 if prob>0.5 else 0.0 for prob in target_predicted])`<br>
现在，让我们得到这1000条序列的实际类：<br>
`class_actual = np.array(target_test[0:1000])`<br>
接下来让我们从输出中获取不正确的样本，代码及结果如下<br>
```
incorrect_samples = np.where(class_predicted != class_actual)
incorrect_samples = incorrect_samples[0]
len(incorrect_samples)
```
**output**:82<br>
因此，我们发现这些文本中有82篇被错误地分类；这是我们在这里计算的1000个文本的8.2%。让我们看一下第一个错误分类的文本，代码及结果如下：<br>
```
index = incorrect_samples[0]
index
```
**output**:35<br>
```
incorrect_predicted_text = input_text_test[index]
incorrect_predicted_text
```
```
'BEING Warner Brothers\' second historical drama featuring Civil War and Battle of the Little Big Horn, General George Armstrong Custer, THEY DIED WITH THEIR BOOTS ON (Warner Brothers, 1941) was the far more accurate of the two; especially when contrasted with SANTA FE TRAIL (Warner Brothers, 1940), which really didn\'t set the bar very high.<br /><br />ALTHOUGH both pictures were starring vehicles for Errol Flynn, there was a change in the casting the part of General Custer. Whereas it was "Dutch", himself, Ronald Reagan portraying the flamboyant, egomaniacal Cavalryman in the earlier picture, with Mr. Flynn playing Virginian and later Confederate Hero General, J.E.B. (or Jeb) Stuart; Errol took on the Custer part for THEY DIED WITH THEIR BOOTS ON.<br /><br />ONCE again, the Warner Brothers\' propensity for using a large number of reliable character actors from the "Warner\'s Repertory Company" are employed in giving the film a sort of authenticity, and all is really happening right before our very own eyes. Major roles are taken by some better known actors and actresses, such as: Elizabeth Bacon/Mrs. Custer (co-star Olivia de Havilland), Ned Sharpe (Arthur Kennedy), Samuel Bacon (Gene Lockhart), Chief Crazy Horse (Anthony Quinn), "Californy" (Charlie Grapwin), Major Taipe (Stanley Ridges), General Phillip Sheridan (John Litel), Callie (the Bacon\'s Maid, Hattie McDaniel). <br /><br />THE rest of the cast is just chock full of uncredited, though skilled players such as: Joe Sawyer, Eleanor Parker, Minor Watson, Tod Andrews, Irving Bacon, Roy Barcroft, Lane Chandler, Spencer Charters, Frank Ferguson, Francis Ford, William Forrest, George Eldridge, Russell Hicks, William Hopper, Hoppity Hooper, Eddie Keane, Fred Kelsey, Sam McDaniel, Patrick McVey, Frank Orth, Eddie Parker, Addison Richards, Ray Teal, Jim Thorpe (All-American, himself), Minerva Urecal, Dick Wessel, Gig Young and many, many more.<br /><br />THE film moves very quickly, particularly in the early goings; then sort of slows down out of necessity as the story moves along to the Post Civil War years, the assignment of Custer as a Colonel in the 7th Cavalry and the ultimate destiny at the Little Big Horn, in Montana. Under the guidance of Director, Griffith Veteran, Raoul Walsh, the film hits a greatly varied array of emotions; from the very serious, exciting battle scenes and convincing historical scenes; looking as if they were Matthew Brady Civil War Photos. As with most any of Mr. Walsh\'s films, he punctuates and expedites the end of many a scene with a little humor; but not going overboard and thus risking the chance of turning the film into a comedy (farce, actually).<br /><br />AS previously mentioned, this is much more factual than its predecessor, SANTA FE TRAIL (last time we\'ll mention it, honest Schultz, Scout\'s Honor!). However, that is not to say that it wasn\'t without a few little bits of "Artistic and Literary License; as indeed, just about any Biopic will have. It would be impossible to make any similar type of film if indeed every fact and incident were to be tried to be included in the screenplay. Perhaps the most erroneous inclusion as well as the most obvious invocation of Literary License is that business about Custer\'s being accidentally promoted to the rank of Brigadier General. It just didn\'t happen that way, yet the "gag" both helped the film to move along; while it underscored the whole light, carefree feeling that permeated the early part of the film.<br /><br />DIRECTOR Walsh and Mr. Flynn collaborated in giving us what would seem to be a characterization of this legendary Civil War Hero that was very close to the real life man. And they did this on top of the recreation of an incident, being the Massacre by the Lakota Sioux, the Cheyenne and the Fukowi of Custer and his 7th Cavalry at the Little Big Horn. At the time of its occurrence, June 25, 1876, "Custer\'s Last Stand" was as big an incident and shock to the Americans\' National Psyche as were the Japanese Attack on Pearl Harbor (December 7, 1941) or the Atrocities perpetrated by the Islamic Fascists to New York\'s Twin Trade Towers and the United States\' Armed Forces\' Headquarters in the Pentagon, Arlington, Virginia on September 11, 2002.<br /><br />JUST as so many films of that period of WORLD WAR II (and the years immediately before), there were so many incidents in it that were, if not intentionally done, were demonstrations of virtues that would be needed in time of another Global Conflict, such as we were in by the time of THEY DIED WITH THEIR BOOTS ON was finishing up its original Theatrical release period.<br /><br />POODLE SCHNITZ!!'
```
让我们看看这个示例的模型输出以及实际的类<br>
`target_predicted[index]`<br>
**output**:0.11293286<br>
`class_actual[index]`<br>
**output**:1.0<br>
从这个结果我们可以看出，预测出的情感值与实际的情感值是不同的。<br>
现在，让我们根据一组新的数据样本测试我们训练了的模型，并查看其结果，这里共有8个样本：<br>
```
test_sample_1 = "This movie is fantastic! I really like it because it is so good!"
test_sample_2 = "Good movie!"
test_sample_3 = "Maybe I like this movie"
test_sample_4 = "Meh ..."
test_sample_5 = "If I were a drunk teenager then this movie "
test_sample_6 = "Bad movie"
test_sample_7 = "Not a good movie"
test_sample_8 = "This movie really sucks! Can I get my money back please？"
test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5, test_sample_6, test_sample_7, test_sample_8]
```
现在，让我们将它们转换为整数tokens，并对这些数字序列截长补短<br>
```
test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen = max_num_tokens, padding = seq_pad, truncating = seq_pad)
```
然后填充它们：<br>
`test_samples_tokens_pad.shape`<br>
**output**:(8,544)<br>
最后，针对它们运行模型，得到如下结果：<br>
`rnn_type_model.predict(test_samples_tokens_pad)`<br>
```
array([[0.96684575],
       [0.96288127],
       [0.9389076 ],
       [0.96079415],
       [0.927937  ],
       [0.88653195],
       [0.95413595],
       [0.8341638 ]], dtype=float32)
```     
接近0的值表示消极情感，接近1的值表示积极情感。<br>
## 总结
&emsp;&emsp;本章中，我们覆盖了一个有趣的应用，即情感分析。情感分析被不同的公司用来跟踪顾客对他们的产品的满意程度。甚至政府也使用情绪分析解决方案来跟踪公民对他们未来想要做的事情的满意程度。<br>
&emsp;&emsp;接下来，我们将重点关注一些可以用于半监督和非监督应用的高级深度学习体系结构。<br>
### 组员信息：
学号|姓名|专业
-|-|-
201802210505|蒋小丽|应用统计
201802210506|祖  娇|应用统计
<br>
