# 第11章 表示学习---实现字嵌入

&emsp;&emsp;机器学习是一门以统计学和线性代数为基础的科学。由于反向传播的架构,在大多数机器学习或深度学习中，应用矩阵运算是很常见的。这就是为什么深度学习，或者一般的机器学习，只接受实值量作为输入的主要原因。这个事实与机器翻译、情绪分析等许多应用相矛盾;它们有文本作为输入。因此，为了在这个应用程序中使用深度学习，我们需要以深度学习接受的形式使用它!

&emsp;&emsp;在这一章中，我们将介绍表示学习的领域，即表示学习
&emsp;&emsp;从文本中学习实值表示的方法，同时保留实际文本的语义。例如， 代表爱的词应该非常接近代表爱慕的词，因为它们用在非常相似的上下文中。因此，本章将讨论以下主题:

## 表征学习概论

&emsp;&emsp;到目前为止，我们使用的所有机器学习算法或体系结构都需要输入为实值或实值量的矩阵，这是机器学习中的一个常见主题。例如，在卷积神经网络中，我们需要原始输入作为模型输入的图像的像素值。在这一部分中，我们处理的是文本，因此我们需要以某种方式对文本进行编码并产生实值量，这些实值量可以输入到机器学习算法中。为了将输入文本编码为实值量，我们需要使用一门叫做自然语言处理(NLP)的中级科学。

&emsp;&emsp;我们提到过，在这种管道中，我们将文本输入到机器学习模型中，比如情绪分析，这将是有问题的，也不会起作用，因为我们不会能够应用反向传播或任何其他操作，如在输入上的点积，这是一个字符串。因此，我们需要使用一个NLP机制，使我们能够建立一个文本的中间表示形式，可以携带与文本相同的信息，也可以输入到机器学习模型中。

&emsp;&emsp;我们需要将输入文本中的每个单词或令牌转换为实值向量。这些如果向量不携带原始输入的模式、信息、含义和语义，那么它们将是无用的。例如，在真实的文本中，爱和爱慕这两个词是非常重要的彼此相似，意思相同。我们需要表示它们的实值向量的结果它们彼此接近并且在同一个向量空间中。所以,这两个词的向量表示以及另一个与它们不相似的词将会是这样的图:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204124918413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

表示学习-实现词嵌入 					
&emsp;&emsp;有许多技术可以用于这项任务。这种技术被称为嵌入，将文本嵌入到另一个实值向量空间中。
&emsp;&emsp;稍后我们会看到，这个向量空间实际上非常有趣，因为你会发现你可以把一个单词的向量从其他与它相似的单词中提取出来，或者在这个空间中做一些地理上的处理。

## Word2Vec
&emsp;&emsp;Word2Vec是NLP领域应用最广泛的嵌入式技术之一。该模型通过查看上下文信息，从输入文本中创建实值向量输入字出现。因此，你会发现相似的词会在非常相似的语境中被提及，因此模型会知道这两个词应该一起放在特定的嵌入空间中彼此接近。
&emsp;&emsp;从下图的叙述中，模型将学习到爱慕和爱这两个词在非常相似的语境中，应该放在非常接近的位置生成的向量空间。“喜欢”这个词的语境可能和“爱”这个词有点相似，但不会像单词爱慕那样接近“爱”:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204125217577.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

&emsp;&emsp;Word2Vec模型也依赖于输入句子的语义特征;例如，单词adore和love主要用于积极的语境，通常放在名词短语或名词之前。同样，模型会知道这两个词有某些含义。
&emsp;&emsp;它更有可能把这两个向量的向量表示放在相似的上下文中。所以，句子的结构会告诉单词2vec模型很多类似的单词。

&emsp;&emsp;在实践中，人们向Word2Vec模型输入大量的文本。该模型将学习如何为相似的词生成相似的向量，并且它将为在输入文本中的每个唯一的词。所有这些词的向量将被组合，最终的输出将是一个嵌入矩阵，其中每行表示特定唯一单词的实值向量表示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204125413627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

&emsp;&emsp;因此，模型的最终输出将是训练语料库中所有唯一单词的嵌入矩阵。通常，好的嵌入矩阵可能包含数百万个实数值向量。
&emsp;&emsp;Word2Vec建模使用窗口扫描句子，然后我们可以根据上下文信息预测窗口中间词的向量的Word2Vec模型每次扫描一个句子。与任何机器学习技术类似，我们需要为Word2Vec模型定义一个成本函数及其相应的优化标准，这将使模型能够为每个唯一的图像生成实值向量，并根据其上下文信息将向量相互关联。


## 构建Word2Vec模型
&emsp;&emsp;在本节中，我们将详细介绍如何构建一个Word2Vec模型。正如我们前面提到的，我们的最终目标是拥有一个经过训练的模型，该模型能够为输入文本数据生成实值向量表示，也称为单词嵌入。

&emsp;&emsp;在模型的训练过程中，我们将使用极大似然法，在已知模型之前看到的称之为h的单词的输入句子中，来最大化下一个单词w的概率。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2018120412583857.png)

&emsp;&emsp;在这里，相对于上下文h的兼容性，score函数计算一个值来表示目标单词w。这个模型讲用来训练以最大化训练输入数据的可能性(log likelihood用于数学简化和使用log进行推导):

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204125928491.png)

&emsp;&emsp;所以，ML法会试图最大化上面的方程，得到a概率语言模型。但是这个的计算是非常昂贵的，因为我们需要使用分数函数来计算每个概率词汇w’，在该模型的相应当前上下文h中。这将发生在每一个训练步骤。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204130041455.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

&emsp;&emsp;由于构建概率语言模型的计算成本很高，人们倾向于使用不同的计算成本较低的技术，例如连续字袋(CBOW)和跳格模型。
&emsp;&emsp;通过训练这些模型，建立了一种逻辑回归的二元分类方法，将实际目标词wt和h噪声或想象词分离开来上下文。下面的图表使用CBOW技术简化了这个想法:这种最大似然方法将用softmax函数表示:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204130206630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

&emsp;&emsp;下一个图显示了构建Word2Vec模型时可以使用的两种架构:



![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204130232489.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

&emsp;&emsp;说的更正式些，这些技术的目标函数最大化了以下几点:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204131031190.png)

&emsp;&emsp;在这里:


![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204131109268.png)

二元逻辑回归的概率是基于是在数据集D中看到上下文h中的单词w的模型，数据集D是根据向量计算的。这个向量表示已学习的嵌入，是我们可以从一个有噪声的概率分布(如训练输入示例的单位图)中生成的虚构或有噪声的词。

&emsp;&emsp;综上所述，这些模型的目的是区分真实和虚构输入，因此给真实的单词分配更高的概率，而给虚构的或有噪声的单词分配更少的概率(https:IIpapers.nips.ccIpaperI5021- distributed-representations-of-吐ords-and-phrases-and-their- compositionality.pdf)。
&emsp;&emsp;当模型将高概率分配给实单词，低概率分配给噪声单词时，这一目标最大化。
 &emsp;&emsp;从技术上讲，将高概率赋给真实单词的过程称为负抽样，并有良好的数学动机来使用这个损失函数:它提出在极限下近似softmax函数的更新。但是在计算上，它是特别的有吸引力，因为计算损失函数现在只根据我们选择的噪声词的数量(k)来缩放，而不是所有的词汇(V)。这使得训练变得更快。我们将准确使用非常相似的噪声对比估计(NCE) 损耗(https:II papers.nips.ccIpaperI5165-learning-word-embeddings-efficiently-with-noise-contrastiveestimation.pdf)，而这里 tensorflow有一个方便的助手函数。

## 这是skip-gram体系结构的一个实际示例
&emsp;&emsp;让我们通过一个实际的例子，看看skip-gram模型将如何在这种情况下工作:
the quick brown fox jumped over the lazy dog
&emsp;&emsp;首先，我们需要建立一个词及其上下文的数据集。定义上下文取决于我们，但它必须有意义。因此，我们将在目标单词周围设置一个窗口，从右边取一个单词，从左边取一个单词。
([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox)
通过这种语境技术，我们最终会得到以下一组词及其对应的语境:生成的单词及其对应的上下文将被表示为对。跳格模型的思想与CBOW模型相反。在跳格模型，我们将尝试预测单词的上下文基于它的目标词。例如，考虑到第一对，skip-gram模型将尝试预测和从目标词，等等。因此，我们可以重写我们的数据集如下:
&emsp;&emsp;现在，我们有了一组输入和输出对。
&emsp;&emsp;我们尝试模拟特定步骤t的训练过程，因此，skip-gram模型将取第一个训练样本，其中输入为单词，目标输出为词。接下来，我们还需要构造噪声输入，所以我们要随机地从输入数据的单位图中选择。为了简单起见，噪声向量的大小将只有一个。例如，我们可以选择这个词作为一个嘈杂的例子。

&emsp;&emsp;现在，我们可以继续计算实对和噪声对之间的损失为:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204132141214.png)

&emsp;&emsp;在这种情况下，目标是更新参数以改进前面的目标函数。通常，我们可以使用梯度。因此，我们将尝试计算相对于目标函数参数损失的梯度，它将被表示为

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204132232378.png)
 
 &emsp;&emsp;经过训练后，我们可以根据实值向量表示的降维结果对其进行形象化处理。你会发现这个向量空间非常有趣是因为你可以用它做很多有趣的事情。例如，你可以在这个空间里通过说国王是皇后就像男人是女人。我们甚至可以通过从皇后向量中减去国王向量并加上男人来得到女人向量;这样做的结果将非常接近女性的实际学习向量。你也可以在这个空间里学习地理。
&emsp;&emsp;前面的例子给出了这些向量背后的很好的直觉，以及它们如何对大多数NLP应用程序(如机器翻译或词性)有用(POS)标记。

## Skip-gram Word2Vec实现
&emsp;&emsp;在理解了skip-gram模型如何工作的数学细节之后，我们将实现skip-gram，它将单词编码为具有一定值的实值向量属性(因此得名Word2Vec)。通过实现这个体系结构，您将了解学习另一种表示的过程是如何工作的。
&emsp;&emsp;文本是许多自然语言处理应用程序(如机器翻译、情感分析和文本到语音系统)的主要输入。所以，学习一个实值文本表示将帮助我们使用不同的深度学习技术来完成这些任务。
在这本书的前几章中，我们介绍了一种叫做one-hot编码的东西，它产生了一个0的向量，除了这个向量表示的单词的索引。所以,您可能想知道我们为什么不在这里使用它。这种方法效率很低，因为通常你会有一大堆不同的单词，可能大概有50000个单词，使用一热编码就会产生一个49,999个向量集合为0，只有一个条目设置为1。这样一个非常稀疏的输入将会导致巨大的计算浪费，因为我们会在神经网络的隐层中做矩阵乘法。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204132647226.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

&emsp;&emsp;正如我们前面提到的，使用one-hot编码的结果将是非常稀疏的向量，尤其是当你有大量不同的单词需要编码的时候。
&emsp;&emsp;下图显示，当我们将除一项以外的所有0的稀疏向量乘以一个权重矩阵时，输出将仅是该矩阵的一行对应于稀疏向量的一个值:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204132710607.png)

&emsp;&emsp;为了避免这种巨大的计算浪费，我们将使用嵌入，这只是一个完全连接层与一些嵌入权重。在这一层，我们跳过了这个低效的乘法，从一个叫做权值矩阵的东西中查找嵌入层的嵌入权值。
&emsp;&emsp;因此，我们将使用这个权重查找这个权重矩阵来查找嵌入的权重，而不是计算产生的浪费。首先，需要构建此查找获取。为此，我们将把所有输入单词编码为整数，如下图所示，然后为了得到这个单词的相应值，我们将使用它的整数表示作为这个权重矩阵中的行数的过程。
&emsp;&emsp;查找特定单词的对应嵌入值称为嵌入查找。如前所述，嵌入层将只是一个完全连接的层，其中单位数表示嵌入维数。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204132736834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3MDkzODMx,size_16,color_FFFFFF,t_70)

&emsp;&emsp;你可以看到这个过程是非常直观和直接的;我们只需要遵循以下步骤:
1、将查找表定义为权重矩阵。
2、将嵌入层定义为具有特定的完全连接的隐藏层。单位数量(嵌入维数)
3、使用权重矩阵查找作为计算上的替代方法。不必要的矩阵乘法
4、最后，将查找表训练为任意权重矩阵。

&emsp;&emsp;如前所述，我们将在此构建一个skip-gram Word2Vec模型，这是一种学习词汇表示的有效方法，同时保留了词汇所具有的语义信息。
&emsp;&emsp;因此，让我们继续使用skip-gram体系结构构建一个Word2Vec模型，该体系结构被证明优于其他体系结构。

## 数据分析和预处理
&emsp;&emsp;在本节中，我们将定义一些帮助函数，使我们能够构建一个良好的Word2Vec模型。对于这个实现，我们将使用一个清理过的版本维基百科(http:IImattmahoney.netIdcItextdata.html).


### 因此，让我们从导入这个实现所需的包开始:
#### 导入此实现所需的包
```javascript
import numpy as np   #导入NumPy函数库且别名np
import tensorflow as tf  #导入tensorflow且别名tf
import matplotlib.pyplot as plt  #导入绘图工具并别名plt
from sklearn.manifold import TSNE #从sklearn.manifoldz中导入TSNE算法 
```
#### 用于下载数据集的包
```javascript
from urllib.request import urlretrieve  #导入urlretrieve函数
from os.path import isfile, isdir #从Python的os.path模块中导入了 isdir() 和 isfile()函数
from tqdm import tqdm #使用tqdm显示进度条
import zipfile #导入zipfile模块 -单独压缩/解压文件
```
#### 用于数据预处理的包
```javascript
import re #导入re模块，方便直接调用来实现正则匹配
from collections import Counter #从collections模块导入counter工具，用于支持便捷和快速地计数
import random #导入random模块
```
 
 ### 接下来，我们将定义一个类，如果之前没有下载数据集，它将用于下载数据集:
```javascript
#### 在这个实现中，我们将使用来自MA的维基百科的清理版本。
#### 因此，我们将定义一个帮助下载数据集的助手类。
wiki_dataset_folder_path = 'wikipedia_data' #维基数据集的文件夹路径
wiki_dataset_filename = 'text8.zip' #维基数据集的文件名
wiki_dataset_name = 'Text8 Dataset'  #维基数据集的名称

class DLProgress(tqdm):
    last_block = 0 
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num   

if not isfile(wiki_dataset_filename): 
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=wiki_dataset_name) as pbar:
        urlretrieve(
            'http://mattmahoney.net/dc/text8.zip',
            wiki_dataset_filename,
            pbar.hook) 
```
#### 现在检查数据是否已被提取
```javascript
if not isdir(wiki_dataset_folder_path):
    with zipfile.ZipFile(wiki_dataset_filename) as zip_ref:
        zip_ref.extractall(wiki_dataset_folder_path)
        
with open('wikipedia_data/text8') as f:
    cleaned_wikipedia_text = f.read() 
```
&emsp;&emsp;我们可以看看这个数据集的前100个字符: 接下来，我们将对文本进行预处理，因此我们将定义一个助手函数 这将帮助我们将诸如标点符号之类的特殊字符替换为已知令牌。 另外，为了减少输入文本中的噪音，您可能需要删除文本中不经常出现的单词:
```javascript
def preprocess_text(input_text):
    input_text = input_text.lower()
    input_text = input_text.replace('.', ' <PERIOD> ')
    input_text = input_text.replace(',', ' <COMMA> ')
    input_text = input_text.replace('"', ' <QUOTATION_MARK> ')
    input_text = input_text.replace(';', ' <SEMICOLON> ')
    input_text = input_text.replace('!', ' <EXCLAMATION_MARK> ')
    input_text = input_text.replace('?', ' <QUESTION_MARK> ')
    input_text = input_text.replace('(', ' <LEFT_PAREN> ')
    input_text = input_text.replace(')', ' <RIGHT_PAREN> ')
    input_text = input_text.replace('--', ' <HYPHENS> ')
    input_text = input_text.replace('?', ' <QUESTION_MARK> ')
    
    input_text = input_text.replace(':', ' <COLON> ')
    text_words = input_text.split()
     # 忽略所有出现五个词的单词
    text_word_counts = Counter(text_words)
    trimmed_words = [word for word in text_words if text_word_counts[word] > 5]
    return trimmed_words
```
 #### 现在，让我们在输入文本上调用这个函数并查看输出: 让我们看看我们有多少词和不同的词，为预处理版本的文本:
```javascript
print('Preprocessing the text')  #打印（预处理文本…）
preprocessed_words = preprocess_text(cleaned_wikipedia_text)
print("Total number of words in the text: {}".format(len(preprocessed_words)))#打印文本中的单词总数
print("Total number of unique words in the text: {}".format(len(set(preprocessed_words))))#打印文本中唯一单词的总数
```
&emsp;&emsp;这里，我在创建字典来将单词转换成整数，整数按降序分配，因此最常用的单词(the) 是整数，其次是最常用的gets，依此类推。这些单词被转换为整数并存储在列表中。
&emsp;&emsp;正如在本节前面提到的，我们需要使用单词的整数索引在权重矩阵中查找它们的值，因此我们将使用单词to integers和integers单词。这将帮助我们查找单词，也得到具体的单词索引。例如，输入文本中重复次数最多的单词将在位置0处建立索引，其次是重复次数第二多的单词，依此类推。
&emsp;&emsp;那么，让我们定义一个函数来创建这个查找表:
```javascript
def create_lookuptables(input_words):#定义函数创建查找表
    """
    Creating lookup tables for vocan
    Function arguments:
    param words: Input list of words
    """
    input_word_counts = Counter(input_words)
    sorted_vocab = sorted(input_word_counts, key=input_word_counts.get, reverse=True)
    integer_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_integer = {word: ii for ii, word in integer_to_vocab.items()}
    #返回一组二进制文件
    return vocab_to_integer, integer_to_vocab
    现在，让我们调用已定义的函数来创建查找表:
    print('Creating the lookup table...')
vocab_to_integer, integer_to_vocab = create_lookuptables(preprocessed_words)
integer_words = [vocab_to_integer[word] for word in preprocessed_words]
```
&emsp;&emsp;为了建立一个更精确的模型，我们可以删除不太改变上下文的单词，如of, for, the等等。因此，实践证明我们可以建立更精确的 模型的同时丢弃这些类型的词。从上下文中删除与上下文无关的单词的过程称为子抽样。为了定义单词丢弃的通用机制，Mikolov引入了一个计算某个单词丢弃概率的函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181204180116243.png)

&emsp;&emsp; t是单词丢弃参数
  &emsp;&emsp;f(wi)是输入数据集中特定目标字wi的频率。

&emsp;&emsp;因此，我们将实现一个助手函数，它将计算数据集中每个单词的丢弃概率p(wi):
```javascript
print('Subsampling...')#打印（“子采样…”）
#去除无益词汇
word_threshold = 1e-5
word_counts = Counter(integer_words) #将整数字计总并赋值单词数计数
total_number_words = len(integer_words)#将整数词的长度赋值总计单词的数目
#计算单词的频率
frequencies = {word: count/total_number_words for word, count in word_counts.items()}

#计算丢弃概率的总数
prob_drop = {word: 1 - np.sqrt(word_threshold/frequencies[word]) for word in word_counts}
training_words = [word for word in integer_words if random.random() < (1 - prob_drop[word])]
```
&emsp;&emsp;与其平等对待所有上下文相关的单词，我们将为那些离目标单词有点远的单词分配更少的权重。例如，如果我们选择窗口为C = 4，然后我们将从1到C的范围中选择一个随机数L，然后从当前单词的历史和未来中抽取L单词样本。现在我们来定义这个函数

#定义在特定窗口中返回特定索引周围的单词的函数
```javascript
def get_target(input_words, ind, context_window_size=5):

    # 从当前单词的历史和特征中选择用于生成单词的随机数
    rnd_num = np.random.randint(1, context_window_size + 1)
    start_ind = ind - rnd_num if (ind - rnd_num) > 0 else 0
    stop_ind = ind + rnd_num
    target_words = set(input_words[start_ind:ind] + input_words[ind + 1:stop_ind + 1])
    return list(target_words)
```
&emsp;&emsp;另外，让我们定义一个生成器函数，从训练样本中生成一个随机的批处理，并得到该批处理中每个单词的上下文单词:

#定义一个函数来生成词批作为元组（输入，目标）
```javascript
def generate_random_batches(input_words, train_batch_size, context_window_size=5):
    num_batches = len(input_words) // train_batch_size
    
#处理全部批次
    input_words = input_words[:num_batches * train_batch_size]
    for ind in range(0, len(input_words), train_batch_size):
        input_vals, target = [], []
        input_batch = input_words[ind:ind + train_batch_size]
        
        # 获取每个单词的上下文
        for ii in range(len(input_batch)):
            batch_input_vals = input_batch[ii]
            batch_target = get_target(input_batch, ii, context_window_size)
            target.extend(batch_target)
            input_vals.extend([batch_input_vals] * len(batch_target))
        yield input_vals, target
```
## 建筑模型
&emsp;&emsp;接下来，我们将使用以下结构构建计算图:

&emsp;&emsp;我们将使用一个嵌入层来学习这些单词的一个特殊实值表示。因此，这些词将以one-hot 向量的形式出现。其思想是训练这个网络来建立权重矩阵。那么，让我们从创建模型的输入开始: 
```javascript
train_graph = tf.Graph()
```
&emsp;&emsp;另外，我们不需要自己实现查找函数，因为它已经实现了 在Tensorflow可用: tf.nn.embedding lookup()。因此，它将使用整数编码，并在权重矩阵中找到相应的行。权重矩阵将从均匀分布中随机初始化:
```javascript
with train_graph.as_default():
    inputs_values = tf.placeholder(tf.int32, [None], name='inputs_values')
    labels_values = tf.placeholder(tf.int32, [None, None], name='labels_values')

num_vocab = len(integer_to_vocab)
num_embedding = 300
with train_graph.as_default():
    embedding_layer = tf.Variable(tf.random_uniform((num_vocab, num_embedding), -1, 1))
    # 接下来，我们将使用Tf.n.EngtudioLoopUp函数获取隐藏层的输出。
    embed_tensors = tf.nn.embedding_lookup(embedding_layer, inputs_values)
num_sampled = 100

with train_graph.as_default():
```
```javascript
    # 创建SoftMax权值和偏差
    softmax_weights = tf.Variable(tf.truncated_normal((num_vocab, num_embedding)))
    softmax_biases = tf.Variable(tf.zeros(num_vocab), name="softmax_bias")
    
    # 用负抽样法计算模型损失
    model_loss = tf.nn.sampled_softmax_loss(
        weights=softmax_weights,biases=softmax_biases，
        labels=labels_values,
        inputs=embed_tensors,
        num_sampled=num_sampled,
        num_classse=num_vocab)
 
model_cost = tf.reduce_mean(model_loss)
model_optimizer = tf.train.AdamOptimizer().minimize(model_cost)
```
&emsp;&emsp;为了验证我们的训练模型，我们将取样一些常见的或常见的词和一些不常见的词，并尝试打印我们的最接近的词集基于学习了skip-gram体系结构的表示:
```javascript
with train_graph.as_default():

    # 相似性评价中的随机词集
    valid_num_words = 16 
    valid_window = 100  
    
    #从（0100）和（10001100）每个范围选取8个样本。低ID意味着更频繁
    valid_samples = np.array(random.sample(range(valid_window), valid_num_words // 2))
    valid_samples = np.append(valid_samples,
                              random.sample(range(1000, 1000 + valid_window), valid_num_words // 2))
    valid_dataset_samples = tf.constant(valid_samples, dtype=tf.int32)
    
    # 余弦距离的计算
norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_layer), 1, keep_dims=True))
    normalized_embed = embedding_layer / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embed, valid_dataset_samples)
    cosine_similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embed))
&emsp;&emsp;现在，我们已经为我们的模型准备好了所有的细节，我们准备开始训练过程。
```
 
 ## 训练
 让我们开始培训过程：
```javascript
num_epochs = 10 
train_batch_size = 1000 
contextual_window_size = 10 

with train_graph.as_default():
    saver = tf.train.Saver()
    
with tf.Session(graph=train_graph) as sess:
    #创建一个新的tensorflow的session，并指定了训练计算图
    iteration_num = 1 
    average_loss = 0 
    
    sess.run(tf.global_variables_initializer())
    for e in range(1, num_epochs + 1):
    
        # 生成随机批量训练
        batches = generate_random_batches(training_words, train_batch_size, contextual_window_size)
        
        # 迭代批处理样本
        for input_vals, target in batches:
        
            # 创建使用feed_dict将数据投入到训练损失中
            feed_dict = {inputs_values: input_vals,
                         labels_values: np.array(target)[:, None]}

            train_loss, _ = sess.run([model_cost, model_optimizer], feed_dict=feed_dict)

            # 定义后平均损失=前平均损失+训练损失
            average_loss += train_loss

            # 100次迭代后打印结果
            if iteration_num % 100 == 0: #这里的%取余数
                print("Epoch Number {}/{}".format(e, num_epochs),
                      "Iteration Number: {}".format(iteration_num),
                      "Avg. Training loss: {:.4f}".format(average_loss / 100))
                average_loss = 0

            if iteration_num % 1000 == 0:

                # 用余弦相似度求单词的最近单词
                similarity = cosine_similarity.eval()
                for i in range(valid_num_words):
                    valid_word = integer_to_vocab[valid_samples[i]]

                    # 最近邻居数
                    top_k = 8
                    nearest_words = (-similarity[i, :]).argsort()[1:top_k + 1]
                    msg = 'The nearest to %s:' % valid_word
                    for k in range(top_k):
                        similar_word = integer_to_vocab[nearest_words[k]]
                        msg = '%s %s,' % (msg, similar_word)
                    print(msg)

            iteration_num += 1 
    save_path = saver.save(sess, "checkpoints/cleaned_wikipedia_version.ckpt")
    embed_mat = sess.run(normalized_embed)
    #运行上述代码片段10个epoch后，您将得到以下输出:
```
&emsp;&emsp;我们可以很明显的看到，随着迭代次数的不断增加，训练损失值在不断的下降，可以看到我们的训练的模型对名词、动词、形容词等类型的单词的相似词汇的识别是非常精确的。因此由skip-gram word2vec得到的向量空间表达是非常高质量的，近义词在向量空间上的位置也是非常接近的。为了帮助我们更清楚地了解嵌入矩阵，我们将使用维数缩减技术，如TSNE，来降低实值向量到二维，然后我们把它们形象化用对应的词标记每个点:
```javascript
%matplotlib inline #内嵌画图
with train_graph.as_default():
    saver = tf.train.Saver() #首先定义一个Saver类
#定义会话
with tf.Session(graph=train_graph) as sess:

    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embedding_matrix = sess.run(embedding_layer)#定义嵌入矩阵
num_visualize_words = 500 
tsne_obj = TSNE()
embedding_tsne=
tsne_obj.fit_transform(embedding_matrix[:num_visualize_words, :])
fig, ax = plt.subplots(figsize=(14, 14))
for ind in range(num_visualize_words):
    plt.scatter(*embedding_tsne[ind, :], color='steelblue')
    plt.annotate(integer_to_vocab[ind], (embedding_tsne[ind, 0], embedding_tsne[ind, 1]), alpha=0.7)
```
## 总结
&emsp;&emsp;在这一章中，我们讨论了表征学习的概念以及为什么它对没有实值的输入的形式的深度学习或机器学习很有用。此外，我们还介绍了将单词转换为实值向量Word2Vec所采用的一种技术，这种技术具有非常有趣的特性。最后，我们实现了使用skip-gram体系结构的Word2Vec模型。

&emsp;&emsp;接下来，您将在一个情感分析示例中看到这些学习到的表示的实际用法，我们需要将输入文本转换为实值向量。

学号|姓名|专业
-|-|-
201802210511|张德鑫|应用统计
201802210514|林华锐|应用统计
201802210515|李根浩|应用统计
<br>
