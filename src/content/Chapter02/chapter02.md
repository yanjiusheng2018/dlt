# Chapter2  数据建模实例——泰坦尼克号
&emsp;&emsp;线性模型是数据科学领域算法学习的基础。理解线性模型的工作原理对于数据科学的学习过程至关重要，因为它是大多数复杂学习算法(包括神经网络)的基本组成部分。<br>
&emsp;&emsp;在这一章中，我们将深入研究数据科学领域的一个著名问题，泰坦尼克号的例子。学习这个例子的目的是介绍用于分类的线性模型，并了解一个完整的机器学习系统的途径，我们将从数据处理和探索到模型评估开始学习。本章将涵盖以下主题:<br>
&emsp;&emsp;●线性回归模型<br>
&emsp;&emsp;●线性分类模型<br>
&emsp;&emsp;●泰坦尼克号模型的建立和培训<br>
&emsp;&emsp;●不同类型的误差<br>
## 2.1 线性回归模型
&emsp;&emsp;线性回归模型是最基本的回归模型，广泛应用于预测型数据的分析。回归模型的总体思想是要验证两件事:<br>
&emsp;&emsp;1.一组解释变量或输入变量能够很好地预测输出变量吗？模型使用的自变量可以解释因变量的变化吗？<br>
&emsp;&emsp;2.因变量的哪些特征特别重要?它们如何影响因变量(用参数的大小和符号表示)?这些回归参数用于解释一个输出变量(因变量)和一个或多个输入特性(自变量)之间的关系。<br>
&emsp;&emsp;回归方程将表示输入变量(自变量)对输出变量(因变量)的影响。这个方程最简单的形式，有一个输入变量和一个输出变量，由这个公式定义y = c + b*x。这里，y=被估计得参数，c=常数，b=回归系数，x=自变量。<br>
&emsp;&emsp;<br>
### 2.1.1 原因
&emsp;&emsp;线性回归模型是许多学习算法的基石，但这并不是它们流行的唯一原因。以下是它们受欢迎的主要原因:<br>
&emsp;&emsp;**1.应用广泛**:线性回归是最古老的回归技术，在很多方面应用非常广泛，例如预测、财务分析等。<br>
&emsp;&emsp;**2.运行速度快**:线性回归算法非常简单，不包括过于费时的数学计算。<br>
&emsp;&emsp;**3.易于使用(不需要太多的调优)**:线性回归非常容易使用，而且大多数情况下，它是机器学习或数据科学课程中要学习的第一种方法，因为为了获得更好的性能，你没有太多的超参数需要调优。<br>
&emsp;&emsp;**4.高度可解释性**:线性回归因其简洁性和易于检查各组预测系数的贡献度而具有高度可解释性;你可以很容易地理解模型性能并为非技术人员解释模型的输出内容。如果一个系数为零，则相关的预测变量没有贡献。如果系数不为零，则可以很容易确定特定预测变量的贡献度。<br>
&emsp;&emsp;**5.许多其他方法的基础**:线性回归被认为是许多学习方法的基础，如神经网络和它的成长部分，深度学习。<br>
### 2.1.2 以金融为例
&emsp;&emsp;为了更好地理解线性回归模型，我们将以一个广告为例来说明。我们将尝试预测一些公司的销售额，考虑到这些公司在电视、广播和报纸广告上花费的金钱。<br>
### 依赖关系
&emsp;&emsp;为了给我们的广告数据样本建立线性回归模型，我们将使用Stats models库来为线性模型获得良好的特性，但接着，我们将使用scikit-learn，它对数据科学有非常有用的功能。<br>
### 使用pandas导入数据
&emsp;&emsp;Python中有很多库，你可以使用它们来读取、转换或写入数据。其中一个库是panda (http://pandas.pydata.org/)。panda是一个开源库，具有用于数据分析的强大功能和工具，以及非常容易使用的数据结构。<br>
&emsp;&emsp;你可以很容易地以许多不同的方式安装pandas。安装pandas的最好方法是通过conda（http://pandas.org/pandas-docs/stable/install.html#installing-pandas-with-anaconda）<br>
&emsp;&emsp;“bconda是一个开源的包管理系统和环境管理系统，用于安装多个版本的软件包及其相应的数据分析工具并在它们之间轻松切换。它可以在Linux、OS X和Windows上运行，是为Python程序创建的，但可以打包和分发任何软件。--------conda网站。”<br>
&emsp;&emsp;你可以通过安装Anaconda轻松地获得conda，这是一个开放的数据科学平台。<br>
&emsp;&emsp;所以，让我们来看看如何使用熊猫来阅读广告数据样本。首先，我们需要导入pandas：<br>
&emsp;&emsp;所以，让我们来看看如何使用熊猫来阅读广告数据样本。首先，我们需要导入pandas：<br>
&emsp;&emsp;接下来，我们可以使用pandas.read_csv方法将数据加载到一个易于使用的pandas数据结构中，称为DataFrame。有关pandas.read_csv及其参数的更多信息，请参阅此方法的pandas文档（http://pandas.pydata.org/pandas-docs/stable/qenerated/pandas.read_csv.html）：<br>
&emsp;&emsp;#read advertising data samples into a DataFrame<br>
&emsp;&emsp;Advertising_data = pd.read_csv(‘http://www-bcf.use.edu/~qareth/ISL/Advertising.csv’,index_col=0)<br>
&emsp;&emsp;传递给pandas.read_csv方法的第一个参数是一个表示文件路径的字符串值。字符串可以是包含IUUQ、GUQ、T和GJMF的URL。传递的第二个参数是列的索引，它将用作数据行的标签/名称。<br>
&emsp;&emsp;现在，我们有了DataFrame数据，它包含URL中提供的广告数据，每一行都由第一列标记。如前所述，panda提供了易于使用的数据结构，你可以将其用作数据的容器。这些数据结构有一些与之相关联的方法，你将使用这些方法来转换和/或操作数据。<br>
&emsp;&emsp;现在，让我们看一下广告数据的前五行:<br>
&emsp;&emsp;#DataFrame.head methond above the first n row of the data where the<br>
&emsp;&emsp;#default value of n is 5,DataFrame.head(n=5)<br>
&emsp;&emsp;Advertising_data.head()<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/1.png)
### 了解广告数据
&emsp;&emsp;这个问题属于监督学习类型，其中我们有解释特征(输入变量)和响应(输出变量)。输入/输入变量分别是什么?<br>
&emsp;&emsp;1.电视:在一个特定的市场上为一个产品在电视上做广告的花费(以几千美元计)<br>
&emsp;&emsp;2.收音机:花在收音机上的广告费用<br>
&emsp;&emsp;3.报纸:花在报纸上的广告费用<br>
&emsp;&emsp;响应/结果/输出变量是什么?<br>
&emsp;&emsp;销售:在特定市场上的单个产品的销量(以数以千计的小部件为单位)<br>
&emsp;&emsp;我们还可以使用DataFrame来知道我们数据中的样本/观察数:<br>
&emsp;&emsp;#print the shape of the DataFrame<br>
&emsp;&emsp;Advertising_data.shape<br>
&emsp;&emsp;Output:<br>
&emsp;&emsp;(200,4)<br>
&emsp;&emsp;所以，广告数据中有200个观察结果。<br>
### 数据分析与可视化
&emsp;&emsp;为了理解数据的基本形式、输入变量和输出变量之间的关系以及更多内在联系，我们可以使用不同类型的可视化。为了理解广告数据输入变量和输出变量之间的关系，我们将使用散点图。<br>
&emsp;&emsp;为了对数据进行不同类型的可视化，可以使用Matplotlib(http://matplotlib.org/)，这是一个用于可视化的Python 2D库。要获得Matplotlib，你可以按照他们的安装说明:http://matplotlib.org/users/installing.html.<br>
&emsp;&emsp;让我们导入可视化库Matplotlib:<br>
&emsp;&emsp;Import matplotlib.pyplot as plt<br>
&emsp;&emsp;#The next line will allow us make inlines plots that could appear<br>
&emsp;&emsp;directly in the notebook<br>
&emsp;&emsp;#without poping up in a different window<br>
&emsp;&emsp;Matplotlib inline<br>
&emsp;&emsp;现在，我们使用散点图来可视化广告数据特性和响应变量之间的关系:<br>
&emsp;&emsp;fig, axs=plt.subplots{1,3,sharey=True}<br>
&emsp;&emsp;#Adding the scatterplots to the grid<br>
&emsp;&emsp;Advertising_data.plot(kind=’scatter’,x=’TV’,y=’sales’,ax=axs[0],figsize=(16,8))<br>
&emsp;&emsp;Advertising_datd.plot(kind=’scatters’,x=’radio’,y=’sales’,ax=axs[1])<br>
&emsp;&emsp;Advertising_datd.plot(kind=’scatters’,x=’radio’,y=’sales’,ax=axs[1])<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>



