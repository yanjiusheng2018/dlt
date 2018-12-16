# Chapter2  数据建模实例——泰坦尼克号
&emsp;&emsp;线性模型是数据科学领域算法学习的基础。理解线性模型的工作原理对于数据科学的学习过程至关重要，因为它是大多数复杂学习算法(包括神经网络)的基本组成部分。

&emsp;&emsp;在这一章中，我们将深入研究数据科学领域的一个著名问题，泰坦尼克号的例子。学习这个例子的目的是介绍用于分类的线性模型，并了解一个完整的机器学习系统的途径，我们将从数据处理和探索到模型评估开始学习。本章将涵盖以下主题:

&emsp;&emsp;●线性回归模型<br>
&emsp;&emsp;●线性分类模型<br>
&emsp;&emsp;●泰坦尼克号模型的建立和培训<br>
&emsp;&emsp;●不同类型的误差<br>

## 2.1 线性回归模型
&emsp;&emsp;线性回归模型是最基本的回归模型，广泛应用于预测型数据的分析。回归模型的总体思想是要验证两件事:

&emsp;&emsp;1.一组解释变量或输入变量能够很好地预测输出变量吗？模型使用的自变量可以解释因变量的变化吗？<br>
&emsp;&emsp;2.因变量的哪些特征特别重要?它们如何影响因变量(用参数的大小和符号表示)?这些回归参数用于解释一个输出变量(因变量)和一个或多个输入特性(自变量)之间的关系。

&emsp;&emsp;回归方程将表示输入变量(自变量)对输出变量(因变量)的影响。这个方程最简单的形式，有一个输入变量和一个输出变量，由这个公式定义y = c + b*x。这里，y=被估计得参数，c=常数，b=回归系数，x=自变量。

### 2.1.1 原因——线性回归广泛应用的原因
&emsp;&emsp;线性回归模型是许多学习算法的基石，但这并不是它们流行的唯一原因。以下是它们受欢迎的主要原因:<br>
&emsp;&emsp;**1. 应用广泛**: 线性回归是最古老的回归技术，在很多方面应用非常广泛，例如预测、财务分析等。<br>
&emsp;&emsp;**2. 运行速度快**: 线性回归算法非常简单，不包括过于费时的数学计算。<br>
&emsp;&emsp;**3. 易于使用(不需要太多的调优)**: 线性回归非常容易使用，而且大多数情况下，它是机器学习或数据科学课程中要学习的第一种方法，因为为了获得更好的性能，你没有太多的超参数需要调优。<br>
&emsp;&emsp;**4. 高度可解释性**: 线性回归因其简洁性和易于检查各组预测系数的贡献度而具有高度可解释性;你可以很容易地理解模型性能并为非技术人员解释模型的输出内容。如果一个系数为零，则相关的预测变量没有贡献。如果系数不为零，则可以很容易确定特定预测变量的贡献度。<br>
&emsp;&emsp;**5. 许多其他方法的基础**: 线性回归被认为是许多学习方法的基础，如神经网络和它的成长部分，深度学习。<br>

### 2.1.2 广告类型对公司销售额的影响——一个金融实例
&emsp;&emsp;为了更好地理解线性回归模型，我们将以一个广告为例来说明。我们将尝试预测一些公司的销售额，考虑到这些公司在电视、广播和报纸广告上花费的金钱。

### 依赖关系
&emsp;&emsp;为了给我们的广告数据样本建立线性回归模型，我们将使用Stats models库来为线性模型获得良好的特性，但接着，我们将使用scikit-learn，它对数据科学有非常有用的功能。

### 使用pandas导入数据
&emsp;&emsp;Python中有很多库，你可以使用它们来读取、转换或写入数据。其中一个库是panda <http://pandas.pydata.org/> 。panda是一个开源库，具有用于数据分析的强大功能和工具，以及非常容易使用的数据结构。

&emsp;&emsp;你可以很容易地以许多不同的方式安装pandas。安装pandas的最好方法是通过conda [](http://pandas.org/pandas-docs/stable/install.html#installing-pandas-with-anaconda)。

&emsp;&emsp;“bconda是一个开源的包管理系统和环境管理系统，用于安装多个版本的软件包及其相应的数据分析工具并在它们之间轻松切换。它可以在Linux，OS X和Windows上运行，是为Python程序创建的，但可以打包和分发任何软件。——conda网站。”

&emsp;&emsp;你可以通过安装Anaconda轻松地获得conda，这是一个开放的数据科学平台。

&emsp;&emsp;所以，让我们来看看如何使用pandas来阅读广告数据样本。首先，我们需要导入pandas：<br>

`import pandas as pd`

&emsp;&emsp;接下来，我们可以使用pandas.read_csv方法将数据加载到一个易于使用的pandas数据结构中，称为DataFrame。有关pandas.read_csv及其参数的更多信息，请参阅此方法的pandas文档[](http://pandas.pydata.org/pandas-docs/stable/qenerated/pandas.read_csv.html):

```
#read advertising data samples into a DataFrame

Advertising_data = <br>
pd.read_csv(‘http://www-bcf.use.edu/~qareth/ISL/Advertising.csv’,index_col=0)
```
&emsp;&emsp;传递给pandas.read_csv方法的第一个参数是一个表示文件路径的字符串值。字符串可以是包含IUUQ、GUQ、T和GJMF的URL。传递的第二个参数是列的索引，它将用作数据行的标签/名称。

&emsp;&emsp;现在，我们有了DataFrame数据，它包含URL中提供的广告数据，每一行都由第一列标记。如前所述，panda提供了易于使用的数据结构，你可以将其用作数据的容器。这些数据结构有一些与之相关联的方法，你将使用这些方法来转换和/或操作数据。

&emsp;&emsp;现在，让我们看一下广告数据的前五行:

```
#DataFrame.head methond above the first n row of the data where the
#default value of n is 5,DataFrame.head(n=5)
Advertising_data.head()
```
Output

![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E8%A1%A81.png)

### 广告数据基本信息
&emsp;&emsp;这个问题属于监督学习类型，其中我们有解释特征(输入变量)和响应(输出变量)。输入/输入变量分别是什么?

&emsp;&emsp;●**电视**: 在一个特定的市场上为一个产品在电视上做广告的花费(以几千美元计)<br>
&emsp;&emsp;●**收音机**: 花在收音机上的广告费用<br>
&emsp;&emsp;●**报纸**: 花在报纸上的广告费用

&emsp;&emsp;响应/结果/输出变量是什么?

&emsp;&emsp;●**销售**: 在特定市场上的单个产品的销量(以数以千计的小部件为单位)

&emsp;&emsp;我们还可以使用DataFrame来知道我们数据中的样本/观察数:

```
#print the shape of the DataFrame
&emsp;&emsp;Advertising_data.shape
&emsp;&emsp;Output:
&emsp;&emsp;(200,4)
```
&emsp;&emsp;所以，广告数据中有200个观察结果。

### 数据分析与可视化
&emsp;&emsp;为了理解数据的基本形式、输入变量和输出变量之间的关系以及更多内在联系，我们可以使用不同类型的可视化。为了理解广告数据输入变量和输出变量之间的关系，我们将使用散点图。

&emsp;&emsp;为了对数据进行不同类型的可视化，可以使用Matplotlib [](http://matplotlib.org/)，这是一个用于可视化的Python 2D库。要获得Matplotlib，你可以按照他们的安装说明:[](http://matplotlib.org/users/installing.html)。

&emsp;&emsp;让我们导入可视化库Matplotlib:

```
Import matplotlib.pyplot as plt
#The next line will allow us make inlines plots that could appear
directly in the notebook
#without poping up in a different window
Matplotlib inline
```
&emsp;&emsp;现在，我们使用散点图来可视化广告数据特性和响应变量之间的关系:

```
fig, axs=plt.subplots{1,3,sharey=True}
#Adding the scatterplots to the grid
Advertising_data.plot(kind=’scatter’,x=’TV’,y=’sales’,ax=axs[0],figsize=(16,8))
Advertising_datd.plot(kind=’scatters’,x=’radio’,y=’sales’,ax=axs[1])
Advertising_datd.plot(kind=’scatters’,x=’radio’,y=’sales’,ax=axs[2])
```
<div align="center">
<img src=https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE1.png>
</div>
<div align="center">
图1 广告数据特征和各特征变量的散点图
</div>


&emsp;&emsp;现在，我们需要看看广告将如何帮助增加销量。所以，我们需要问自己几个问题。有价值的问题像是广告和销量之间的关系，哪种广告对销量的贡献更大，以及每种广告对销量的大致影响。我们将尝试用一个简单的线性模型来回答这些问题。

### 简单回归模型
&emsp;&emsp;线性回归模型是一种学习算法，它使用解释特征(或输入或预测因子)的组合来预测定量(也称为数值)响应。

&emsp;&emsp;只有一个特征的简单线性回归模型的形式如下:

&emsp;&emsp;&emsp;&emsp;y = beta0 + beta1x

&emsp;&emsp;这里：<br>
&emsp;&emsp;●y为预测数值(response) c销量<br>
&emsp;&emsp;●x是输入变量<br>
&emsp;&emsp;●beta0叫做截距<br>
&emsp;&emsp;●beta1是输入变量x的电视广告的系数

&emsp;&emsp;beta0和beta1都被认为是模型的系数。为了创建一个模型, 在广告的例子中该模型可以预测销量的价值,我们需要学习这些系数是因为beta1的度量了x 对y的影响。例如,如果beta1 = 0.04,这意味着额外的100美元花在电视广告上将使得销量增加4个单位。所以，我们需要继续看看我们应该如何学习这些系数。

### 学习模型的系数
&emsp;&emsp;为了估计模型的系数，我们需要用回归线对数据进行拟合，回归线给出的答案与实际销量情况类似。为了得到最适合数据的回归线，我们将使用一种称为最小二乘的准则。因此，我们需要找到一条使预测值和实际值之间的差最小的直线。换句话说，我们需要找到一个回归线，使残差平方和最小。图2说明了这一点:

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE2.png">
</div>
<div align="center">
图2 使残差平方和达到最小的直线回归拟合
</div>


&emsp;&emsp;下面是图2中存在的元素:

&emsp;&emsp;●黑点表示x(电视广告)和y(销售额)的实际或观察值<br>
&emsp;&emsp;●蓝线表示最小二乘直线(回归线)<br>
&emsp;&emsp;●红线表示残差，残差是预测值和观察值(实际值)之间的差值

&emsp;&emsp;这就是我们的系数与最小二乘直线(回归线)的关系:

&emsp;&emsp;●beta0是截距，也就是x =0时y的值<br>
&emsp;&emsp;●beta1是斜率，它表示y的变化量除以x的变化量

&emsp;&emsp;图3给出了对此的图形解释:

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE3.png">
</div>
<div align="center">
图3 最小二乘直线和回归系数的关系
</div>


&emsp;&emsp;现在，让我们开始使用Statsmodels来学习这些系数:

```
#To use the formula notation below,we need to import the module like the following
Import statsmodels.formula.api as smf
#create a fitted model in one line of code(which will represent the least squares line)
Lm=smf.ols(formula=’sales~TV’,data=advertising_data).fit()
#show the trained model cofficients
Lm.params
```
Output:

```
Intercept     7.032594
TV            0.047537
dtype         float64
```
&emsp;&emsp;正如我们所提到的，线性回归模型的优点之一是很容易解释，所以让我们继续解释这个模型。

### 解释模型系数
&emsp;&emsp;我们来看看如何解释模型的系数，例如TV广告系数(beta1):

&emsp;&emsp;投入/功能(电视广告)支出增加1单位会使销量(响应)增加0.047537个单位。换句话说，在电视广告上多花100美元，就会增加4.7537个单位的销量。<br>

&emsp;&emsp;从电视广告数据中建立一个学习模型的目的是预测未知数据的销售情况。那么，让我们看看如何使用学习过的模型来基于电视广告的给定价值来预测销售额(我们不知道)。<br>

### 使用模型进行预测
&emsp;&emsp;假设我们有未知的电视广告支出数据，我们想知道它们对公司销售情况的影响。所以，我们需要用学习过的模型来完成。假设我们想知道5万美元的电视广告会增加多少销售额。

&emsp;&emsp;让我们用我们学过的模型系数来做这样的计算：

&emsp;&emsp;&emsp;&emsp;&emsp;y = 7.032594 + 0.047537*x

```
#manually calculating the increase in the sales based on $50k
7.032594+0.047537*50000
```

Output:<br>
`
9,409.444
`

&emsp;&emsp;我们也可以用Statsmodels来做预测。首先，我们需要提供pandas DataFrame的电视广告花费情况，因为Statsmodels的预测要求:

```
#creating a Pandas Dataframe to match Statsmodels interface expectations
new_TVAdSpending=pd.DataFrame({‘TV’:[50000]})
new_TVAdSpending.head()
```
Output:

![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E8%A1%A82.png)

&emsp;&emsp;现在，我们可以继续使用predict函数来预测销售价值:

```
#use the model to make predictions on new value
Preds = lm.predict{new_TVAdSpending}
```
Output:

`
Array([ 9.40942557 ])
`

&emsp;&emsp;让我们看看学习过的最小二乘直线是怎样的。为了画出这条线，我们需要两个点，每个点都用这对来表示:( x,predict_value_of_x)。<br>
&emsp;&emsp;那么，让我们取电视广告的最小值和最大值:

```
# create a DataFrame with the minimum and maximum values of TV
X_min_max=pd.DataFrame({‘TV’:[advertising_data.TV.min(),advertising_data.TV.max()]})
X_min_max.head()
```
Output:

![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E8%A1%A82.png)<br>
&emsp;&emsp;让我们得到这两个值对应的预测值：

```
# predictions for X min and max values
predictions=lm.predict(X_min_max)
predictions
```
Output:

`
Array([7.0658692,21.12245377])
`

&emsp;&emsp;现在，让我们画出实际数据然后用最小二乘线来拟合:

```
#plotting the actual observeddata
advertising_data.plot(kind=’scatter’,x=’TV’,y=’sales’)
#plotting the least squares line
Plt.plot (new_TVAdSpending,  preds, c=’red’, linewidth=2)
```

Output:

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE4.png">
</div>
<div align="center">
图4 真实值与最小二乘回归直线
</div>

&emsp;&emsp;本示例的扩展和进一步的解释将在下一章进行解释。

## 2.2 线性分类模型
&emsp;&emsp;在本节中，我们将讨论logistic回归，这是被广泛使用的分类算法之一。

&emsp;&emsp;logistic回归是什么？logistic 回归的简单的定义：它是一种线性判别的分类算法。

&emsp;&emsp;我们用以下两点来说明这个定义：

&emsp;&emsp;1. 与线性回归不同，logistic回归在给定一组特征或输入变量的情况下，不会尝试去估计或预测这些被给定的数值变量的值。相反，logistic回归算法的输出是给定样本或观察值属于特定类的概率。简单地说，假设我们有一个二元分类的问题。在这种类型的问题中，我们输出变量中只有两个类，例如，患病或不患病。某特定样本属于患病类的概率为P0且该样本属于非患病类的概率为P1=1-P0。因此，logistic回归算法的输出总是在0-1之间。<br>
&emsp;&emsp;2.你也许知道有很多用于回归或分类的学习算法，并且每种学习算法对数据样本都有自己的假设。对所选定的数据选择合适的学习算法的能力将会随着对这个主题的实践和良好理解而逐渐产生。因此，logistic回归算法的中心假设是，我们的输入空间或特征空间可以被一个线性曲面分割成两个区域（每个类一个），如果我们只有两个特征，它可以是一条线，如果我们有三个特征，它可以是一个平面，以此类推。这个分类边界的位置和方向将由选定的数据决定。如果选定的数据满足这个约束条件，即能够将选定样本分隔成与每个类对应的具有线性曲面的区域，那么说明你选定的数据是线性可分的。下图5说明了这个假设。在图5中，用三维空间展示：输入或特征以及两个可能的类：患病的（红色）和非患病的（蓝色）。分界面将这个区域区分成两个区域，因为它是线性的并且帮助模型区分属于不同类别的样本，因此称这是一个线性判别。

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE5.png">
</div>
<div align="center">
图5 分为两类的线性曲面分割图
</div>

&emsp;&emsp;如果你的样本数据不是线性可分离的，你可以将你的数据转换到更高纬度的空间，通过添加更多的特性来实现。

### 2.2.1 分类与逻辑回归
&emsp;&emsp;在前一节中，我们学习了如何预测连续型的数量（例如，电视广告对公司销售的影响）作为输入值的线性函数（例如，电视、广播和报纸广告）。但对某些情况而言，输出将不再是连续型的量。例如，预测某人是否患病是一个分类问题，我们需要一个不同的学习算法来解决这一问题。在本节中，我们将深入研究Logistic回归的数学分析，这是一种用于分类任务的学习算法。

&emsp;&emsp;在线性回归中，我们试图用线性函数模型     来预测数据集中第i个样本x(i)的输出变量y(i)的值。对于诸如二进制标签(y(i)∈{0，1})之类的分类任务，以上的线性函数模型不是一个很好的解决方案。

&emsp;&emsp;Logistic回归是我们可以用于分类任务的众多学习算法之一，我们使用不同的假设类，同时试图预测特定样本属于一类的概率和它属于零类的概率。因此，Logistic回归中，我们将尝试学习以下函数：

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%85%AC%E5%BC%8F1.png">
</div>

方程![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE10.png)通常被称为	sigmoid或logistic函数，它将回归值压缩到一个固定的范围[0,1]，如下图所示。因为z的输出值被压缩在[0,1]之间，我们可以将 理解为一个概率。

&emsp;&emsp;我们的目标是寻找参数 的值，使得当输入样本x属于一类 的概率大于该样本属于零类的概率：

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE6.png">
</div>
<div align="center">
图6 sigmoid函数形态
</div>


&emsp;&emsp;因此，假设我们有一组训练样本，他们有相应的二进制标签           。我们需要最小化以下成本函数，该函数能够衡量给定参数的性能。


<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%85%AC%E5%BC%8F2.png">
</div>

&emsp;&emsp;注意，对于每个训练样本，以上方程的两项之和中只有一项是非零的（取决于标签 的值是0还是1）。当y(i)=1时，最小化模型成本函数意味着我们需要使参数变大，当y=0时 使1-h(θ)变大。

&emsp;&emsp;现在，我们有一个成本函数来计算给定的h(θ)与我们的训练样本的匹配程度。我们可以学习使用优化技术对训练样本进行分类，使J(θ)最小化，并找到参数θ的最佳值.一旦完成这一任务，我们就可以使用这些参数将一个新的测试样本分为1或0类，检查这两个类中哪类是最可能的。如果在定x条件下P(y=1)<p(y=0)，则输出0，否则输出1，这与类之间定义0.5的阈值并检查h(x)>0.5是相同的。

&emsp;&emsp;为了使成本函数J(θ)最小化，我们可以使用一种优化技术来找到使成本函数最小化的最佳参数值θ。我们可以使用一个叫做梯度的微积分工具，它用来找到成本函数的最大增长率。然后，我们可以用相反的方向来求这个函数的最小值；例如J(0)的梯度用VJ(0)来表示，即对模型参数的成本函数取梯度。因此，我们需要提供一个函数来计算J(0)和VJ(9)的值，以供对任意的参数θ进行选择。如果我们对J(0)上的代价函数求关于0的梯度或导数，我们会得到如下结果：

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%85%AC%E5%BC%8F3.png">
</div>

&emsp;&emsp;用向量形式表示为:

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%85%AC%E5%BC%8F4.png">
</div>

&emsp;&emsp;现在，我们对逻辑回归有了数学上的理解，让我们继续使用这种新的学习方法来解决一个分类任务
 
## 2.3 泰坦尼克号模型的建立和训练
&emsp;&emsp;泰坦尼克号的沉没是历史上最臭名昭著的事件之一。这起事故导致2,224名乘客和机组人员中有1,502人死亡。在这个问题中，我们将使用数据科学来预测乘客是否能在这场悲剧中幸存下来，然后根据悲剧的实际统计数据来测试我们模型的性能。

&emsp;&emsp;为了了解泰坦尼克号的例子，你需要做以下事情：

&emsp;&emsp;1. 单击 [](http://github.com/ ahmed-menshawy/ML_Titannic/archive/master.zip),在ZIP文件中下载此存储库。或从终端执行:<br>
&emsp;&emsp;2. Git克隆: [](https://github.com/ahmed-menshawy/ML_Titanic.git)<br>
&emsp;&emsp;3. 安装(virtualenv)[](http://virtualenv.readthedocs.org/en/latest/installation.html)<br>
&emsp;&emsp;4. 导航到解压克隆repo的目录，并用virtualenv ml_titania创建虚拟环境<br>
&emsp;&emsp;5. 使用aource ml_titanic/bin/ Activate激活环境<br>
&emsp;&emsp;6. 使用pip Install -r requirements.txt安装所需的依赖关系<br>
&emsp;&emsp;7. 从命令行或终端执行ipython笔记本<br>
&emsp;&emsp;8. 遵循本章中的示例代码<br>
&emsp;&emsp;9. 完成后，用deactivate命令关闭虚拟环境<br>

### 2.3.1 数据处理和可视化
&emsp;&emsp;在本节中，我们将做一些数据预处理和分析。数据探索和分析被认为是应用机器学习的最重要的步骤之一，也可能被认为是最重要的步骤，因为在这个步骤中，你会了解你的朋友-----数据，它会在训练过程中一直伴随着你。此外，了解你的数据将使你能够缩小可能用于检查哪个算法最适合你的数据的候选算法集。

&emsp;&emsp;让我们从导入实现所需的包开始:

```
import matplotlib.pyplot as plt
%matplotlib inline

from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import series,DataFrame
from patsy import dmatrices
from sklearn import datdsets,svm

import numpy as np
import pandas as pd
import statamodels.api as sm

from scipy import stats
stats.chisqprob=lambds chisq,df:stats.chi2.sf(chisq,df)
```

&emsp;&emsp;让我们用pandas读入泰坦尼克号乘客和船员的数据:

`
&emsp;&emsp;Titanic_data=pd.read_csv(“data/titanic_train.csv”)
`

&emsp;&emsp;接下来，让我们查看数据集的维度，并且看看我们有多少样本数据，有多少解释性特征描述了我们的数据集:

`
&emsp;&emsp;Titanic_data.shape
`

Output :<br>
`
{891,12}
`

&emsp;&emsp;因此，我们总共有891个观测值，数据样本，或乘客/机组记录，以及描述这一记录的12个解释特征:

```
List(titanic_data)
Output :
{‘PassengerId’,’Survived’,’Pclass’,’Name’,’Sex’,’Age’,’SibSp’,’Parch’,’Ticket’,’Fare’,’Cabin’,’Embarked’}
```

&emsp;&emsp;让我们看看一些样本/观察数据:

`
Titanic_data[500:510]
`
Output:

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE7.png">
</div>
<div align="center">
图7 泰坦尼克号数据库样本
</div>

&emsp;&emsp;现在，我们有一个pandas数据模型，它包含了我们需要分析的891名乘客的信息。DataFrame的列表示每个乘客/机组人员的解释性特征，比如姓名、性别或年龄。<br>
&emsp;&emsp;这些解释特性中有一些是完整的，没有任何缺失值，例如幸存特性，它有891个条目。其他解释性特性包含了缺失的值，例如年龄特性，它只有714个条目。DataFrame中的任何缺失值都表示为NaN。<br>
&emsp;&emsp;如果你研究所有的数据集特性，你会发现机票和客舱特性有许多缺失值(NaNs)，因此它们不会为我们的分析增加多少价值。为了处理这个问题，我们将从DataFrame中删除它们。

&emsp;&emsp;使用以下代码完全从DataFrame删除机票和客舱功能:

`
itanic_data=titanic_data.drop([‘Ticket’,’Cabin’],axis=1)
`

&emsp;&emsp;在我们的数据集中存在这样的缺失值有很多原因。但是为了保持数据集的完整性，我们需要处理这些缺失的值。一般我们会选择删除缺失值。<br>
&emsp;&emsp;使用以下代码将所有剩余特性中的缺失值删除

`
&emsp;&emsp;titanic_data=titanic_data.dropna()
`

&emsp;&emsp;现在，我们有一个竞争数据集，我们可以用来做分析。如果你决定只删除所有NaNs，而不首先删除机票和舱位特性，你将发现大多数数据集被删除，因为.dropna() 方法从DataFrame中删除一个观察值，即使其中一个特性中只有一个缺失值。

&emsp;&emsp;我们做一些数据可视化，看看一些特征的分布，理解解释特征之间的关系:

```
#declaring graph parameters
Fig=plt.figre(figsize=(18,6))
alpha=alpha_scatterplot=0.3
alpha_bar_chart=0.55
#defining a grid of subplots to contain all the figures
Ax1=plt.subplot2grid((2,3),(0,0))
#Add the first bar plot which represents the count of people who survived vs not survived.
titanic_data.Survived.value_counts().plot(kind=’bar’,alpha=alpha_bar_chart)
#Adding margins to the plot
Ax1.set_xlim(-1,2)
#Adding bar plot title
plt.title(“Distribution of Survival,(1=Survived)”)
plt.subplot2grid((2,3),(0,1))
plt.scatter(titanic_data.survived,titanic_data.Age,alpha=alpha_scatterplot)
#setting the value of the y label (age)
Plt.ylabel(“Age”)
#formatting the grid
Plt.grid(b=True,which=’major’,axis=’y’)
Plt.title(“Survival by Age,(1=Survived)”)
ax3=plt.subplot2grid((2,3),(0,2))
titanic_data.Pclass.value_counts().plot(kind=”barh”,alpha=alphas_bar_chart)
ax3.set_ylim(-1,len(titanic_data.Plass.value_counts()))
plt.title("Class Distribution")
plt.subplot2grid((2,3),(1,0),colsoan=2)
#plotting kernel density estimate of the subse of the 1st class passenger’s
age
titanic_data.Age[titaic_data.Pclass==1].plot(kind=’kde’)
titanic_data.Age[titaic_data.Pclass==2].plot(kind=’kde’)
titanic_data.Age[titaic_data.Pclass==3].plot(kind=’kde’)
#Adding x label (age) to the plot
Plt.xlabel(“Age”)
Plt.title(“Age Distribution within classes”)
#Add legend to the plot.
Plt.legend((‘1st Class’,’2nd Class’,’3rd Class’),loc=’best’)
ax5=plt.subplot2grid((2,3),(1,2))
titanic_data.Embarked.value_counts().plot(kind=’bar’,alpha=alpha_bar_chart)
ax5.set_xlim(=1,len(titanic_data.Embarked.value_counts()))
plt.title(“Passengers per boarding location”)
```

<div align="center">.
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE8.png">
</div>
<div align="center">
图8 泰坦尼克号样本数据的基本信息图
</div>

&emsp;&emsp;正如我们所提到的，这个分析的目的是基于可用的特性来预测一个特定的乘客是否能在悲剧中幸存，例如旅行级别(数据中称为pclass)、性别、年龄和票价。所以，让我们看看我们是否可以从视觉上更好地理解乘客谁幸存和死亡。

&emsp;&emsp;首先，我们画一个条形图，看看每个类别的观察人数(幸存/死亡):

```
plt.pigure(figsize=(6,4))
fig,ax=plt.subplots()
titanic_data.Survived.value_counts().plot(kind=’barh’,color=“blue”,alpha=.65)
ax.set_ylim(-1,len(titanic_data.Survived.value_counts()))
plt.title(“Breakdown of survivals(0=Died,1=Survived)”)
```
<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE9.png">
</div>
<div align="center">
图9 幸存情况分布图
</div> 

&emsp;&emsp;让我们通过按性别细分前面的图表来对数据有更多的了解:

```
fig=plt.figure(figsize=(18,6))
#Plotting gender based analysis for the survivals.
Male=titanic_data.Survived[titanic_data sex==’male’].value_counts().sort_index()
Female=titanic_data.Survived[titanic_data.Sex==’female’].value_counts().sort_index
ax1=fig.add_subplot(121)
male.plot(kind=’barh’,label=’Male’,aloha=0.55)
female.plot(kind=’barh’,color=’*FA2379’,label=’Female’,alpha=0.55)
plt.title(“Gender analysis of survivals (raw value counts)”);
plt.legend(loc=’best’)
ax1.set_ylim(-1,2)
```

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE10.png">
</div>
<div align="center">
图10 按性别特征预测未来幸村情况
</div>

&emsp;&emsp;现在，关于这两个可能的类(幸存和死亡)，我们有了更多的信息。探索和可视化步骤是必要的，因为它让你更深入地了解数据的结构，并帮助你选择适合你的问题的学习算法。如你所见，我们从非常基本的绘图开始，然后增加了绘图的复杂性，以发现关于我们正在使用的数据的更多信息。

### 2.3.2 数据分析——监督机器学习
&emsp;&emsp;分析的目的是预测幸存者。因此，结果是否会幸存，这是一个二元分类问题;在这里，你只有两种可能的类。

&emsp;&emsp;对于二元分类问题，我们可以使用很多学习算法。逻辑回归就是其中之一。正如维基百科解释的那样:

&emsp;&emsp;在统计、逻辑回归或分对数回归是一种回归分析用于预测分类因变量的结果 (因变量可以承担有限数量的值,其大小是没有意义的,但其大小的顺序可能是也可能不是有意义的)基于一个或多个预测变量。也就是说，它被用来估计定性反应模型中参数的经验值。用逻辑函数将描述单个试验可能结果的概率作为解释变量(预测变量)的函数建模。经常(在本文中也是如此)“logistic回归”专指因变量为二元的问题，即可用类别数为两类以上的问题称为多项logistic回归，如果多个类别是有序的，则称为有序logistic回归。Logistic回归通过概率分数作为因变量的预测值来衡量一个分类因变量和一个或多个自变量之间的关系，这些自变量通常(但不一定)是连续的。[1]处理的问题与使用类似技术的probit回归相同。

&emsp;&emsp;为了使用逻辑回归，我们需要创建一个公式来告诉模型我们给它的特征/输入的类型:

```
#model formula
#here the~sigh is an=sigh,and the feature of our dataset
#are written as a formula to predict survived. The C() lets our
#regression know that those variable are categorical.
#Ref:http://patsy.readthedocs.org/en/latest/formulas.html
Formula=’Survived ~ C(Pclass) + C(Sex)+ Age + SibSp + C(Embarked)’
#create a results dictionary to hold our regression results for easy analysis latter
results{}
#create a regression friendly dataframe using patsy’s dmatrices function
y,x = dmatrices(formula, data=titanic_data, return_type=’dataframe’)
model = sm.Logit(y,x)
# fit our model to the training data
res = model.fit()
# save the result for outputting predictions later
results[‘Logit’] = [res,formula]
res.summary()
```

Output

```
Opitimization terminated successfully
Current function value:0.444388
Iterations 6
```

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE11.png">
</div>
<div align="center">
图11 Logistic回归结果
</div>

&emsp;&emsp;现在，让我们画出实际值与模型的预测值以及残差的图形，残差是目标变量的实际值和预测值之差:

```
# Plot Predictions Vs Actual
plt.figure(figsize=(18,4));

plt.subplot(121,axisbg=”#DBDBDB”)
# generate predictions from our fitted model
ypred = res.predict(x)
plt.plot(x.index,ypred,’bo’,x.index,y,’mo’,alpha=.25);
plt.grid(color=’white’,linestyle=’dashed’)
plt.title(‘Logit predictions,Blue:\nFitted/predicted values: Red’);

#Residuals
ax2 = plt.subplot(122,axisbg=”#DBDBDB”)
plt.plot(res.resid_dev, ‘r-’)
plt.grid(color=’white’,linestyle=’dashed’)
ax2.set_xlim(-1,len(res.resid_dev))
plt.title(‘Logit Residuals’);
```

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE12.png">
</div>
<div align="center">
图12 对Logistic回归模型的理解
</div>

&emsp;&emsp;现在，我们已经建立了逻辑回归模型，在此之前，我们对数据集做了一些分析和探索。前面的示例展示了构建机器学习解决方案的通用管道。

&emsp;&emsp;大多数时候，实践者会陷入一些技术陷阱，因为他们缺乏理解机器学习概念的经验。例如，有人可能在测试集上获得99%的准确率，然后没有对数据中类的分布进行任何调查(比如有多少样本是负的，有多少样本是正的)，他们就建立了模型

&emsp;&emsp;为了突出其中的一些概念，并区分需要注意的不同类型的错误和真正需要关注的错误，我们将继续下一节。

## 2.4 不同类型的错误
&emsp;&emsp;在机器学习中，有两种类型的错误，而作为数据科学的新手，你需要理解两者之间的关键区别。如果你最终将错误类型的错误最小化，那么整个学习系统将是无用的，并且该学习系统将无法实践未知数据，为了减少实践者对两类错误的理解，我们将在下面两节中解释这两类错误。

## 2.5 明显(训练集)错误
&emsp;&emsp;这是你不需要考虑最小化的第一类错误。为这种类型的错误取一个小值并不意味着你的模型可以很好地处理不可见的数据(一般化)。为了更好地理解这种类型的错误，我们将给出一个类场景的简单示例。在课堂上解决问题的目的不是为了在考试中再次解决同样的问题，而是为了能够解决其他问题，而这些问题与你在课堂上练习过的问题必然是相似的。考试问题可能来自同一类课堂问题，但不一定相同。

&emsp;&emsp;明显的错误是经过训练的模型在我们已经知道真实结果/输出的训练集中执行的能力。如果你在训练集上得到0个错误，那么这对你来说是一个很好的指示器，你的模型(大部分)在看不见的数据上不能很好地工作(不会一般化)。另一方面，数据科学是关于使用训练集作为基础知识的学习算法，以工作在未来看不见的数据。

&emsp;&emsp;在图13中，红色曲线表示明显的错误。每当您你提高模型的记忆能力(例如通过增加解释特性的数量来增加模型的复杂性)时，你就会发现这个明显的错误接近于零。可以看出，如果你有和观察/样本一样多的特征，那么明显的误差将为零:

<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE13.png">
</div>
<div align="center">
图13 明显误差（红色）和泛化误差（蓝色）
</div>

## 2.6 泛化/真正的错误
&emsp;&emsp;这是数据科学中第二个也是更重要的错误类型。构建学习系统的全部目的是为了在测试集上获得更小的泛化误差；换句话说，就是想得到一个模型，该模型在一组没有在训练阶段使用过的观察样本上很好的工作。用上一节的课堂考试场景为例，可以把泛化错误看作是解决考试问题的能力，这些问题不一定和你在课堂上解决的问题相似，这样你就可以学习和熟悉这门课。因此，泛化性能是模型使用其在训练阶段学到的技能（参数）的能力，以便正确预测不可见数据的结果或输出。

&emsp;&emsp;在图13中，浅蓝色表示泛化误差。可以看到，随着模型复杂度的增加，泛化误差会减小，直到某个时候模型开始失去其增加的能力，泛化误差会减小。曲线的这一部分，你得到的泛化误差失去了它不断增加泛化能力的这一部分曲线，被称为过拟合。

&emsp;&emsp;本节的要点是尽可能减少泛化误差。

## 2.7 总结
&emsp;&emsp;线性模型是一个非常强大的工具，如果你的数据符合它的假设，你可以使用它作为初始学习算法。理解线性模型将帮助你理解使用线性模型作为构建模块的更复杂的模型。

&emsp;&emsp;接下来，我们将继续使用泰坦尼克号示例，更详细地处理模型复杂性和评估。模型复杂性是一个非常强大的工具，为了提高泛化误差，你需要小心使用它。误解它会导致过度拟合问题。
