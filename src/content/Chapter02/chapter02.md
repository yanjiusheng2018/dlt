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
&emsp;&emsp;Advertising_datd.plot(kind=’scatters’,x=’radio’,y=’sales’,ax=axs[2])<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE2.png)
&emsp;&emsp;现在，我们需要看看广告将如何帮助增加销量。所以，我们需要问自己几个问题。有价值的问题像是广告和销量之间的关系，哪种广告对销量的贡献更大，以及每种广告对销量的大致影响。我们将尝试用一个简单的线性模型来回答这些问题。<br>
### 简单的回归模型
&emsp;&emsp;线性回归模型是一种学习算法，它使用解释特征(或输入或预测因子)的组合来预测定量(也称为数值)响应。<br>
&emsp;&emsp;只有一个特征的简单线性回归模型的形式如下:<br>
&emsp;&emsp;&emsp;&emsp;y = beta0 + beta1x<br>
&emsp;&emsp;这里：<br>
&emsp;&emsp;y为预测数值(response) c销量<br>
&emsp;&emsp;x是输入变量<br>
&emsp;&emsp;beta0叫做截距<br>
&emsp;&emsp;beta1是输入变量x-----电视广告的系数<br>
&emsp;&emsp;beta0和beta1都被认为是模型的系数。为了创建一个模型, 在广告的例子中该模型可以预测销量的价值,我们需要学习这些系数是因为beta1的度量了x 对y的影响。例如,如果beta1 = 0.04,这意味着额外的100美元花在电视广告上将使得销量增加4个单位。所以，我们需要继续看看我们应该如何学习这些系数。<br>
### 学习模型的系数
&emsp;&emsp;为了估计模型的系数，我们需要用回归线对数据进行拟合，回归线给出的答案与实际销量情况类似。为了得到最适合数据的回归线，我们将使用一种称为最小二乘的准则。因此，我们需要找到一条使预测值和实际值之间的差最小的直线。换句话说，我们需要找到一个回归线，使残差平方和最小。图2说明了这一点:<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE3.png)
&emsp;&emsp;下面是图2中存在的元素:<br>
&emsp;&emsp;黑点表示x(电视广告)和y(销售额)的实际或观察值<br>
&emsp;&emsp;蓝线表示最小二乘直线(回归线)<br>
&emsp;&emsp;红线表示残差，残差是预测值和观察值(实际值)之间的差值<br>
&emsp;&emsp;这就是我们的系数与最小二乘直线(回归线)的关系:<br>
&emsp;&emsp;beta0是截距，也就是x =0时y的值<br>
&emsp;&emsp;beta1是斜率，它表示y的变化量除以x的变化量<br>
&emsp;&emsp;图3给出了对此的图形解释:<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE4.png)
&emsp;&emsp;现在，让我们开始使用Statsmodels来学习这些系数:<br>
&emsp;&emsp;#To use the formula notation below,we need to import the module like the following<br>
&emsp;&emsp;Import statsmodels.formula.api as smf<br>
&emsp;&emsp;#create a fitted model in one line of code(which will represent the least squares line)<br>
&emsp;&emsp;Lm=smf.ols(formula=’sales~TV’,data=advertising_data).fit()<br>
&emsp;&emsp;#show the trained model cofficients<br>
&emsp;&emsp;Lm.paramsOutput:<br>
&emsp;&emsp;Intercept        7.032594<br>
&emsp;&emsp;TV             0.047537<br>
&emsp;&emsp;dtype: float64<br>
&emsp;&emsp;正如我们所提到的，线性回归模型的优点之一是很容易解释，所以让我们继续解释这个模型。<br>
### 解释模型系数
&emsp;&emsp;我们来看看如何解释模型的系数，例如TV ad系数(beta1):<br>
&emsp;&emsp;投入/功能(电视广告)支出增加1单位会使销量(响应)增加0.047537个单位。换句话说，在电视广告上多花100美元，就会增加4.7537个单位的销量。<br>
&emsp;&emsp;从电视广告数据中建立一个学习模型的目的是预测未知数据的销售情况。那么，让我们看看如何使用学习过的模型来基于电视广告的给定价值来预测销售额(我们不知道)。<br>
### 使用模型进行预测
&emsp;&emsp;假设我们有未知的电视广告支出数据，我们想知道它们对公司销售情况的影响。所以，我们需要用学习过的模型来完成。假设我们想知道5万美元的电视广告会增加多少销售额。<br>
&emsp;&emsp;让我们用我们学过的模型系数来做这样的计算：<br>
&emsp;&emsp;&emsp;&emsp;&emsp;y = 7.032594 + 0.047537 *x<br>
&emsp;&emsp;#manually calculating the increase in the sales based on $50k<br>
&emsp;&emsp;7.032594+0.047537*50000<br>
&emsp;&emsp;Output:<br>
&emsp;&emsp;9,409.444<br>
&emsp;&emsp;我们也可以用Statsmodels来做预测。首先，我们需要提供pandas DataFrame的电视广告花费情况，因为Statsmodels的预测要求:<br>
&emsp;&emsp;#creating a Pandas Dataframe to match Statsmodels interface expectations<br>
&emsp;&emsp;new_TVAdSpending=pd.DataFrame({‘TV’:[50000]})<br>
&emsp;&emsp;new_TVAdSpending.head()<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE5.png)
&emsp;&emsp;现在，我们可以继续使用predict函数来预测销售价值:<br>
&emsp;&emsp;#use the model to make predictions on new value<br>
&emsp;&emsp;Preds = lm.predict{new_TVAdSpending}<br>
&emsp;&emsp;Output<br>
&emsp;&emsp;Array([9.40942557])<br>
&emsp;&emsp;让我们看看学习过的最小二乘直线是怎样的。为了画出这条线，我们需要两个点，每个点都用这对来表示:( x,predict_value_of_x)。<br>
&emsp;&emsp;那么，让我们取电视广告的最小值和最大值:<br>
&emsp;&emsp;# create a DataFrame with the minimum and maximum values of TV<br>
&emsp;&emsp;X_min_max=pd.DataFrame({‘TV’:[advertising_data.TV.min(),advertising_data.TV.max()]})<br>
&emsp;&emsp;X_min_max.head()<br>
&emsp;&emsp;Output<br>
&emsp;&emsp;TV<br>
&emsp;&emsp;0      0.7<br>
&emsp;&emsp;1      296.4<br>
&emsp;&emsp;让我们得到这两个值对应的预测值：<br>
&emsp;&emsp;# predictions for X min and max values<br>
&emsp;&emsp;predictions=lm.predict(X_min_max)<br>
&emsp;&emsp;predictions<br>
&emsp;&emsp;Output:<br>
&emsp;&emsp;Array([7.0658692,21.12245377])<br>
&emsp;&emsp;现在，让我们画出实际数据然后用最小二乘线来拟合:<br>
&emsp;&emsp;#plotting the actual observeddata<br>
&emsp;&emsp;advertising_data.plot(kind=’scatter’,x=’TV’,y=’sales’)<br>
&emsp;&emsp;#plotting the least squares line<br>
&emsp;&emsp;Plt.plot(new_TVAdSpending,preds,c=’red’,linewidth=2)<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE7.png)
&emsp;&emsp;本示例的扩展和进一步的解释将在下一章进行解释。<br>
## 2.2 线性分类模型
&emsp;&emsp;在本节中，我们将讨论logistic回归，这是被广泛使用的分类算法之一。<br>
&emsp;&emsp;logistic回归是什么？logistic 回归的简单的定义：它是一种线性判别的分类算法。<br>
&emsp;&emsp;我们用以下两点来说明这个定义：<br>
&emsp;&emsp;1. 与线性回归不同，logistic回归在给定一组特征或输入变量的情况下，不会尝试去估计或预测这些被给定的数值变量的值。相反，logistic回归算法的输出是给定样本或观察值属于特定类的概率。简单地说，假设我们有一个二元分类的问题。在这种类型的问题中，我们输出变量中只有两个类，例如，患病或不患病。某特定样本属于患病类的概率为P_0且该样本属于非患病类的概率为P_1=1-P_0。因此，logistic回归算法的输出总是在0-1之间。<br>
&emsp;&emsp;2.你也许知道有很多用于回归或分类的学习算法，并且每种学习算法对数据样本都有自己的假设。对所选定的数据选择合适的学习算法的能力将会随着对这个主题的实践和良好理解而逐渐产生。因此，logistic回归算法的中心假设是，我们的输入空间或特征空间可以被一个线性曲面分割成两个区域（每个类一个），如果我们只有两个特征，它可以是一条线，如果我们有三个特征，它可以是一个平面，以此类推。这个分类边界的位置和方向将由选定的数据决定。如果选定的数据满足这个约束条件，即能够将选定样本分隔成与每个类对应的具有线性曲面的区域，那么说明你选定的数据是线性可分的。<br>
&emsp;&emsp;下图5说明了这个假设。在图5中，用三维空间展示：输入或特征以及两个可能的类：患病的（红色）和非患病的（蓝色）。分界面将这个区域区分成两个区域，因为它是线性的并且帮助模型区分属于不同类别的样本，因此称这是一个线性判别。<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE8.png)
&emsp;&emsp;如果你的样本数据不是线性可分离的，你可以将你的数据转换到更高纬度的空间，通过添加更多的特性来实现。<br>
### 2.2.1 分类与逻辑回归
&emsp;&emsp;在前一节中，我们学习了如何预测连续型的数量（例如，电视广告对公司销售的影响）作为输入值的线性函数（例如，电视、广播和报纸广告）。但对某些情况而言，输出将不再是连续型的量。例如，预测某人是否患病是一个分类问题，我们需要一个不同的学习算法来解决这一问题。在本节中，我们将深入研究Logistic回归的数学分析，这是一种用于分类任务的学习算法。<br>
&emsp;&emsp;在线性回归中，我们试图用线性函数模型 来预测数据集中第 样本 的输出变量 的值。对于诸如二进制标签(ye（0，1）)之类的分类任务，以上的线性函数模型不是一个很好的解决方案。<br>
&emsp;&emsp;Logistic回归是我们可以用于分类任务的众多学习算法之一，我们使用不同的假设类，同时试图预测特定样本属于一类的概率和它属于零类的概率。因此，Logistic回归中，我们将尝试学习以下函数：<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE9.png)
&emsp;&emsp;方程![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE10.png) 通常被称为	sigmoid或logistic函数，它将 的值压缩到一个固定的范围[0，1]，如下图所示。因为z的输出值被压缩在[0，1]之间，我们可以将 理解为一个概率。<br>
&emsp;&emsp;我们的目标是寻找参数 的值，使得当输入样本x属于一类 的概率大于该样本属于零类的概率：<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE11.png)
&emsp;&emsp;假设我们有一组训练样本，他们有相应的二进制标签。我们需要最小化以下成本函数，该函数能够衡量给定ho的性能。<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter02/chapter02_image/%E5%9B%BE12.png)
&emsp;&emsp;注意，对于每个训练样本，以上方程的两项之和中只有一项是非零的（取决于标签 的值是0还是1）。当 =1时，最小化模型成本函数意味着我们需要使 变大，当y=0时 使1- 变大。<br>
&emsp;&emsp;现在，我们有一个成本函数来计算给定的 与我们的训练样本的匹配程度。我们可以学习使用优化技术对训练样本进行分类，使 最小化，并找到参数 的最佳值.一旦完成这一任务，我们就可以使用这些参数将一个新的测试样本分为1或0类，检查这两个类中哪类是最可能的。如果P(y=1)<p(y=0)，则输出0，否则输出1，这与类之间定义0.5的阈值并检查h(x)>0.5相同的。<br>
&emsp;&emsp;为了使成本函数 最小化，我们可以使用一种优化技术来找到使成本函数最小化的最佳参数值 。我们可以使用一个叫做梯度的微积分工具，它用来找到成本函数的最大增长率。然后，我们可以用相反的方向来求这个函数的最小值；例如J(0)的梯度用VJ(0)来表示，即对模型参数的成本函数取梯度。因此，我们需要提供一个函数来计算J(0)和VJ(9)的值，以供对任意的参数 进行选择。如果我们对J(0)上的代价函数求关于0的梯度或导数，我们会得到如下结果<br>
![](https://github.com/yanjiusheng2018/dlt/blob/master/image/tupian2.jpg)
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
&emsp;&emsp;<br>
