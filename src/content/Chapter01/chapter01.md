# 第01章  鸟瞰 数据科学
&emsp;&emsp;据科学或机器学习是让机器能够从数据集中学习而不被告知或编程的过程。
例如，这是极其困难地去编写一个程序，该程序可以手写数字作为输入图像，并根据所写的图像输出0到9的值。这同样适用于将传入邮件归类为垃圾邮件或非垃圾邮件的任务。为了解决这些任务，数据科学家使用来自数据科学或机器学习领域的学习方法和工具来教计算机如何自动识别数字，通过给计算机一些能够区分一个数字与另一个数字的解释性特征。同样，对于垃圾邮件问题，我们可以通过特定的学习算法来教计算机如何区分垃圾邮件和非垃圾邮件，而不是使用正则表达式和写上百条规则来对传入的电子邮件进行分类。  
&emsp;&emsp;对于垃圾邮件过滤应用程序，您可以使用基于规则的方法对其进行编码，但是它不足以用于生产中，比如邮件服务器中。建立一个学习系统是一个理想的解决方案。  
&emsp;&emsp;你可能每天都使用数据科学的应用程序，通常不知道它。例如，您的国家可能正在使用一个系统来检测您邮寄的信件的邮政编码，以便自动将其转发到正确的区域。如果你正在使用亚马逊，他们经常给你推荐要买的东西，他们通过学习你经常搜索或购买的东西来做到这一点。  
&emsp;&emsp;构建一个经过训练的机器学习算法将需要一个历史数据样本库，从这些样本库中，它将学习如何区分不同的示例，并从这些数据中得出一些知识和趋势。之后，被训练算法可用于对未知数据进行预测。从该数据中学习算法将使用原始的历史数据，并试图提出一些知识和趋势。在本章中，我们将鸟瞰数据科学，看它是如何作为一个黑匣子工作的，以及数据科学家每天面临的挑战。我们将讨论以下主题：
* 从一个实例理解数据科学
* 数据科学算法的设计过程
* 学会学习
* 鱼类识别检测模型的实现
* 不同学习类型
* 数据规模与行业需求
## 从一个实例理解数据科学

&emsp;&emsp;为了说明构建特定数据的学习算法的过程和挑战，让我们考虑一个真实的例子。自然资源保护局正在与其他捕鱼公司和伙伴合作，监测捕鱼活动，去保护未来的渔业。因此，他们希望将来使用相机来扩大这种监控过程。通过部署这些相机将产生的数据量将非常繁琐，并且手工处理将非常昂贵。因此，该局希望开发一种学习算法，自动检测和分类不同种类的鱼，以加快视频审查过程。
图1.1显示了由水利部署的相机拍摄的图像样本。这些图像将被用于构建系统。
<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter01/chapter01_image/1.1.jpg">
</div>
<div align="center">
图1.1 水利部署的相机拍摄的图像样本
</div>
&emsp;&emsp;因此，我们在这个例子中的目标是分离不同种类的鱼类，例如鲔鱼、鲨鱼和渔船捕获的更多物种。作为一个说明性的例子，我们可以将问题限制在只有两个类，金枪鱼和月鱼。
<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter01/chapter01_image/1.2.jpg">
</div>
<div align="center">
图1.2 金枪鱼（左）和月鱼（右）
</div>  
&emsp;&emsp;在限制我们的问题只包含两种类型的鱼之后，我们可以从收集到的随机图像中取样并开始注意到这两种类型之间的一些物理差异。例如，考虑以下物理差异：<br>
长度：你可以看到，相比于月鱼，金枪鱼是更长的。<br>
宽度：奥帕鱼比金枪鱼更宽 <br>
颜色：你可以看到，奥帕鱼往往更红，而金枪鱼往往是蓝色和白色，等等。<br>
&emsp;&emsp;我们可以利用这些物理差异作为特征，可以帮助我们的学习算法(分类器）区分这两种类型的鱼。<br>
&emsp;&emsp;在日常生活中，我们常用一个物体的解释性特征来区分围绕在我们周围的物体。甚至婴儿也用这些解释性的特征来了解周围的环境。对于数据科学也是一样的，为了构建一个能够区分不同对象（例如，鱼类类型）的学习模型，我们需要给它一些要学习的解释性特征（例如，鱼类长度）。为了使模型更加确定，减少混淆误差，可以在一定程度上增加对象的解释特征。<br>  
&emsp;&emsp;鉴于这两种类型的鱼之间存在物理差异，这两种不同的鱼种群具有不同的模型或描述。因此，我们的分类任务的最终目标是让分类器学习这些不同的模型，然后给出这两种类型之一的图像作为输入。分类器将通过选择对应于该图像的模型（金枪鱼模型或奥帕鱼模型）来对其进行分类。<br>  
&emsp;&emsp;在这种情况下，金枪鱼和奥帕鱼的集合将作为我们的分类器的知识基础。最初，知识库（训练样本）将被标记，对于每个图像，您将事先知道它是金枪鱼还是月鱼。因此，分类器将利用这些训练样本对不同类型的鱼进行建模，然后利用训练阶段的输出来自动标记分类器在训练阶段没有看到的未标记的鱼。<br>  
&emsp;&emsp;这种未标记的数据通常被称为测试集数据。生命周期的训练阶段如下图所示：<br>
<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter01/chapter01_image/1.3.jpg">
</div>
<div align="center">
图1.3 训练过程
</div>  
&emsp;&emsp;监督数据科学就是从具有已知目标或输出的历史数据中学习，例如鱼型，然后使用这个学习模型来预测我们不知道目标的情况或数据样本。  
&emsp;&emsp;让我们来看看分类器的训练阶段将如何工作：
预处理：在这个步骤中，我们将尝试使用相关的细分技术从图像中分割鱼。
特征提取：在通过减去背景从图像中分割出鱼之后，我们将测量每个图的物理差异（长度、宽度、颜色等）。最后，你会得到如图1.4所示的东西。
<div align="center">
<img src="https://github.com/yanjiusheng2018/dlt/blob/master/src/content/Chapter01/chapter01_image/1.4.jpg">
</div>
<div align="center">
图1.4  两种鱼的长度测量的直方图
</div>   
&emsp;&emsp;最后，我们将这些数据馈入分类器，以模拟不同的鱼类型。  
&emsp;&emsp;正如我们所看到的，我们可以根据我们提出的物理差异（特征），如长度、宽度和颜色，在视觉上区分金枪鱼和欧帕鱼。
我们可以用长度特征来区分这两种鱼。因此，我们可以通过观察鱼的长度和观察鱼是否超过某个值来区分它们。
因此，基于我们的训练样本，我们可以得出以下规则：
```
if length(fish)>length*then label(fish)=Tuna
overwise label(fish)=opah
```
&emsp;&emsp;正如你可能注意到的，由于两个直方图之间有重叠，所以这不是一个我们预期想得到的结果，因为长度特性不仅仅是唯一的用于区分两种类型鱼的完美特性。因此，我们可以尝试结合更多的特征，如宽度，然后组合它们。  
&emsp;&emsp;所以，如果我们设法测量训练样本的宽度，我们可以得到如下直方图：

**图1.5两种鱼的宽度测量的直方图**  
&emsp;&emsp;正如你所看到的，仅仅依赖于一个特性不会给出准确的结果，并且输出模型会执行许多错误分类。相反，我们可以将这两个特征结合起来，得出一些看起来合理的东西。  
&emsp;&emsp;因此，如果我们结合这两个特征，我们可能会得到如下图形：

**图1.6 两种鱼的长度和宽度测量组合**  
&emsp;&emsp;结合阅读资料长度和宽度特性，我们将得到像前一个图中的一个散点图。我们有红点表示金枪鱼，绿点表示月鱼，我们可以建议这条黑线作为规则或决策边界，以区分这两种类型的鱼。  
&emsp;&emsp;例如，如果一条鱼的读数高于这个决策边界，那么它是金枪鱼；否则，它将被预测为月鱼。我们可以设法增加规则的复杂度，以避免任何错误，并获得如下图中所示的决策边界：

**图 1.7  基于训练数据增加定义边界规则的复杂度几面避免定义错误**  
&emsp;&emsp;该模型的优点是，我们得到几乎0个错误分类的训练样本。但实际上这并不是使用数据科学的目标。数据科学的目标是建立一个模型，能够对未观测的数据进行概括和执行。为了检验我们是否建立了一个恰当的模型，我们将引入一个新的阶段，称为测试阶段。其中，我们给训练过的模型一个未被识别的图像，并期望模型可以分配正确的标签。也就是金枪鱼和月鱼。  
&emsp;&emsp;数据科学的最终目标是建立一个模型，它将在实际生活中很好地工作，而不是仅仅在训练集。所以当你看到你的模型在训练集上表现出色就像图1.7一样，不要高兴太早。通常，这种模型在识别图像中鱼的类型方面不能很好地工作。过度拟合使你的模型只在训练集上工作得很好，并且大多数从业人员都陷入了这个陷阱。
相比较提出这样一个复杂的模型，你可以提出一个不是很复杂的模型，可以概括这个测试阶段。下图显示了使用较不复杂的模型以获得较少的分类错误并归纳训练数据：

**图1.8 用一个较不复杂的模型去归纳测试集数据**
## 数据科学算法的设计过程
&emsp;&emsp;不同的学习系统通常遵循相同的设计步骤。他们首先获取知识库，从数据中选择相关的解释性特征，进行一组候选学习算法，同时关注每个算法的性能，最后是评估过程，它度量了训练过程的成功程度。  
&emsp;&emsp;在本节中，我们将更详细地讨论所有这些不同的设计步骤：

**图1.11  模型学习过程**
### 数据预处理
&emsp;&emsp;学习周期的这个组成部分代表了我们算法的知识基础。因此，为了帮助学习算法对不可见的数据进行准确的决策，我们需要以最好的形式提供这种数据库。因此，我们的数据可能需要大量的清洗和预处理（转换）。
#### 数据清洗
&emsp;&emsp;大多数数据集都需要这一步骤，其中要消除错误、噪声和冗余。我们需要我们的数据准确、完整、可靠和无偏，因为使用不好的知识库可能产生许多问题，例如：
* 不准确和有偏的结论
* 误差增大
* 降低了普遍性，这是模型在不可见数据上表现良好的能力。
#### 数据预处理
&emsp;&emsp;在这个步骤中，我们对我们的数据进行转换，使其一致和具体。在预处理数据时，可以考虑很多不同的转换：
* 重新定义、命名（更换标签）：这意味着将分类值转换为数字，如 如果使用一些学习方法，分类值是危险的，而且数字也会在值之间强加一个顺序。
* 重新缩放(归一化）：将连续值转换为一定的范围，通常〔1, 1〕或〔0, 1〕
* 新特征：从现有功能中构建新特征。 例如，肥胖因子=体重/身高
### 特征提取
&emsp;&emsp;样本的解释特征（输入变量）的数量可以是巨大的。$$X<sub>i</sub>=(X<sub>i</sub>^1,X<sub>i</sub>^2,X<sub>i</sub>^3,\cdots,X<sub>i</sub>^d)$$作为训练样本（观察值/例子）d非常大。这方面的一个例子可以是文档分类任务3，其中您得到10000个不同的单词，单词和输入变量将是不同单词的出现次数。  
&emsp;&emsp;大量的输入变量可能会有问题，有时甚至是一个错误，因为我们有很多输入变量和很少的训练样本来帮助我们在学习过程中。为了避免输入变量数量庞大的灾难（维数灾难），数据科学家使用维数减少技术来从输入变量中选择一个子集。例如，在文本分类任务中，他们可以做到以下几点：
* 提取相关输入（例如，互信息测度）
* 主成分分析(主成分分析)
* 分组（簇）相似词（使用相似性度量）
### 模型选择
&emsp;&emsp;这一步是在使用任何降维技术选择输入变量的适当子集之后。选择输入变量的适当子集将使学习过程的其余部分非常简单。  
&emsp;&emsp;在这个步骤中，您正在尝试找出正确的模型来学习。  
&emsp;&emsp;如果你有任何数据科学和应用学习方法的不同领域和不同类型的数据的经验，那么你会发现这一步很容易，因为它需要事先知道你的数据看起来和什么假设可以符合你的数据的性质，并在此基础上选择恰当的学习方法。如果你没有任何先验知识，那也很好，因为你可以通过猜测和尝试不同的参数设置不同的学习方法来做这一步，并选择一个在测试集上提供更好性能的方法。  
&emsp;&emsp;此外，初始数据分析和可视化将有助于你对数据的分布和性质的形式做出一个很好的猜测。
### 学习过程
&emsp;&emsp;通过学习，我们指的是你将用来选择最佳模型参数的优化标准。有各种各样的优化标准：
* 均方误差(MSE)
* 最大似然(ML)准则
* 最大后验概率(MAP)
&emsp;&emsp;优化问题可能难以解决，但模型和误差函数的正确选择有区别。
### 评估你的模型
&emsp;&emsp;在这个步骤中，我们尝试测量模型的泛化误差在不可见的数据上。由于我们只有特定的数据而不事先知道任何不可见的数据，所以我们可以从数据中随机选择一个测试集，并且永远不会在训练过程中使用它，这样它就像有效的不可见的数据一样。有不同的方法可以评估所选模型的性能：
* 简单的保留方法，将数据分割成训练集和测试集
* 基于交叉验证和随机子采样的其他复杂方法  
&emsp;&emsp;本步骤的目的是比较不同模型在相同数据上训练的预测性能，并选择具有较好（较小）测试误差的模型，这将给出比未观测数据更好的泛化误差。通过使用统计方法来测试结果的重要性，您还可以对泛化错误更加确定。
## 开始学习
&emsp;&emsp;构建机器学习系统带来了一些挑战和问题，我们将在本节中讨论它们。这些问题中有很多是特定领域的，而其他则不是。
### 学习的挑战
&emsp;&emsp;下面是您在构建学习系统时通常面临的挑战和问题的概述。
#### 特征提取特征工程
&emsp;&emsp;特征提取是构建学习系统的关键步骤之一。如果通过选择适当的数目的特征，您在这个挑战中做得很好，那么学习过程的其余部分将非常简单。此外，特征提取是领域相关的，它需要先验知识以了解哪些特征对于特定任务可能是重要的。例如，我们的鱼识别系统的特征将与垃圾邮件检测或识别指纹的特征不同。  
&emsp;&emsp;特征提取步骤从您所拥有的原始数据开始。然后建立导出的变量/值（特征），这些变量提供有关学习任务的信息，并有助于下一步的学习和评估（概括）。  
&emsp;&emsp;一些任务将具有大量的特征和较少的训练样本（观察），以便于后续的学习和泛化过程。在这种情况下，数据科学家使用降维技术来将大量的特征减少到更小的集合。
#### 噪声
&emsp;&emsp;在鱼识别任务中，您可以看到长度、重量、鱼颜色以及船的颜色可能变化，并且图像中可能存在阴影、低分辨率的图像和其他对象。所有这些问题都影响了所提出的解释性特征的意义，这些特征应该对我们的鱼类分类任务提供信息。  
&emsp;&emsp;在这种情况下，工作机会会很有帮助。例如，有人可能考虑检测船只ID，并屏蔽船只的某些部分，这些部分很可能不会包含任何要由我们的系统检测到的鱼。这项工作将限制我们的搜索空间。
#### 过度拟合
&emsp;&emsp;正如我们在鱼类识别任务中所看到的，我们试图通过增加模型复杂度和对训练样本的每个实例进行完美分类来提高模型的性能。正如我们将在后面看到的，这样的模型不能处理不可见的数据（比如我们将用来测试模型性能的数据）。经过训练的模型在训练样本上工作得很好，但在测试样本上表现不佳，这就是过度拟合。  
&emsp;&emsp;如果你仔细阅读本章的后半部分，我们建立了一个学习系统，目的是使用训练样本作为我们模型的知识库，以便从中学习，并在看不见的数据上进行推广。训练模型的性能误差对训练数据不感兴趣；相反，我们对训练模型在训练阶段没有涉及的测试样本上的性能（泛化）误差感兴趣。
#### 一种机器学习算法的选择
&emsp;&emsp;有时，您对用于特定任务的模型的执行不满意，并且需要另一类模型。每一个学习策略都有自己的假设，关于它将作为学习基地使用的信息。作为一名信息研究人员，你必须发现哪些怀疑最适合你的信息；通过这样，你将有能力承认尝试了一类模型，并拒绝另一类。
#### 先验知识
&emsp;&emsp;如模型选择和特征提取的概念中所讨论的，如果您具有以下方面的先验知识，则可以处理这两个问题：
* 适当的特征
* 模型选择零件
&emsp;&emsp;对鱼类识别系统中的解释性特征有先验知识使我们能够区分不同类型的鱼类。我们可以通过努力设想我们的信息，并获得一些独特的鱼类分类信息类型的感觉。在此先验知识的基础上，可以选择模型的APT族。
#### 遗漏值
&emsp;&emsp;缺少特征主要是因为缺少数据或选择不告诉选项。在学习过程中，我们如何处理这种情况？例如，假设我们发现特定鱼种的宽度由于某种原因而丢失。有很多方法来处理这些缺失的特征。
## 鱼类识别/检测模型的实现
&emsp;&emsp;为了特别介绍机器学习和深度学习的能力，我们将实现鱼识别示例。不需要理解代码的内部细节。本节的重点是向您介绍一个典型的机器学习管道。  
&emsp;&emsp;我们这个任务的知识基础是一堆图像，每一个都被标记为月鱼或金枪鱼。对于这个实现，我们将使用在成像和计算机视觉领域总体上取得突破的深度学习体系结构之一。这种体系结构被称为卷积神经网络（CNNs）它是一种利用图像处理的卷积运算从图像中提取特征的深层学习体系结构，能够解释我们要分类的对象。现在，您可以把它想象成一个魔术盒，它将拍摄我们的图像，从中学习如何区分我们两个类（月鱼和金枪鱼），然后我们将通过给它添加未标记的图像来测试这个盒子的学习过程，并查看它是否能够分辨出图像中有哪种类型的鱼。  
&emsp;&emsp;不同类型的学习将在后面的部分介绍，所以您稍后将理解为什么我们的鱼识别任务属于监督学习类别。  
&emsp;&emsp;在这个例子中，我们将使用Keras。目前，你可以认为Keras是一个API，它使得构建和使用深度学习方式比平常更容易。让我们开始吧！  
&emsp;&emsp;从KRAS网站我们有：Keras是一个高级的神经网络API，用Python编写，能够在TunSoFalm、CNTK或TeaNo上运行。它的发展重点在于实现快速实验。能够以最少的延迟从想法到结果是进行良好研究的关键。
### 知识库/数据集
&emsp;&emsp;正如我们前面提到的，我们需要一个数据的历史基础，这些数据将用于教导学习算法，学习算法将在以后完成。但我们还需要另一个数据集来测试其在学习过程之后执行任务的能力。综上所述，在学习过程中，我们需要两种类型的数据集：
1.第一个是知识库，在这里我们有输入数据及其对应的标签，例如鱼图像和对应的标签（月鱼或金枪鱼）。这些数据将被提供给学习算法以从中学习，并尝试发现模式/.，这将有助于以后对未标记图像进行分类。
2.第二个主要测试模型将从知识库中学习到的知识应用于未标记图像或未看见数据的能力，并查看它是否工作良好。  
&emsp;&emsp;正如你所看到的，我们只有数据，我们将作为我们学习方法的知识基础。我们手头的所有数据都会有正确的输出。因此，我们需要以某种方式弥补这些数据，这些数据没有任何正确的输出与之关联（我们将要应用模型的输出）。  
&emsp;&emsp;在执行数据科学的时候，我们将做如下工作：
训练阶段：我们从我们的知识库中展示我们的数据并训练我们的模型，将输入数据连同正确的输出一起输入到模型中。
检验阶段：在这个阶段，我们将测量 训练模型正在进行中。此外，我们使用不同的模型属性技术，通过使用（回归的R-平方得分、分类器的分类误差、IR模型的回忆和精度等）来测量训练模型的性能。
测试阶段通常分为两个步骤：
1.在第一步中，我们使用不同的学习方法或模型，并根据验证数据选择性能最好的方法（验证步骤）
2.然后根据测试集（测试步骤）测量和报告所选模型的精度。  
&emsp;&emsp;现在让我们看看我们如何得到这个数据，我们将应用模型，看看它是如何训练。  
&emsp;&emsp;由于我们没有任何没有正确输出的训练样本，所以我们可以从我们将要使用的原始训练样本中构造一个训练样本。所以我们可以把我们的数据样本分成三个不同的集合（如图所示）：

**图1.9:Splitting data into train,validation,and test sets**
* 动车组：将被用作我们模型的知识库，通常是从原始数据样本中得到70%。
* 验证集：这将被用来从一系列模型中选择最佳表演模型，通常是从原始数据样本中得到10%。
* 测试集：这将用于测量和报告所选模型的精度，通常它会和验证集一样大。
如果您只有一个正在使用的学习方法，则可以取消验证集，并将数据重新分割为训练集和测试集。通常，数据科学家使用的是75比25或70比30的比例。
### 数据分析预处理
&emsp;&emsp;在本节中，我们将对输入图像进行分析和预处理，并将其以可接受的格式用于我们的学习算法，这里是卷积神经网络。
因此，让我们从导入这个实现所需的包开始。
```
# Importing the required packages
import numpy as np
np.random.seed(2016)
import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
```
&emsp;&emsp;为了使用数据集中提供的图像，我们需要使它们具有相同的大小。OpenCV是一个很好的选择，这样做，从OpenCV网站：
OpenCV（开源计算机视觉库）是在BSD许可证下发布的，因此可以免费用于学术和商业用途。它具有C++、C、Python和Java接口，支持Windows、Linux、Mac OS、IOS和Android。OpenCV是为计算效率而设计的，并对实时应用有很强的关注。在优化的C/C++中，OpenCV可以利用多核处理。启用OpenCL，它可以利用底层异构计算平台的硬件加速。  
&emsp;&emsp;可以通过使用Python包管理器来安装OpenCV
```
# Parameters
# ----------
# img_path : path
#    path of the image to be resized
def rezize_image(img_path):
   #reading image file
   img = cv2.imread(img_path)
   #Resize the image to to be 32 by 32
   img_resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
   return img_resized
```
&emsp;&emsp;现在我们需要加载我们的数据集的所有训练样本，并根据先前的函数调整每个图像的大小。因此，我们将实现一个函数，该函数将从每种鱼类型的不同文件夹加载训练样本：
```
def load_training_samples():
    #Variables to hold the training input and output variables
    train_input_variables = []
    train_input_variables_id = []
    train_label = []
    # Scanning all images in each folder of a fish type
    print('Start Reading Train Images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
       folder_index = folders.index(fld)
       print('Load folder {} (Index: {})'.format(fld, folder_index))
       imgs_path = os.path.join('..', 'input', 'train', fld, '*.jpg')
       files = glob.glob(imgs_path)
       for file in files:
           file_base = os.path.basename(file)
           # Resize the image
           resized_img = rezize_image(file)
           # Appending the processed image to the input/output variables of the classifier
           train_input_variables.append(resized_img)
           train_input_variables_id.append(file_base)
           train_label.append(folder_index)
    return train_input_variables, train_input_variables_id, train_label
```
&emsp;&emsp;正如我们所讨论的，我们有一个测试集，将作为未知的数据，以测试我们的模型的泛化能力。因此，我们需要对测试图像进行同样的操作，加载图像并进行大小调整处理：
```
def load_testing_samples():
    # Scanning images from the test folder
    imgs_path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(imgs_path))
    # Variables to hold the testing samples
    testing_samples = []
    testing_samples_id = []
    #Processing the images and appending them to the array that we have
    for file in files:
       file_base = os.path.basename(file)
       # Image resizing
       resized_img = rezize_image(file)
       testing_samples.append(resized_img)
       testing_samples_id.append(file_base)
    return testing_samples, testing_samples_id
```
&emsp;&emsp;现在我们需要调用前一个函数到另一个函数中。函数用来加载和调整训练样本的大小。 此外，它将添加几行代码来将训练数据转换为NumPy格式，重新整形该数据以适合我们的分类器，并最终将其转换：
```
def load_normalize_training_samples():
    # Calling the load function in order to load and resize the training samples
    training_samples, training_label, training_samples_id = load_training_samples()
    # Converting the loaded and resized data into Numpy format
    training_samples = np.array(training_samples, dtype=np.uint8)
    training_label = np.array(training_label, dtype=np.uint8)
    # Reshaping the training samples
    training_samples = training_samples.transpose((0, 3, 1, 2))
    # Converting the training samples and training labels into float format
    training_samples = training_samples.astype('float32')
    training_samples = training_samples / 255
    training_label = np_utils.to_categorical(training_label, 8)
    return training_samples, training_label, training_samples_id
```
&emsp;&emsp;我们还需要做同样的测试：
```
def load_normalize_testing_samples():
    # Calling the load function in order to load and resize the testing samples
    testing_samples, testing_samples_id = load_testing_samples()
    # Converting the loaded and resized data into Numpy format
    testing_samples = np.array(testing_samples, dtype=np.uint8)
    # Reshaping the testing samples
    testing_samples = testing_samples.transpose((0, 3, 1, 2))
    # Converting the testing samples into float format
    testing_samples = testing_samples.astype('float32')
    testing_samples = testing_samples / 255
    return testing_samples, testing_samples_id
```
### 模型建立
&emsp;&emsp;现在是创建模型的时候了。正如我们提到的，我们将使用称为CNN的深度学习体系结构作为该鱼识别任务的学习算法。同样，在本章中，您不需要理解任何先前或即将到来的代码，因为我们只演示了如何在Keras和TensorFlow的帮助下通过仅使用几行代码作为深度学习平台来解决复杂的数据科学任务。  
&emsp;&emsp;还请注意，美国有线电视新闻网和其他深度学习架构将在后面的章节中更详细地解释：

**图1.10：CNN architecture**  
&emsp;&emsp;因此，让我们继续创建一个功能，将负责创建CNN架构，将在我们的鱼类识别任务中使用：
```
def create_cnn_model_arch():
    pool_size = 2 # we will use 2x2 pooling throughout
    conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
    kernel_size = 3 # we will use 3x3 kernels throughout
    drop_prob = 0.5 # dropout in the FC layer with probability 0.5
    hidden_size = 32 # the FC layer will have 512 neurons
    num_classes = 8 # there are 8 fish types
    # Conv [32] -> Conv [32] -> Pool
    cnn_model = Sequential()
    cnn_model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
    cnn_model.add(Convolution2D(conv_depth_1, kernel_size, kernel_size, activation='relu',
      dim_ordering='th'))
    cnn_model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    cnn_model.add(Convolution2D(conv_depth_1, kernel_size, kernel_size, activation='relu',
      dim_ordering='th'))
    cnn_model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(2, 2),
      dim_ordering='th'))
    # Conv [64] -> Conv [64] -> Pool
    cnn_model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    cnn_model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, activation='relu',
      dim_ordering='th'))
    cnn_model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    cnn_model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, activation='relu',
      dim_ordering='th'))
    cnn_model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(2, 2),
     dim_ordering='th'))
    # Now flatten to 1D, apply FC then ReLU (with dropout) and finally softmax(output layer)
    cnn_model.add(Flatten())
    cnn_model.add(Dense(hidden_size, activation='relu'))
    cnn_model.add(Dropout(drop_prob))
    cnn_model.add(Dense(hidden_size, activation='relu'))
    cnn_model.add(Dropout(drop_prob))
    cnn_model.add(Dense(num_classes, activation='softmax'))
    # initiating the stochastic gradient descent optimiser
    stochastic_gradient_descent = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)    cnn_model.compile(optimizer=stochastic_gradient_descent,  # using the stochastic gradient descent optimiser
                  loss='categorical_crossentropy')  # using the cross-entropy loss function
    return cnn_model
```
&emsp;&emsp;在开始训练模型之前，我们需要使用模型评估和验证方法来帮助我们评估我们的模型，并了解它的泛化能力。为此，我们将使用一种称为k-折交叉确认. 同样，我们不需要理解这个方法或它是如何工作的，因为我们稍后将详细解释这个方法。  
&emsp;&emsp;因此，让我们开始并创建一个功能，帮助我们评估和验证模型：
```
def create_model_with_kfold_cross_validation(nfolds=10):
    batch_size = 16 # in each iteration, we consider 32 training examples at once
    num_epochs = 30 # we iterate 200 times over the entire training set
    random_state = 51 # control the randomness for reproducibility of the results on the same platform
    # Loading and normalizing the training samples prior to feeding it to the created CNN model
    training_samples, training_samples_target, training_samples_id =
      load_normalize_training_samples()
    yfull_train = dict()
    # Providing Training/Testing indices to split data in the training samples
    # which is splitting data into 10 consecutive folds with shuffling
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    fold_number = 0 # Initial value for fold number
    sum_score = 0 # overall score (will be incremented at each iteration)
    trained_models = [] # storing the modeling of each iteration over the folds
    # Getting the training/testing samples based on the generated training/testing indices by
      Kfold
    for train_index, test_index in kf:
       cnn_model = create_cnn_model_arch()
       training_samples_X = training_samples[train_index] # Getting the training input variables
       training_samples_Y = training_samples_target[train_index] # Getting the training output/label variable
       validation_samples_X = training_samples[test_index] # Getting the validation input variables
       validation_samples_Y = training_samples_target[test_index] # Getting the validation output/label variable
       fold_number += 1
       print('Fold number {} from {}'.format(fold_number, nfolds))
       callbacks = [
           EarlyStopping(monitor='val_loss', patience=3, verbose=0),
       ]
       # Fitting the CNN model giving the defined settings
       cnn_model.fit(training_samples_X, training_samples_Y, batch_size=batch_size,
         nb_epoch=num_epochs,
             shuffle=True, verbose=2, validation_data=(validation_samples_X,
               validation_samples_Y),
             callbacks=callbacks)
       # measuring the generalization ability of the trained model based on the validation set
       predictions_of_validation_samples =
         cnn_model.predict(validation_samples_X.astype('float32'),
         batch_size=batch_size, verbose=2)
       current_model_score = log_loss(Y_valid, predictions_of_validation_samples)
       print('Current model score log_loss: ', current_model_score)
       sum_score += current_model_score*len(test_index)
       # Store valid predictions
       for i in range(len(test_index)):
           yfull_train[test_index[i]] = predictions_of_validation_samples[i]
       # Store the trained model
       trained_models.append(cnn_model)
    # incrementing the sum_score value by the current model calculated score
    overall_score = sum_score/len(training_samples)
    print("Log_loss train independent avg: ", overall_score)
    #Reporting the model loss at this stage
    overall_settings_output_string = 'loss_' + str(overall_score) + '_folds_' + str(nfolds) + '_ep_' + str(num_epochs)
    return overall_settings_output_string, trained_models
```
&emsp;&emsp;现在，在建立模型和使用k-fold交叉验证方法来评估和验证模型之后，我们需要在测试集上报告训练模型的结果。为了做到这一点，我们还将使用k倍的交叉验证，但是这次通过测试来看看我们训练的模型有多好。  
&emsp;&emsp;因此，让我们定义一个函数，它将把经过训练的美国有线电视新闻网模型作为输入，然后使用我们拥有的测试集来测试它们：
```
def test_generality_crossValidation_over_test_set( overall_settings_output_string, cnn_models):
    batch_size = 16 # in each iteration, we consider 32 training examples at once
    fold_number = 0 # fold iterator
    number_of_folds = len(cnn_models) # Creating number of folds based on the value used in the training step
    yfull_test = [] # variable to hold overall predictions for the test set
    #executing the actual cross validation test process over the test set
    for j in range(number_of_folds):
       model = cnn_models[j]
       fold_number += 1
       print('Fold number {} out of {}'.format(fold_number, number_of_folds))
       #Loading and normalizing testing samples
       testing_samples, testing_samples_id = load_normalize_testing_samples()
       #Calling the current model over the current test fold
       test_prediction = model.predict(testing_samples, batch_size=batch_size, verbose=2)
       yfull_test.append(test_prediction)
    test_result = merge_several_folds_mean(yfull_test, number_of_folds)
    overall_settings_output_string = 'loss_' + overall_settings_output_string \ + '_folds_' +
      str(number_of_folds)
    format_results_for_types(test_result, testing_samples_id, overall_settings_output_string)
```
### 模型训练与测试
&emsp;&emsp;现在我们准备通过调用主函数来开始模型训练阶段。  
建设和培训美国有线电视新闻网 使用10倍的交叉验证建模；然后我们可以调用测试函数来测量模型向测试集推广的能力：
```
if __name__ == '__main__':
  info_string, models = create_model_with_kfold_cross_validation()
  test_generality_crossValidation_over_test_set(info_string, models)
```
### 鱼群识别
&emsp;&emsp;在解释鱼识别示例的主要构建块之后，我们准备看到所有连接在一起的代码片段，并了解我们是如何用几行代码构建如此复杂的系统的。完整的代码放置在附录书的一部分。
## 不同学习类型
&emsp;&emsp;据Arthur Samuel所写的《数据科学赋予计算机学习的能力而不需要明确地编程》。所以，任何一块在不进行显式编程的情况下，为了对看不见的数据进行决策而使用训练示例的软件被认为是学习。数据科学或学习有三种不同的形式。

**图1.12 数据科学的不同类型**
### 监督学习
&emsp;&emsp;大多数数据科学家使用监督学习。监督学习是指你有一些解释性的特征，这些特征被称为输入变量（X），并且您有与训练样本相关联的标签，这些标记被称为输出变量（Y）任何有监督学习算法的目的是从输入变量中学习映射函数（X）到输出变量（Y）。  
&emsp;&emsp;因此，监督学习算法将试图近似地从输入变量中学习映射（X）到输出变量（Y），以便以后可以用来预测Y一个看不见的样品的值。

**图1.13显示了任何监督数据科学系统的典型工作流**
&emsp;&emsp;这种学习叫做监督学习因为您得到了与它相关的每个训练样本的输出。在这种情况下，我们可以说，学习过程是由监督者监督的。该算法对训练样本进行决策，并根据数据的正确标签由主管进行校正。当监督学习算法达到可接受的精度水平时，学习过程将停止。
### 无监督学习
&emsp;&emsp;无监督学习被认为是信息研究人员使用的第二种最常见的学习方式。在这种学习中，只有解释性的特征或输入变量（X），没有任何相应的标签或输出变量。  
&emsp;&emsp;无监督学习算法的目标是获取信息中的隐藏结构和实例。这种学习叫做无监督的鉴于没有与训练样本相关的标记。因此，这是一个没有纠错的学习过程，并且该算法将试图找到自己的基本结构。  
&emsp;&emsp;无监督学习可以进一步分为两种形式：聚类和关联任务  
&emsp;&emsp;聚类集群任务是在哪里发现类似的组 训练样本并将它们分组在一起，例如按主题分组文档联想关联规则学习任务是你想发现的地方。描述你的训练样本中的关系的一些规则，比如看电影的人X也倾向于看电影Y

**图1.14显示了我们分散的无监督学习的一个简单例子。**
### 半监督学习

&emsp;&emsp;半监督学习是一种介于监督学习和非监督学习之间的学习，在这里您已经得到了具有输入变量（输入）的培训示例（X），但只有其中的一些标记为输出变量（Y）。  
&emsp;&emsp;这种学习的一个很好的例子是Flickr，其中用户上传了大量图像，但只有部分图像被标记（如日落、海洋和狗），而其余图像没有标记。  
&emsp;&emsp;要解决陷入这种学习的任务，可以使用下列之一或它们的组合：
* 监督学习:训练学习算法进行预测 关于未标记的数据，然后反馈整个训练样本以学习并预测未知数据。
* 无监督学习:使用无监督学习算法来学习解释性特征或输入变量的底层结构，就像你没有任何标记的训练样本一样。
### 强化学习
&emsp;&emsp;机器学习中的最后一种学习形式是强化学习，其中没有监督者，而只有奖励信号。  
&emsp;&emsp;因此，强化学习算法将尝试做出决定，然后会有一个奖励信号来辨别这个决定是对还是错。此外，这种监督反馈或奖励信号可能不会立即出现，但会延迟几个步骤。例如，算法现在将做出决定，但是只有在经过许多步骤之后，奖励信号才能判断决策是好是坏。
## 数据规模与行业需求
&emsp;&emsp;数据是我们学习计算的信息基础，没有信息，任何令人振奋和富有想象力的思想都将一文不值。所以，如果你有一个像样的信息科学应用与正确的信息，那么你就准备好了。  
&emsp;&emsp;尽管你的信息结构很清晰，但是如今有能力调查并从你的信息中解脱出诱因，然而，由于巨大的信息正在变成今天的口号，所以我们需要信息科学设备和进步。在一个无误的学习时间里，我们可以用这个巨大的信息量来衡量。如今，一切都在产生信息，并有能力适应它，这是一个考验。大型组织，例如Google、Facebook、Microsoft、IBM等，根据客户每天一次产生的大量信息的最终目标，制造他们自己的适应性信息科学安排。  
&emsp;&emsp;TensorFlow，是一个机器智能数据科学平台，由Google在2016年11月9日作为开源库发布。它是一个可扩展的分析平台，使数据科学家能够在可见的时间内构建具有大量数据的复杂系统，并且也使得他们能够使用需要大量数据以获得良好性能的贪婪学习方法。
## 总结
&emsp;&emsp;在本章中，我们构建了一个用于鱼识别的学习系统；我们还看到了如何在TensorFlow和Keras的帮助下使用几行代码来构建复杂的应用程序，比如鱼识别。这个编码示例并不是要从您这边理解的，而是要演示构建复杂系统的可见性，以及数据科学如何变成一个易于使用的工具。  
&emsp;&emsp;我们看到了你作为一个数据科学家在构建学习系统时，在日常生活中所遇到的挑战。  
&emsp;&emsp;我们还研究了构建学习系统的典型设计周期，并解释了该周期中涉及的每个组件的总体思想。  
&emsp;&emsp;最后，我们经历了不同的学习类型，有大公司和小公司每天生成的大数据，以及这些海量数据如何引起警告，以构建可伸缩的工具，从而能够分析和从这些数据中提取价值。  
&emsp;&emsp;此时，读者可能被迄今为止提到的所有信息淹没了，但是本章中解释的大部分内容将在其他章节中讨论，包括数据科学挑战和鱼类识别示例。本章的全部目的是对数据科学及其开发周期有一个全面的了解，而不需要深入理解面临的挑战和编码示例。本章中提到了编码示例，以打破数据科学领域中大多数新手的恐惧，并向他们展示如何在几行代码中完成复杂的系统，如鱼识别。  
&emsp;&emsp;下一步，我们将开始我们的举例来说旅行，通过一个例子来解决数据科学的基本概念。下一部分将主要通过准备著名的泰坦尼克号的例子来为以后的高级章节做准备。本文将介绍许多概念，包括用于回归和分类的不同学习方法、不同类型的性能错误以及最需要关注的问题，以及更多关于处理一些数据科学挑战和处理不同形式的数据样本的内容。

|小组成员 |学号  |
| --------   | -----:  |
|石亦飞 |201802210493|
|于泽   |201802210495|
|陈登科 |201802210492|
