## 背景介绍

推荐系统的多样性能够反应一个推荐列表中内容不相似的程度。通过推荐多样性更高的内容，既能够给用户更多的机会去发现新内容，也能够让推荐系统更容易发现用户潜在的兴趣。但推荐系统的多样性高，就意味着精确度被降低，因此需要设计算法/策略去权衡推荐结果的相关性和多样性。

重排序问题可以看成是一个子集选择问题，目标是从原始数据集合中选择具有高质量但又多样化的的子集。

## 多样性模型实现

### MMR算法python实现

~~~python
import numpy as np
import time

def MMR(itemScoreDict, similarityMatrix, lambdaConstant=0.5, topN=20):
    s, r = [], list(itemScoreDict.keys())
    while len(r) > 0:
        score = 0
        selectOne = None
        # 遍历所有剩余项
        for i in r:
            firstPart = itemScoreDict[i]
            # 计算候选项与"已选项目"集合的最大相似度
            secondPart = 0
            for j in s:
                sim2 = similarityMatrix[i][j]
                if sim2 > secondPart:
                    secondPart = sim2
            equationScore = lambdaConstant * (firstPart - (1 - lambdaConstant) * secondPart)
            if equationScore > score:
                score = equationScore
                selectOne = i
        if selectOne == None:
            selectOne = i
        # 添加新的候选项到结果集r，同时从s中删除
        r.remove(selectOne)
        s.append(selectOne)
    return (s, s[:topN])[topN > len(s)]
~~~

### DPP算法python实现

~~~python
import numpy as np
import math

def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items
~~~

测试代码 （**DPP模型的输入 1.商品的精排分数；2.任意两个商品间的pairwise距离。通过商品精排分数和任意两商品间距离(相似度)计算kernel核矩阵**）

~~~python
from dpp import *
import time

item_size = 200
feature_dimension = 500
max_length = 500
 
scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)  # 精排商品的分数
feature_vectors = np.random.randn(item_size, feature_dimension)  # 商品的特征向量

feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
similarities = np.dot(feature_vectors, feature_vectors.T)  # 相似度矩阵，任意两商品间的距离
kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))  # 核矩阵的计算方式

print('kernel matrix generated!')

t = time.time()
result = dpp(kernel_matrix, max_length)
print('algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
~~~

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211026171824694.png" alt="image-20211026171824694" style="zoom:80%;" />



DPP算法流程

<img src="https://upload-images.jianshu.io/upload_images/24597177-06fd1ed8b787bcf8" alt="img" style="zoom:80%;" />

**测试数据**

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211026171937221.png" alt="image-20211026171937221" style="zoom:80%;" />

## 多样性策略

打散：根据**类目、品牌**进行打散，避免品牌、品类过于集中

穿插：使用规则，插入**热门item，相似item**，提高用户体验

## 多样性评判公式：

公式(1)，针对单个用户，**i,j** 是推荐列表中的item，**|R(u)|**是推荐给用户u的item个数 (分母表示n个样本中pair对的个数）; **sim(i,j)**是计算项目之间相似性的方法，这里采用余弦相似度。一般Diversity值越大，推荐列表多样性效果越好。

公式(2)，针对整个推荐用户，分别计算每个用户的Diversity值，然后求**平均**。

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211021145512630.png" alt="image-20211021145512630" style="zoom:70%;" />

### 存在问题:

上述评测方法只考虑了两个item之间的差别，没有考虑多个item之间的整体差别。

eg：第1组3个item，A与B点积为1，C与A、B的点积都为0，通过上述计算这3个item的多样性为(0+1+1)/3=2/3;    第2组3个item，A、B、C之间的点积都为0.5，多样性为(0.5+0.5+0.5)/3=0.5; 单从指标上来看，第1组的多样性要高于第二组。但实际第2组多样性更好，因为第1组item中A与B完全一样，第2组3个item两两之间的差别比较均匀。

### 更好的方法：

**在整个子集的特征空间中定义多样性**

核矩阵L (kernel_matrix) 是由 ![[公式]](https://www.zhihu.com/equation?tex=n) 维欧氏空间中 ![[公式]](https://www.zhihu.com/equation?tex=k) 个向量的内积组成的矩阵，可被称之为Gram矩阵。核矩阵L是一个半正定矩阵，因此L可被分解为 ![[公式]](https://www.zhihu.com/equation?tex=L%3DB%5ET+B)，具体地 ![[公式]](https://www.zhihu.com/equation?tex=B) 的每一个列向量可以构建为相关性分数( item score) ![[公式]](https://www.zhihu.com/equation?tex=r_i+%5Cge+0) 与归一化后的商品item特征向量 <img src="https://www.zhihu.com/equation?tex=f_i+%5Cin+R%5ED+%28%7C%7Cf_i%7C%7C_2%3D1%29" alt="[公式]" style="zoom:80%;" /> 的乘积。因此，核矩阵中的**元素**可以被写成：<img src="https://www.zhihu.com/equation?tex=L_%7Bij%7D+%3D+%5Clangle+B_i%2CB_j+%5Crangle+%3D+%5Clangle+r_i+f_i%2C+r_j+f_j+%5Crangle+%3D+r_i+r_j+%5Clangle+f_i%2C+f_j+%5Crangle" alt="[公式]" style="zoom:90%;" />。其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Clangle+f_i%2C+f_j+%5Crangle+%3D+S_%7Bij%7D) 是商品 ![[公式]](https://www.zhihu.com/equation?tex=i) 与商品 ![[公式]](https://www.zhihu.com/equation?tex=j) 之间的相似度的度量。

![[公式]](https://www.zhihu.com/equation?tex=det%28L%29%3D%5Cleft%7C+L+%5Cright%7C%3D%5Cleft%7C+Diag%5Cleft%28+r_%7Bu%7D+%5Cright%29+%5Cright%7C%5Ccdot%5Cleft%7CS+%5Cright%7C%5Ccdot%5Cleft%7C+Diag%5Cleft%28+r_%7Bu%7D+%5Cright%29+%5Cright%7C+%3D%5Cprod_%7Bi%5Cin+R_%7Bu%7D%7Dr_%7Bu%2Ci%7D%5E%7B2%7D%5Ccdot%5Cleft%7C+S+%5Cright%7C%3D%5Cprod_%7Bi%5Cin+R_%7Bu%7D%7Dr_%7Bu%2Ci%7D%5E%7B2%7D%5Ccdot+det%28S+%29)

取对数：(公式描述：假设选出的候选集用集合![R_{u}](https://private.codecogs.com/gif.latex?R_%7Bu%7D)表示(刚开始集合为空)，那![L_{R_{u}}](https://private.codecogs.com/gif.latex?L_%7BR_%7Bu%7D%7D)是![L](https://private.codecogs.com/gif.latex?L)的子矩阵，问题就变成了不断的从![L](https://private.codecogs.com/gif.latex?L)中取元素，使选出的![L_{R_{u}}](https://private.codecogs.com/gif.latex?L_%7BR_%7Bu%7D%7D)能够有最大的体积。)

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211024204420829.png" alt="image-20211024204420829" style="zoom:80%;" />

其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Csum%5Climits_%7Bi+%5Cin+R_u%7D+log%28r_%7Bu%2Ci%7D%5E2%29+) 表示用户和商品的相关性， ![[公式]](https://www.zhihu.com/equation?tex=det%28S%29) 表示商品之间的多样性。

 ![[公式]](https://www.zhihu.com/equation?tex=det%28%5Ccdot%29) 表示矩阵的行列式，![[公式]](https://www.zhihu.com/equation?tex=Diag%28r_u%29) 是对角阵（diagonal matrix），它的对角向量![[公式]](https://www.zhihu.com/equation?tex=r_u)（diagonal vector）是相关性度量向量(文章中没有对r<sub>u</sub>如何计算做任何定义) ，对角元素 ![[公式]](https://www.zhihu.com/equation?tex=r_%7Bu%2Ci%7D) 表示用户 ![[公式]](https://www.zhihu.com/equation?tex=u) 对物品 ![[公式]](https://www.zhihu.com/equation?tex=i) 的兴趣偏好程度(用户会用点击行为来表达兴趣)， ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bu%7D) 表示推荐系统向用户推荐的item集合，![[公式]](https://www.zhihu.com/equation?tex=S) 是相似度矩阵。r<sub>i</sub>表示商品得分( item score r<sub>i</sub>)



## 测试方案

A/B Test ： 在线验证推荐算法/策略的有效性

具体实施：多个方案并行测试，每个方案只设一个不同变量



## 其他多样性评价指标

**推荐多样性评价指标？**

- **列表内相似度 (intra-list similarity，ILS)**

  根据推荐列表内文档对的相似程度来度量

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211026145335312.png" alt="image-20211026145335312" style="zoom:60%;" />

针对**单用户**，一般来说相似度值越大，单个用户推荐列表多样性越差；其中，i，j为item，n为推荐列表样本数(列表长度)，sim(.)为相似性度量；对一个存在n个样本的列表而言，一共有n(n-1)/2个pair。

- **海明距离(Hamming distance，Hamm)**

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211026150041788.png" alt="image-20211026150041788" style="zoom:80%;" />

海明距离，衡量**不同用户**的推荐结果的差异性，H<sub>ij</sub>越大说明不同用户间的多样性程度越高；其中，L为推荐列表长度(item数量n)，Q<sub>ij</sub>为系统推荐给用户i和用户j两个推荐列表中相同item的数量。



 **ILMD-MRR (相似性-相关性)关系**

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211101165413506.png" alt="image-20211101165413506" style="zoom:40%;" />



## 论文中的优化方案

在每次选出一个item时，都从余下的item中选出和已经选择的item中差异性最大，相关性最明显(边际收益最大)的item，用的贪心算法，这样最终的候选集就是最优解。

该PDD方法在计算方法上做出优化。通过增量更新Cholesky因子，将计算复杂度降至O(*M*<sup>3</sup>)，运行O(*N*<sup>2</sup>*M*)的时间来返回N个items，使它在大规模实时场景中变得可用。

1. 对子矩阵 L<sub>Y</sub> 做 Cholesky  (科列斯基) 分解
2.  矩阵分解后，每次迭代使用增量更新的方式更新参数，将每次迭代的计算复杂度降低至一次方。



另外，设计了一个滑动窗口方法。假设window size为：w<N，复杂度可以减小到O(wNM)。这个使得该算法可用于在一个短的滑动窗口内保证多样性。

- M：items总数目
- N：最终返回N个items结果
- w：window size



相关性和多样性测试

dpp 算法：在排序后的m个item中选出k（k<m）个item，使这k个item的排序分尽量高，同时多样性最大化。这时不能只考虑多样性了，还要兼顾排序分。






$$
log P(R_u) \propto \theta \cdot \sum\limits_{i \in R^u} r_{u,i} + (1-\theta) \cdot log det(S_{R_u})
$$

$$
L' = Diag(exp(\alpha r_u)) \cdot S \cdot Diag(exp(\alpha r_u))
$$

$$
\alpha = \theta / (2(1-\theta))
$$

$$
log P(R_u \cup \lbrace i \rbrace) - log P(R_u) \propto \theta \cdot r_{u,i} + (1-\theta) \cdot (log det(S_{R_u \cup \lbrace i \rbrace}) - log det(S_{R_u}))
$$

注意：特征向量需要归一化，将值控制在[0,1]







## 工程优化

1. ![[公式]](https://www.zhihu.com/equation?tex=r_u)**的选择**

   使用精排得分表征user和item之间的关系，论文中通过将![[公式]](https://www.zhihu.com/equation?tex=r_u)平滑操作，使用*exp*(*α**r**<sub>u</sub>*)代替![[公式]](https://www.zhihu.com/equation?tex=r_u)(表示e<sup>x</sup>求值)，

   此时*L*=*Diag*(*α*r<sub>u</sub>)⋅*S*⋅*Diag*(*αr*<sub>u</sub>)，其中*α*=*θ*/2(1−*θ*)，*θ*∈[0, 1] 作为一个超参数，可以手动权衡u-i相关性及i-i相似度。 **超参数 *θ* 越大（对应的α也越大)，越偏向于相关性**。

   **超参数*θ*测试**

   当设定

   ```python
   item_size = 10
   feature_dimension = 20
   scores = np.exp(0.01*np.array([0.99, 0.98, 0.88, 0.78, 0.77,0.73,0.72,0.65,0.59,0.54]) + 0.2)  # 精排后分数固定，超参数θ可以权衡u-i相关性及i-i相似度,超参数θ越大（对应的α也越大)，越偏向于相关性。
   feature_vectors.shape=(10, 20)  # feature_vectors 数值固定，shape为10行5列
   ```

   再次声明重排目的：使item的精排scores分尽量高，同时多样性最大化。

   输出结果为item精排得分对应的item下标

   ```python
   当α=0.01时，[0, 1, 2, 8, 5, 7, 3, 9, 4, 6]
   当α=0.11时，[0, 1, 2, 5, 8, 7, 3, 9, 4, 6]
   当α=0.21时，[0, 1, 2, 5, 7, 8, 3, 9, 4, 6]
   当α=0.31时，[0, 1, 2, 5, 7, 3, 8, 9, 4, 6]
   当α=0.41时，[0, 1, 2, 5, 7, 3, 8, 4, 9, 6]
   当α=0.51时，[0, 1, 2, 5, 7, 3, 8, 4, 9, 6]
   当α=0.61时，[0, 1, 2, 5, 7, 3, 6, 4, 8, 9]
   当α=0.71时，[0, 1, 2, 5, 4, 3, 6, 8, 9, 7]
   当α=0.81时，[0, 1, 2, 5, 4, 3, 6, 8, 9, 7]
   当α=0.91时，[0, 1, 2, 5, 4, 3, 6, 8, 9, 7]
   当α=1   时，[0, 1, 2, 5, 3, 4, 6, 8, 9, 7]
   当α=2   时，[0, 1, 2, 4, 5, 3, 6, 7, 8, 9]
   当α=4.5 时，[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   ```

​		 

2 .商品的向量表示feature_vectors必须要做归一化（l<sub>2</sub> normalize）

DPP用在推荐系统中时，需要保证任意两个商品的相似度在0到1之间，而向量内积的范围在 [-1, 1] ，极端情况下 ![[公式]](https://www.zhihu.com/equation?tex=f_i%3D-f_j) ,这种情况下相似度为-1，子矩阵的行列式为0，无法进行log计算。

为了避免这种情况发生，需要对表示向量做如下处理： ![[公式]](https://www.zhihu.com/equation?tex=f_i%27%3D%5B1++f_i%5D%2F%5Csqrt%7B2%7D) ，从而

![[公式]](https://www.zhihu.com/equation?tex=S_%7Bi%2Cj%7D%3D%5Cfrac%7B1%2B%3Cf_i%2Cf_j%3E%7D%7B2%7D+%5Cin+%5B0%2C+1%5D)

~~~python
# 构建item的feature_vectors
feature_vectors = np.random.randn(item_size, feature_dimension)
# 将item的feature_vectors归一化
feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
~~~



