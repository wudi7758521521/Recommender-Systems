# 项目背景

在推荐场景下，给用户推荐相关商品的同时，需要保证推荐结果的多样性，即排序结果存在着相关性与多样性的权衡。可以通过Maximal Marginal Relevance (MMR) 算法减少排序结果的冗余，同时保证结果的相关性。

在购物车-有货猜你喜欢楼层，采用基于Item-based CF协同过滤的矩阵分解算法进行Top-N商品推荐，再通过MMR重排序算法，可对推荐结果实现多样性重排序。

# MRR

## MRR简介

先用某种推荐模型如协同中的item或是矩阵分解等挖掘出top-N商品，然后将item分数以及item相似矩阵输入到MMR中再进行多样性的调整进行重排序及top推荐

## MRR原理

MMR算法将排序结果的相关性与多样性综合于下列公式中：

<img src="https://pic1.zhimg.com/v2-72abd551d9bcd88082d549e89878ef48_r.jpg" alt="preview" style="zoom:50%;" />

其中，Q : 用户;  d : 推荐结果集合;  C : R 中已被选中集合;  λ : 权重系数，调节推荐结果相关性与多样性( ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 越大，推荐结果越相关； ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 越小，推荐结果多样性越高)

sim(Q, D<sub>i</sub>)表示user和item的scores相关性得分(Sim1(Q, D<sub>i</sub>)可以是协同过滤算法出来的相关度)，sim(D<sub>i</sub>, D<sub>j</sub>)表示当前item i与已经成为候选推荐商品的item j的相似度(sim(D<sub>i</sub>, D<sub>j</sub>)可以是余弦相似度)。

这个公式的含义是每次从未选取列表中选择一个使得和（可以是用户也可以是）的相关性与与已选择列表集合的相关性的差值最大，这就同时考虑了最终结果集合的相关性和多样性，通过λ因子来平衡。



MMR算法需要输入推荐商品的相关分数  及  商品间相似度矩阵。

利用**协同过滤算法矩阵分解**的Item Factors作为商品的向量表征，计算余弦相似度，并将相似度进行线性映射到[0,1]区间，得到商品相似度矩阵。用户 ![[公式]](https://www.zhihu.com/equation?tex=u) 对商品 ![[公式]](https://www.zhihu.com/equation?tex=i) 的相关分数计算如下式：

<img src="C:\Users\ankai2\AppData\Roaming\Typora\typora-user-images\image-20211029163125291.png" alt="image-20211029163125291" style="zoom:70%;" />

## MRR代码

~~~ python
import numpy as np
import time

def MMR(itemScoreDict, similarityMatrix, lambdaConstant=0.5, topN=20):
    s = []
    r = list(itemScoreDict.keys())
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

测试代码

~~~python
if __name__ == '__main__':
    item_size = 11
    feature_dimension = 20
    # itemScoreDict = np.arange(item_size)
    # itemScoreDict = dict(enumerate(itemScoreDict))
    #
    # print(itemScoreDict)
    itemScoreDict = {1: 0.89000000000000001,2: 0.90000000000000002, 
                     3: 0.91000000000000003, 4: 0.92000000000000004, 
                     5: 0.93000000000000005,6: 0.94000000000000006, 
                     7: 0.95000000000000007, 8: 0.95999999999999996, 
                     9: 0.96999999999999997,10: 0.97999999999999998}	
    feature_vectors = np.random.randn(item_size, feature_dimension)  # 商品特征embeding
    similarityMatrix = np.dot(feature_vectors, feature_vectors.T)  # item的相似度矩阵
    t = time.time()
    result = MMR(itemScoreDict, similarityMatrix, lambdaConstant=0.5, topN=10)
    print('MMR running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
    print(result)
~~~





**矩阵分解**

缺失的m_ij在p或者q中对应位置不会参与迭代更新。因为你没法优化那个缺失位置的式子。

最终的目标是靠哪些不缺失值的位置，迭代优化出最优的P,Q，然后那些缺失的位置的值就认为是等于是q<sup>T</sup><sub>j</sub>p<sub>i</sub>



矩阵分解：本质是根据已经购买产品的组合，找到背后的因素（文中的k维隐含特征），然后使用隐含特征，来而推测出未购买产品的分数。