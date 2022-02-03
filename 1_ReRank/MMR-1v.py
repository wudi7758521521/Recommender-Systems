import numpy as np
import time


def MMR(itemScoreDict, similarityMatrix, lambdaConstant=0.9, topN=20):
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


if __name__ == '__main__':
    item_size = 20
    feature_dimension = 30
    # itemScoreDict = np.arange(item_size)
    # itemScoreDict = dict(enumerate(itemScoreDict))
    # print(itemScoreDict)
    itemScoreDict = {1: 0.89000000000000001,
                     2: 0.90000000000000002, 3: 0.91000000000000003, 4: 0.92000000000000004, 5: 0.93000000000000005,
                     6: 0.94000000000000006, 7: 0.95000000000000007, 8: 0.95999999999999996, 9: 0.96999999999999997,
                     10: 0.97999999999999998}
    feature_vectors = np.random.randn(item_size, feature_dimension)  # 商品特征embeding
    similarityMatrix = np.dot(feature_vectors, feature_vectors.T)  # item的相似度矩阵
    t = time.time()
    result = MMR(itemScoreDict, similarityMatrix, lambdaConstant=0.5, topN=20)
    print('MMR running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
    print(result)
