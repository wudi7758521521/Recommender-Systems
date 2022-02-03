import time

import numpy as np
import math


class DPPModel(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs['item_count']
        self.item_embed_size = kwargs['item_embed_size']
        self.max_iter = kwargs['max_iter']
        self.epsilon = kwargs['epsilon']

    def build_kernel_matrix(self):
        rank_score = np.random.random(size=(self.item_count))  # 用户和每个item的相关性
        item_embedding = np.random.randn(self.item_count, self.item_embed_size)  # item的embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵
        self.kernel_matrix = rank_score.reshape((self.item_count, 1)) \
                             * sim_matrix * rank_score.reshape((1, self.item_count))

    def dpp(self):
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))
        j = np.argmax(d)
        Yg = [j]
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                if iter == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)
            if d[j] < self.epsilon:
                break
            Yg.append(j)
            iter += 1
        return Yg


if __name__ == "__main__":
    kwargs = {
        'item_count': 100,
        'item_embed_size': 100,
        'max_iter': 500,
        'epsilon': 0.01
    }
    start = time.time()
    dpp_model = DPPModel(**kwargs)
    dpp_model.build_kernel_matrix()
    dpp_model.dpp()
    end = time.time() - start
    print(dpp_model.dpp())
    print(end)
