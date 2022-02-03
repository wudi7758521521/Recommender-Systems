from dpp import *
import time
from multiprocessing import Pool



item_size = 1000
feature_dimension = 500
max_length = 500

scores = np.exp(0.01 * np.random.randn(item_size) + 0.2)
feature_vectors = np.random.randn(item_size, feature_dimension)

feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
similarities = np.dot(feature_vectors, feature_vectors.T)
kernel_matrix = scores.reshape((item_size, 1)) * similarities * scores.reshape((1, item_size))

print('kernel matrix generated!')
# print('排序之前',feature_vectors.shape)

# print('排序之前',feature_vectors)
t = time.time()
result = dpp(kernel_matrix, max_length)
print('algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
# print(kernel_matrix.shape)
# print(kernel_matrix)





window_size = 8
t = time.time()
result_sw = dpp_sw(kernel_matrix, window_size, max_length)
print('sw algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))

# print(result)
# print(result_sw)
# # print(dpp_sw(kernel_matrix, 8,max_length, epsilon=1E-10))