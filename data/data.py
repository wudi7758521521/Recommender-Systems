# 两种方法都能打开
import pickle
import numpy as np
import pandas as pd
'''
pickle方式打开
meta中的item数量一定小于reviews，因为meta中只保留在reviews出现过的item
'''


# # 225094个item
# f = open('./meta.pkl','rb')
# data = pickle.load(f)
# data.to_csv('./meta1.csv')
#
# print(data)

# # 2598209 个用户观看行为
# f = open('./reviews.pkl','rb')
# data = pickle.load(f)
# # 禁用科学计算法
# pd.set_option('display.float_format',lambda x : '%.2f' % x)
# data.to_csv('./reviews1.csv')
# print(data)

# # # #2548790 条映射后的数据
# f = open('./remap.pkl','rb')
# data = pickle.load(f)
# ## 判断有无重复值，False表示当前位置和上个位置无重复，True表示当前位置和上个位置有重复
# ## data = data['reviewerID'].duplicated()
# ##data = data.duplicated(subset='reviewerID')
# # 禁用科学计算法
# pd.set_option('display.float_format',lambda x : '%.2f' % x)
# data.to_csv('./remap.csv', sep='\t', index=0, encoding='utf8')
# print(data)




# 148379 条映射后的数据
f = open('./remap.pkl','rb')
data = pickle.load(f)
# 判断有无重复值，False表示当前位置和上个位置无重复，True表示当前位置和上个位置有重复
# data = data['reviewerID'].duplicated()
# data = data.duplicated(subset='reviewerID')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
data.to_csv('./remap.csv',sep='\t',index=0,encoding='utf8')

print(data)



# data = pd.read_csv('./video_info.csv', encoding='utf8', sep='\t', low_memory=False)
# print(data)
#
# data = pd.read_csv('./df.csv', encoding='utf8', sep='\t', low_memory=False)
# print(data)


# meta_df = pd.read_csv('../data/meta.csv', encoding='utf8', sep='\t', low_memory=False)
# # meta_df = pd.read_csv('../data/meta.csv', encoding='utf8', sep='\t')
# print(meta_df)