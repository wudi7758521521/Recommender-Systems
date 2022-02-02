# -*- coding: utf-8 -*

import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

# random.seed(2020)


path=sys.path[0]

def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=20):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    # with open('../data/remap.pkl', 'rb') as f:
    #     reviews_df = pickle.load(f)
    #     # cate_list = pickle.load(f)
    #     user_count, item_count, example_count = pickle.load(f)
    reviews_df = pd.read_csv(path + '/../data/remap.csv', encoding='utf8', sep='\t', low_memory=False)
    example_count = reviews_df.shape[0]
    meta_df = pd.read_csv(path + '/../data/meta.csv', encoding='utf8', sep='\t', low_memory=False)
    item_count = len(meta_df)
    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']
    # # 保存reviews_df进行查看
    # reviews_df.to_csv("./reviews_d.csv",sep='\t',encoding='utf8',index=False,header=True)
    # print(reviews_df)

    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        # pos_list就是用户真实购买的商品， 下面针对每个购买的商品， 产生一个用户没有购买过的产品
        pos_list = hist['item_id'].tolist()

        # print(pos_list)
        # print(user_id)
        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)  # 这儿产生一个不在真实用户购买的里面的
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []  # 历史购买商品
        for i in range(1, len(pos_list)):
            # 取消hist中的类别参数 cate_list[pos_list[i-1]]
            # hist.append([pos_list[i - 1], cate_list[pos_list[i-1]]])

            hist.append([pos_list[i - 1]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:  # 最后一个的时候
                # 构建测试集
                # test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                # test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                test_data.append([user_id, hist_i, [pos_list[i]], 1])
                test_data.append([user_id, hist_i, [neg_list[i]], 0])
                # print(test_data)
            elif i == len(pos_list) - 2:  # 倒数第二个的时候
                # 构建验证集
                # val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                # val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                val_data.append([user_id, hist_i, [pos_list[i]], 1])
                # print(val_data)
                val_data.append([user_id, hist_i, [neg_list[i]], 0])
                # print(val_data)
            else:
                # 构建训练集
                # 保存正负样本
                # train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                # train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                train_data.append([user_id, hist_i, [pos_list[i]], 1])
                train_data.append([user_id, hist_i, [neg_list[i]], 0])


    # # 查看正负样本比例
    # sns.countplot("label", data=train)
    # plt.title("train_data")
    # plt.show()

    # sns.countplot("label", data=test)
    # plt.title("test_data")
    # plt.show()
    # plt.close()

    # sns.countplot("label", data=val)
    # plt.title("valid_data")
    # plt.show()


    # print('***********************')
    # print('训练数据集长相train_data')
    # print(train_data)

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    behavior_list = ['item_id']  # , 'cate_id'

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['user_id','hist', 'target_item', 'label'])
    # print('dataframe构建完毕的train长相')
    # print(train)
    # train.to_csv('./train.csv',sep='\t',encoding='utf8')
    val = pd.DataFrame(val_data, columns=['user_id','hist', 'target_item', 'label'])
    # val.to_csv('./val.csv', sep='\t', encoding='utf8')
    test = pd.DataFrame(test_data, columns=['user_id','hist', 'target_item', 'label'])
    # test.to_csv('./test.csv', sep='\t', encoding='utf8')

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0] * len(train)),
               np.array([0] * len(train)),pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    # train_Xpad = pad_sequences(train['hist'], maxlen=maxlen)
    # print('train_Xpad:')
    # print(train_Xpad)
    # print('train_X',train_X)
    # train__y = train['label']
    # print('train__y',train__y)
    val_X = [np.array([0] * len(val)),
             np.array([0] * len(val)),
             pad_sequences(val['hist'], maxlen=maxlen),
             np.array(val['target_item'].tolist())
             ]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    # print(type(test_X))
    # print(test_X)

    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)


if __name__ == '__main__':
    create_amazon_electronic_dataset(path + '../data/remap.csv')
