# -*- coding: utf-8 -*

# from pyhive import hive
import pandas as pd
import numpy as np
import pickle
import random
import tensorflow as tf
from pandas.core.frame import DataFrame
import datetime
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os

# from remap import build_map


# random.seed(2020)

# 设置log输出信息，程序运行时系统打印的信息。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置为2时，屏蔽通知信息和警告信息
# 如果电脑有多个GPU，tensorflow默认全部使用。如果想只使用部分GPU，可以设置CUDA_VISIBLE_DEVICES。在调用python程序时，可以使用
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 终端执行程序时设置使用的GPU,仅设备6可见


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


def create_amazon_electronic_dataset(embed_dim=8, maxlen=20):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')

    reviews_df = pd.read_csv(path + '/../data/remap.csv', encoding='utf8', sep='\t', low_memory=False)
    reviews_df.columns = ['user_id', 'item_id', 'time']

    meta_df = pd.read_csv(path + '/../data/meta_map.csv', encoding='utf8', sep='\t', low_memory=False)
    item_count = len(meta_df)

    # asin_map, asin_key = build_map(meta_df, 'asin')
    # print(meta_df)
    # exit(0)
    # # # 3.读取target，linux改用pyhive直接读取数据,则无需下载文件
    # # conn = hive.Connection(host='10.112.183.31', port=10000, auth='KERBEROS', kerberos_service_name="hive")
    # # target = pd.read_sql("select * from tmp.rec_video_din_category_pred", conn)  # 注意注意，此处sql语句结尾出不带;号
    # target = pd.read_csv(path + '/../data/target_category.csv', sep='\t')
    #
    # target = pd.DataFrame(target)
    # ## target文件的video_id根据meta文件映射的video_id进行一一对应，确保同一个video_id映射为同一个数值
    # target['rec_video_din_category_pred.video_id1'] = target['rec_video_din_category_pred.video_id1'].map(asin_map)
    # target['rec_video_din_category_pred.video_id2'] = target['rec_video_din_category_pred.video_id2'].map(asin_map)
    # target_map = target[['rec_video_din_category_pred.video_id1', 'rec_video_din_category_pred.category',
    #                      'rec_video_din_category_pred.video_id2', 'rec_video_din_category_pred.dt']]  # 只保留四列
    #
    # # target_map.to_csv(path + '/../data/target_map.csv', sep='\t', encoding='utf8', index=False)

    # 3.读取target映射的全量video_id信息表
    # target_map = pd.read_csv(path + '/../data/video_info.csv', sep='\t')
    target_map = pd.DataFrame(meta_df)
    # target_map.columns = ['id', 'label', 'dt']

    # target_map = target_map.drop_duplicates(subset=['id'])  # 去重
    # print(target_map.dtypes)
    # exit(0)
    # 将召回表映射成字典格式
    # key表示video1(原target)，value表示同类别召回的viedo_id(需要预测的target)
    target_recall_map_dict = target_map.groupby('label')['asin'].apply(
        list).to_dict()  # 对于同一个key对应多个value，则把同一key的value构成一个list
    # print(target_recall_map_dict)
    # exit(0)

    # # 将dict类型存为json格式保存
    # import json
    #
    # # jsObj = json.dumps(revi_map)
    # jsObj1 = json.dumps(target_recall_map_dict, ensure_ascii=False)
    # # fileObject = open('../data/revi_map.json', 'w')
    # fileObject1 = open('../data/target_recall_map_dict.json', 'w', encoding='utf-8')
    # # fileObject.write(jsObj)
    # fileObject1.write(jsObj1)
    # # fileObject.close()
    # fileObject1.close()
    # exit(0)

    # target_list = []
    # user_list = []
    # videohist_list = []
    test_data = []
    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        # hist_list是用户观看过的历史video_id,包括最后一个video_id
        hist_list = hist['item_id'].tolist()
        # print('浏览历史',hist_list)

        hist = []  # 历史购买商品
        for i in range(1, len(hist_list)):
            hist.append([hist_list[i - 1]])
            hist_i = hist.copy()
            test_data.append([user_id, hist_i, hist_list[i]])

        # 此处代码写死target的位置，使用hist中的最后一个video_id作为target
        # if len(hist_list) > 1:  # 当hist_list只有一个video时，中断循环
        #
        #     target_video = hist_list[len(hist_list) - 1]  # 以hist_list中的最后一个video_id作为target
        #     target_list.append(target_video)  # 1.存储target_video的列表
        #
        #     user_list.append(user_id)  # 2.存储user_id的列表
        #
        #     video_hist = hist_list[0:len(hist_list) - 1]
        #     videohist_list.append(video_hist)  # 3.存储video_hist的列表
        #
        #     # exit(0)
        #
        #     # test_data.append([user_id, video_hist, target_video])
        # else:
        #     continue
    # print(test_data)
    # print(len(test_data))
    test_df = pd.DataFrame(test_data, columns=['user_id', 'hist', 'target_item'])
    target_list = test_df['target_item'].values.tolist()
    user_list = test_df['user_id'].values.tolist()
    videohist_list = test_df['hist'].values.tolist()
    # test_df.to_csv('../data/test_df.csv', sep='\t')
    # print(target_list)
    # print(len(target_list))
    # exit(0)

    # #召回数据1
    target_list_pre = []
    k_list = []
    for k in target_list:
        # print(k)
        a = meta_df.loc[meta_df['asin'] == k]['label'].tolist()
        label = a[0]
        # print(label)
        # print(meta_df.index[meta_df['asin'] == k].tolist())
        # meta_df.query('asin==k')
        # print(meta_df)
        # exit(0)
        if label in target_recall_map_dict:
            target_recall_map_value = target_recall_map_dict[label]  # dict由key获得value
            value_len = len(target_recall_map_value)
            # print(value_len)
            # exit(0)
            if value_len < 20:
                target = random.sample(target_recall_map_value, value_len)  # 从target_list中随机取1个
                target_list_pre.append(target)
            else:
                target = random.sample(target_recall_map_value, 20)  # 从target_list中随机取1个
                target_list_pre.append(target)
        else:
            k_list.append(k)
            print('召回表中有不存在的video_id，出错！')

    # print(target_list_pre[:10])
    # # # print(len(target_list_pre))
    # exit(0)

    user_id_pred = []
    videohist_pred = []
    for i in range(len(target_list_pre)):
        val_len = len(target_list_pre[i])
        d = [user_list[i]] * val_len
        user_id_pred.extend(d)

        e = [videohist_list[i]] * val_len
        videohist_pred.extend(e)

    target_list_pred = []
    [target_list_pred.extend(i) for i in target_list_pre]  # 去掉列表嵌套

    # create dataframe
    pred_dataframe = {'user_id': user_id_pred, 'hist': videohist_pred, 'target_item': target_list_pred}
    df_pred = pd.DataFrame(pred_dataframe)

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    behavior_list = ['item_id']  # , 'cate_id'

    # if no dense or sparse features, can fill with 0
    print('==================Padding==================')

    test_X1 = {'input_1': tf.reshape(np.array([0] * len(df_pred), dtype=np.int32), [-1, 1]),
               # np.array([0] * len(df_pred), dtype=np.int64)
               'input_2': tf.reshape(np.array([0] * len(df_pred), dtype=np.int32), [-1, 1]),
               'input_3': tf.reshape(np.array(pad_sequences(df_pred['hist'], maxlen=maxlen), dtype=np.int32),
                                     [-1, 20, 1]),
               'input_4': tf.reshape(np.array(target_list_pred, dtype=np.int32), [-1, 1])
               }
    print(test_X1)
    # exit(0)
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, test_X1, user_id_pred, videohist_pred, target_list_pred


if __name__ == '__main__':
    path = sys.path[0]
    embed_dim = 8
    maxlen = 20

    feature_columns, behavior_list, test_X1, user_id_pred, videohist_pred, target_list_pred = create_amazon_electronic_dataset(
        embed_dim, maxlen)

    # 模型加载预测
    loaded = tf.keras.models.load_model(path + '/../DIN.model')  # loaded = tf.saved_model.load(path)
    # loaded.summary()
    # exit(0)
    infer_model = loaded.signatures['serving_default']

    # pre_result1输出结果分数score为字典类型
    pre_result1 = infer_model(**test_X1)
    print(pre_result1)
    # exit(0)

    # 将字典转换为列表
    result_list1 = pre_result1['output_1'].numpy().tolist()  # 将tensor格式转换为numpy格式,最后转化为列表

    result_score_list1 = []
    [result_score_list1.extend(i) for i in result_list1]  # 去掉列表嵌套

    # 创建dataframe类型的hive格式表,用于存储进hive，通过list形式写进dataframe
    dataframe = {'user_id': user_id_pred, 'video_hist': videohist_pred, 'target_pred': target_list_pred,
                 'score': result_score_list1, }
    df = pd.DataFrame(dataframe)

    df = df[['user_id', 'video_hist', 'target_pred', 'score']]
    df[['target_pred', 'score']] = df[['target_pred', 'score']].astype(str)
    df.insert(4, 'dt', pd.datetime.now().strftime("%Y-%m-%d %H:00:00"))
    df.to_csv(path + '/../data/out_result.csv', sep='\t', mode='a', index=False, encoding='utf8')
    print(df)
