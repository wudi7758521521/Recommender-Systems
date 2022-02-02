2022/01/17

1. 同一user同时预测多个target，且当某品类数量较少时(>2000)，则有多少召回多少，召回多少预测多少！
2. 使用pyhive，通过pandas直接读取hive中的数据进行处理

2022/01/18

3. 构建预测数据的时候，不写死选取video_hist中最后一个video_id作为target，而是循环选取video_hist中的video_id作为target！

2022/01/19

4. 模型文件train.py每天运行一次，更新保存的模型DIN.model；预测文件target_pred.py每小时运行一次，更新预测的结果数据。

2022/01/24

5. 构建一个video质量分数汇总表，评估短视频video的质量优劣，按照video的质量分score进行排序；当DIN向user推荐的video数目较少时，使用该video质量分数表作为兜底策略，按照排序顺序向user推荐video。

2022/01/25

6. 取消构建video-video的召回表，使用video_info.csv的label标签，构建label-video的对应关系，通过target的所属label来获取同品类的video进行video的召回。避免了构建video-video品类召回表所耗费的时间

