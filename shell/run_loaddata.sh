#!/bin/bash
source /etc/profile

hive -e "select * from tmp.user_behave_DIN;" >>/home/rec/azkaban-jobs/Ankai/DIN_model/data/user_bhv.csv

hive -e "select * from tmp.video_info_DIN;" >>/home/rec/azkaban-jobs/Ankai/DIN_model/data/video_info.csv

# 加载召回表
hive -e "select * from tmp.rec_video_din_category_pred;" >>/home/rec/azkaban-jobs/Ankai/DIN_model/data/target_category.csv




# 由于一些品类video数量较少，导致某些桶中不含video，因此分桶方法尝试失败
#today_H=$(date +%H)
#hive -e "select * from tmp.rec_video_din_category_pred_bucket where bucket='${today_H}';" >> /home/rec/azkaban-jobs/Ankai/DIN_model/data/target_category_pred_bucket.csv
