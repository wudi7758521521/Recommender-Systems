-- dt跟shell脚本相关联

-- user行为表，5天的训练数据(临时表）
DROP TABLE IF EXISTS tmp.user_behave_DIN;
CREATE TABLE tmp.user_behave_DIN AS
select * from
(SELECT video_id,visit_time,distinct_id as user_id,dt from scot_dws.dws_svideo_bhv
WHERE (dt between '2022-01-23' and '2022-01-23') and (video_id between 0 and 99999999)
and visit_time like '20%')a WHERE user_id REGEXP '^[0-9]*$';

-- video信息表，将video_id映射为数值(临时表）
--DROP TABLE IF EXISTS tmp.video_info_DIN;
--CREATE TABLE tmp.video_info_DIN AS
--SELECT DISTINCT id from scot_dwd.dwd_svideo_info
--WHERE dt='${hivevar:enddate}' and id between 0 and 99999999 and
--names_str is NOT FALSE and names_str is NOT NULL;

DROP TABLE IF EXISTS tmp.video_info_DIN;
CREATE TABLE tmp.video_info_DIN AS
select a.id as id,h.label_name label,a.dt as dt from (
select * from scot_dwd.dwd_svideo_info  where dt='${hivevar:enddate}' and names_str IS NOT FALSE
) a
lateral  view  explode(split(names_str,','))h as label_name

---- viedo由品类关联的召回表
--DROP TABLE IF EXISTS tmp.rec_video_din_category_pred;
--CREATE TABLE tmp.rec_video_din_category_pred AS
--select a.id as video_id1,a.names_str as category,b.id as video_id2,b.dt as dt  from
--(
--
--select a.id,a.names_str,a.dt,h.label_name label_name2 from (
--select * from scot_dwd.dwd_svideo_info  where dt='${hivevar:enddate}' and names_str IS NOT FALSE
--) a
--lateral  view  explode(split(names_str,','))h as label_name
--
--) a
--INNER join
--(
--
--select * from (
--SELECT id ,names_str, dt,label_name2 ,row_number() over(partition by label_name2 order by rand() desc) rank
--from (select b.id,b.names_str,b.dt,ss.label_name label_name2 from (
--select * from scot_dwd.dwd_svideo_info  where dt='${hivevar:enddate}' and names_str is NOT FALSE
--) b
--lateral view explode(split(names_str,','))ss as label_name)b)b where b.rank<51
--
--) b
--on a.label_name2 = b.label_name2;
