1. match.py   输入：roads.shp包含路段信息(RoadMatch/roads_chengdu.shp)  traj.csv需要匹配坐标点的轨迹文件   输出:traj中每个采样点的经纬度匹配到的edge_id和road_list(与真实路段对应的编号) (RoadMatch/processed/matched_chengdu_fast_with_roadlist.csv)
2. depandcal_time.py  输入：上一步的输出 输出：matched_chengdu_fast_dedup_with_dwell.csv（去重并且改轨迹下该edge_id的停留总时长）
3. flow_gen.py从上面输出还原出flow数据

1. adjacent.py 输入：roads_chengdu.shp 输出：路网的邻接矩阵