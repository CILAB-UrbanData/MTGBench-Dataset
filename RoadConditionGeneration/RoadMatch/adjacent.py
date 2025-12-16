import geopandas as gpd
import networkx as nx
import pickle

# 1. 读取shp文件
gdf = gpd.read_file("/mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/RoadMatch/roads_chengdu.shp")

# 只保留 LINESTRING 类型
gdf = gdf[gdf.geometry.type == "LineString"].reset_index(drop=True)

# 2. 建立图
G = nx.Graph()

# 每条道路作为节点
for idx, row in gdf.iterrows():
    osm_id = row["osm_id"]
    G.add_node(osm_id)

# 3. 道路相交则连边
# geopandas 的 sindex 可以加速相交查询
sindex = gdf.sindex
for idx, row in gdf.iterrows():
    osm_id = row["osm_id"]
    geom = row.geometry
    
    # 查找可能相交的候选
    possible_matches_index = list(sindex.intersection(geom.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    
    for _, candidate in possible_matches.iterrows():
        if osm_id == candidate["osm_id"]:
            continue
        if geom.intersects(candidate.geometry):
            G.add_edge(osm_id, candidate["osm_id"])

# 4. 转换为邻接矩阵 (稀疏矩阵)
A = nx.to_scipy_sparse_array(G, dtype=int)

# 5. 存储为 .pkl
with open("/mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/RoadMatch/processed/road_adj_matrix.pkl", "wb") as f:
    pickle.dump(A, f)

print("邻接矩阵大小:", A.shape)
print("已保存到 road_adj_matrix.pkl")
