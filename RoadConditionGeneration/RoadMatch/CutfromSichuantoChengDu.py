import geopandas as gpd
from shapely.geometry import box

# 读取四川道路
roads = gpd.read_file(r"data/GaiyaData/raw/2016年成都滴滴轨迹数据/road/gis_osm_roads_free_1.shp").to_crs(epsg=4326)

# 成都范围 (这里取一个粗框，可以自行缩小)
minx, miny, maxx, maxy = 102.9, 30.0, 104.9, 31.5
chengdu_bbox = box(minx, miny, maxx, maxy)

# 裁剪
roads_cd = roads[roads.intersects(chengdu_bbox)]

# 保存为新的shp
roads_cd.to_file("roads_chengdu.shp")

print(f"裁剪完成，共 {len(roads_cd)} 条道路，已保存到 roads_chengdu.shp")