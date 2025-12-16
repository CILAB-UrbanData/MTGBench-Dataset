#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
map_match_batch_with_roadlist.py

一次性读取同一文件夹下多个 CSV 轨迹文件，逐文件匹配并输出：
 - 每个原 CSV 对应的 matched_{basename}.csv
 - 一份 roads_edges.csv（road_list 映射）

默认并行策略：文件逐个处理；每个文件内部对订单(group by 订单ID)可并行 (cfg.n_workers)。
在 Windows 上为安全起见，若未设置 ALLOW_MP_WINDOWS=1 则 n_workers 将被强制为 1。
"""

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path
import multiprocessing as mp

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------------------
# 配置 dataclass
# -------------------------------
@dataclass
class MMConfig:
    roads_path: str
    input_folder: str
    output_folder: str = "matched_outputs"
    output_roadlist: str = "roads_edges.csv"

    # 候选生成参数
    search_radius_m: float = 80.0
    k_candidates: int = 3

    # HMM 参数
    sigma_z: float = 12.0
    beta: float = 100.0
    v_max: float = 30.0

    # 裁剪 buffer（米）
    clip_buffer_m: float = 2000.0

    # 若需强制投影 EPSG（例如固定投影），可填 epsg 编号
    force_epsg: int = None

    # 并行 worker 数（对每个文件内部订单并行）
    n_workers: int = None  # None 表示自动选择（Unix: cpu_count()-1；Windows: 1 unless ALLOW_MP_WINDOWS=1）


# -------------------------------
# 常用工具
# -------------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def utm_for_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    crs = CRS.from_dict({"proj": "utm", "zone": zone, "south": south})
    return crs


# -------------------------------
# Geometry / Graph helpers (fast implementations)
# -------------------------------
def explode_lines(roads_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if (roads_proj.geometry.geom_type == "MultiLineString").any():
            roads_proj = roads_proj.explode(index_parts=False).reset_index(drop=True)
    except Exception:
        try:
            roads_proj = roads_proj.explode().reset_index(drop=True)
        except Exception:
            pass
    return roads_proj


def build_road_graph_fast(roads_proj: gpd.GeoDataFrame):
    roads_proj = roads_proj.reset_index(drop=True).copy()
    if "edge_id" not in roads_proj.columns:
        roads_proj = roads_proj.reset_index(drop=False).rename(columns={"index": "edge_id"})
    roads_proj = explode_lines(roads_proj)

    node_index_map = {}
    node_coord = {}
    next_node_id = 0
    G = nx.DiGraph()

    def get_node_id(coord):
        nonlocal next_node_id
        key = (float(coord[0]), float(coord[1]))
        nid = node_index_map.get(key)
        if nid is None:
            nid = next_node_id
            node_index_map[key] = nid
            node_coord[nid] = key
            next_node_id += 1
        return nid

    for idx, row in roads_proj.iterrows():
        geom: LineString = row.geometry
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        u = get_node_id(coords[0]); v = get_node_id(coords[-1])
        length = float(geom.length)
        G.add_edge(u, v, length=length, geometry=geom, edge_id=row["edge_id"])
        G.add_edge(v, u, length=length, geometry=LineString(coords[::-1]), edge_id=row["edge_id"])
        roads_proj.at[idx, "_start"] = u
        roads_proj.at[idx, "_end"] = v
        roads_proj.at[idx, "_length"] = length

    keep_cols = [c for c in ["edge_id", "geometry", "_start", "_end", "_length"] if c in roads_proj.columns]
    roads_proj = roads_proj[keep_cols].reset_index(drop=True)
    return G, node_coord, roads_proj, node_index_map


def generate_candidates_fast(point_m: Point,
                             roads_proj: gpd.GeoDataFrame,
                             sindex,
                             search_radius_m: float,
                             k_candidates: int) -> List[Dict[str, Any]]:
    buffer_geom = point_m.buffer(search_radius_m)
    try:
        idxs = list(sindex.intersection(buffer_geom.bounds))
    except Exception:
        idxs = list(roads_proj.index)
    if not idxs:
        return []
    sub = roads_proj.iloc[idxs]
    try:
        sub = sub[sub.intersects(buffer_geom)]
    except Exception:
        pass
    if sub.empty:
        sub = roads_proj.iloc[idxs]

    cand = []
    for ridx, r in sub.iterrows():
        ls: LineString = r.geometry
        try:
            proj_dist = ls.project(point_m)
            snapped = ls.interpolate(proj_dist)
            dist = float(snapped.distance(point_m))
        except Exception:
            continue
        cand.append({
            "edge_id": r.edge_id if "edge_id" in r.index else r.get("edge_id", ridx),
            "geom": ls,
            "snapped_x": float(snapped.x),
            "snapped_y": float(snapped.y),
            "dist_m": dist,
            "_start": int(r._start),
            "_end": int(r._end),
            "proj_to_start": float(proj_dist),
            "seg_length": float(r._length)
        })
    if not cand:
        return []
    return sorted(cand, key=lambda x: x["dist_m"])[:k_candidates]


class SPCache:
    def __init__(self, G):
        self.G = G
        self.cache = {}

    def get_dist(self, source, target):
        if source not in self.cache:
            try:
                self.cache[source] = nx.single_source_dijkstra_path_length(self.G, source, weight="length")
            except Exception:
                self.cache[source] = {}
        return self.cache[source].get(target, np.inf)


def shortest_path_len_between_snaps_fast(spcache: SPCache,
                                         a_start, a_end, a_proj_start, a_proj_end,
                                         b_start, b_end, b_proj_start, b_proj_end,
                                         a_seglen, b_seglen):
    a_to_start = a_proj_start
    a_to_end = a_seglen - a_proj_start
    b_from_start = b_proj_start
    b_from_end = b_seglen - b_proj_start
    combos = [
        (a_start, b_start, a_to_start, b_from_start),
        (a_start, b_end, a_to_start, b_from_end),
        (a_end, b_start, a_to_end, b_from_start),
        (a_end, b_end, a_to_end, b_from_end),
    ]
    best = np.inf
    for a_node, b_node, a_seg, b_seg in combos:
        sp = spcache.get_dist(a_node, b_node)
        total = a_seg + sp + b_seg
        if total < best:
            best = total
    return float(best)


# emission / transition
def emission_log_prob(dist_m: float, sigma: float) -> float:
    return - (dist_m ** 2) / (2.0 * sigma ** 2)


def transition_log_prob(sp_len_m: float, gc_m: float, dt_s: float, beta: float, v_max: float) -> float:
    if np.isinf(sp_len_m):
        return -1e12
    lp = - abs(sp_len_m - gc_m) / max(beta, 1e-6)
    if dt_s > 0:
        v = sp_len_m / dt_s
        if v > v_max:
            lp += - (v - v_max) / v_max * 5.0
    return lp


# -------------------------------
# Viterbi 匹配函数（单订单匹配）——和之前相同的 fast 版
# -------------------------------
# 全局路网变量（供 worker 进程使用）
_ROADS_GLOBAL = None


def viterbi_match_fast(one_traj_df: pd.DataFrame,
                       roads_ll: gpd.GeoDataFrame = None,
                       cfg: MMConfig = None) -> pd.DataFrame:
    global _ROADS_GLOBAL
    if roads_ll is None:
        roads_ll = _ROADS_GLOBAL
    assert cfg is not None
    # 构造经纬点 gdf
    gps_gdf_ll = gpd.GeoDataFrame(
        one_traj_df.copy(),
        geometry=gpd.points_from_xy(one_traj_df["轨迹点经度"], one_traj_df["轨迹点纬度"]),
        crs="EPSG:4326"
    )
    # 粗裁剪
    bbox = gps_gdf_ll.total_bounds
    bbox_poly = gpd.GeoSeries([Point(bbox[0], bbox[1]),
                               Point(bbox[2], bbox[1]),
                               Point(bbox[2], bbox[3]),
                               Point(bbox[0], bbox[3])], crs="EPSG:4326").unary_union.envelope
    try:
        roads_clip_ll = roads_ll.clip(bbox_poly.buffer(0.05))
    except Exception:
        roads_clip_ll = roads_ll.copy()
    if roads_clip_ll.empty:
        roads_clip_ll = roads_ll.copy()

    # 投影到米制
    if cfg.force_epsg is None:
        cx, cy = gps_gdf_ll.unary_union.centroid.coords[0]
        crs_m = utm_for_lonlat(cx, cy)
        gps_gdf = gps_gdf_ll.to_crs(crs_m)
        roads_proj = roads_clip_ll.to_crs(crs_m)
    else:
        gps_gdf = gps_gdf_ll.to_crs(epsg=cfg.force_epsg)
        roads_proj = roads_clip_ll.to_crs(epsg=cfg.force_epsg)

    if "edge_id" not in roads_proj.columns:
        roads_proj = roads_proj.reset_index(drop=False).rename(columns={"index": "edge_id"})
    roads_proj = explode_lines(roads_proj)
    try:
        gps_buffer = gpd.GeoSeries([gps_gdf.unary_union.envelope], crs=gps_gdf.crs).buffer(cfg.clip_buffer_m)
        roads_proj = gpd.overlay(roads_proj, gpd.GeoDataFrame(geometry=gps_buffer), how="intersection")
        if roads_proj.empty:
            roads_proj = roads_clip_ll.to_crs(gps_gdf.crs)
    except Exception:
        pass

    G, node_coord, roads_proj, node_index_map = build_road_graph_fast(roads_proj)
    try:
        sindex = roads_proj.sindex
    except Exception:
        sindex = None

    gps_gdf["GPS时间"] = pd.to_datetime(gps_gdf["GPS时间"])
    gps_gdf = gps_gdf.sort_values("GPS时间").reset_index(drop=True)
    T = len(gps_gdf)

    # 生成候选
    all_cands = []
    for _, r in gps_gdf.iterrows():
        cands = generate_candidates_fast(r.geometry, roads_proj, sindex,
                                         cfg.search_radius_m, cfg.k_candidates)
        all_cands.append(cands)
    # 扩大半径一次尝试
    for i, c in enumerate(all_cands):
        if len(c) == 0:
            all_cands[i] = generate_candidates_fast(gps_gdf.loc[i, "geometry"], roads_proj, sindex,
                                                   cfg.search_radius_m * 2.0, cfg.k_candidates)
    if any(len(c) == 0 for c in all_cands):
        out = gps_gdf_ll.copy()
        out["edge_id"] = None
        out["matched_lon"] = out.geometry.x
        out["matched_lat"] = out.geometry.y
        out["dist_m"] = None
        out["viterbi_lp"] = None
        return pd.DataFrame(out.drop(columns="geometry"))

    emis_logp = [np.array([emission_log_prob(c["dist_m"], cfg.sigma_z) for c in cands], dtype=float)
                 for cands in all_cands]

    cand_infos = []
    for cands in all_cands:
        infos = []
        for c in cands:
            infos.append({
                "snapped_x": c["snapped_x"],
                "snapped_y": c["snapped_y"],
                "dist_m": c["dist_m"],
                "_start": c["_start"],
                "_end": c["_end"],
                "proj_to_start": c["proj_to_start"],
                "seglen": c["seg_length"],
                "geom": c["geom"],
                "edge_id": c["edge_id"]
            })
        cand_infos.append(infos)

    # Viterbi DP
    viterbi = [emis_logp[0].copy()]
    backptr = [np.full(len(all_cands[0]), -1, dtype=int)]
    spcache = SPCache(G)

    for t in range(1, T):
        prev_infos = cand_infos[t - 1]; curr_infos = cand_infos[t]
        vt = np.full(len(curr_infos), -np.inf); bt = np.full(len(curr_infos), -1, dtype=int)
        lon1, lat1 = gps_gdf_ll.loc[t - 1, ["轨迹点经度", "轨迹点纬度"]]
        lon2, lat2 = gps_gdf_ll.loc[t, ["轨迹点经度", "轨迹点纬度"]]
        gc = haversine_m(lon1, lat1, lon2, lat2)
        dt = (gps_gdf.loc[t, "GPS时间"] - gps_gdf.loc[t - 1, "GPS时间"]).total_seconds()
        dt = max(dt, 1e-3)

        for j, cj in enumerate(curr_infos):
            best_val = -np.inf; best_i = -1
            for i, ci in enumerate(prev_infos):
                sp_len = shortest_path_len_between_snaps_fast(
                    spcache,
                    ci["_start"], ci["_end"], ci["proj_to_start"], ci["seglen"] - ci["proj_to_start"],
                    cj["_start"], cj["_end"], cj["proj_to_start"], cj["seglen"] - cj["proj_to_start"],
                    ci["seglen"], cj["seglen"]
                )
                lp_trans = transition_log_prob(sp_len, gc, dt, cfg.beta, cfg.v_max)
                val = viterbi[-1][i] + lp_trans
                if val > best_val:
                    best_val = val; best_i = i
            vt[j] = best_val + emis_logp[t][j]; bt[j] = best_i

        viterbi.append(vt); backptr.append(bt)

    # 回溯
    path_idx = np.zeros(T, dtype=int); path_idx[-1] = int(np.argmax(viterbi[-1]))
    for t in range(T - 2, -1, -1):
        path_idx[t] = int(backptr[t + 1][path_idx[t + 1]])

    # 组织输出，投影回经纬
    transformer = Transformer.from_crs(gps_gdf.crs, "EPSG:4326", always_xy=True)
    matched = []
    for t in range(T):
        c = cand_infos[t][path_idx[t]]
        x = c["snapped_x"]; y = c["snapped_y"]
        lon, lat = transformer.transform(x, y)
        matched.append({
            "traj_id": gps_gdf_ll.loc[t, "订单ID"],
            "timestamp": gps_gdf_ll.loc[t, "GPS时间"],
            "orig_lon": gps_gdf_ll.loc[t, "轨迹点经度"],
            "orig_lat": gps_gdf_ll.loc[t, "轨迹点纬度"],
            "matched_lon": lon,
            "matched_lat": lat,
            "edge_id": c["edge_id"],
            "dist_m": c["dist_m"],
            "viterbi_lp": float(viterbi[t][path_idx[t]])
        })
    return pd.DataFrame(matched)


# -------------------------------
# multiprocessing helpers for per-file order-level parallelism
# -------------------------------
def _init_worker(roads):
    global _ROADS_GLOBAL
    _ROADS_GLOBAL = roads


def _viterbi_worker(args):
    df, cfg = args
    return viterbi_match_fast(df, roads_ll=None, cfg=cfg)


# -------------------------------
# process a single CSV file (read, match per order, save matched csv)
# -------------------------------
def process_single_file(csv_path: Path, roads: gpd.GeoDataFrame, cfg: MMConfig):
    print(f"Processing file: {csv_path.name}")
    traj = pd.read_csv(csv_path)
    need_cols = {"司机ID", "订单ID", "GPS时间", "轨迹点经度", "轨迹点纬度"}
    missing = need_cols - set(traj.columns)
    if missing:
        raise ValueError(f"轨迹CSV {csv_path} 缺少列: {missing}，需要列: {need_cols}")

    groups = list(traj.groupby("订单ID"))
    n_groups = len(groups)
    print(f"  Orders in file: {n_groups}")

    # Determine workers for this file (cfg.n_workers already set globally)
    if cfg.n_workers is None:
        # choose automatically
        if os.name == "nt":
            allow = os.environ.get("ALLOW_MP_WINDOWS", "0") == "1"
            workers = 1 if not allow else max(1, mp.cpu_count() - 1)
        else:
            workers = max(1, mp.cpu_count() // 2 - 1)
    else:
        workers = cfg.n_workers
        if os.name == "nt" and workers > 1 and os.environ.get("ALLOW_MP_WINDOWS", "0") != "1":
            print("Windows detected and ALLOW_MP_WINDOWS not set: forcing workers=1 for safety.")
            workers = 1

    out_list = []

    if workers <= 1:
        # serial
        for tid, df in tqdm(groups, total=n_groups, desc=f"  Matching orders ({csv_path.name})"):
            df = df.sort_values("GPS时间").reset_index(drop=True)
            res = viterbi_match_fast(df, roads_ll=roads, cfg=cfg)
            out_list.append(res)
    else:
        # parallel: create pool with initializer to set roads global
        args = [(df.sort_values("GPS时间").reset_index(drop=True), cfg) for _, df in groups]
        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(roads,)) as pool:
            for res in tqdm(pool.imap(_viterbi_worker, args), total=n_groups, desc=f"  Parallel matching ({csv_path.name})"):
                out_list.append(res)

    # concat outputs and save
    if out_list:
        out = pd.concat(out_list, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["traj_id", "timestamp", "orig_lon", "orig_lat", "matched_lon", "matched_lat", "edge_id", "dist_m", "viterbi_lp"])
    # output filename
    out_name = f"matched_{csv_path.stem}.csv"
    out_path = Path(cfg.output_folder) / out_name
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved matched file: {out_path}")
    return out_path


# -------------------------------
# build road_list and save
# -------------------------------
def build_and_save_road_list(roads: gpd.GeoDataFrame, cfg: MMConfig):
    print("Generating road_list (exploded edges) ...")
    roads_expl = explode_lines(roads.reset_index(drop=True).copy()).reset_index(drop=True)
    if "edge_id" not in roads_expl.columns:
        roads_expl = roads_expl.reset_index(drop=False).rename(columns={"index": "edge_id"})
    # compute length in meters using UTM of network center if possible
    try:
        cx, cy = roads_expl.unary_union.centroid.coords[0]
        crs_m = utm_for_lonlat(cx, cy)
        roads_m = roads_expl.to_crs(crs_m)
        roads_expl["length_m"] = roads_m.geometry.length
    except Exception:
        roads_expl["length_m"] = roads_expl.to_crs(epsg=3857).geometry.length

    # start/end lon/lat
    starts = []; ends = []
    for geom in roads_expl.geometry:
        try:
            coords = list(geom.coords)
            s = coords[0]; e = coords[-1]
            starts.append(s); ends.append(e)
        except Exception:
            starts.append((None, None)); ends.append((None, None))
    roads_expl["start_lon"] = [s[0] for s in starts]
    roads_expl["start_lat"] = [s[1] for s in starts]
    roads_expl["end_lon"] = [e[0] for e in ends]
    roads_expl["end_lat"] = [e[1] for e in ends]

    # pick non-geometry columns to keep
    non_geom_cols = [c for c in roads_expl.columns if c != "geometry"]
    road_list = roads_expl[non_geom_cols].copy()
    # add WKT if wanted
    try:
        road_list["geometry_wkt"] = roads_expl.geometry.apply(lambda g: g.wkt if g is not None else None)
    except Exception:
        road_list["geometry_wkt"] = None

    # save
    out_path = Path(cfg.output_folder) / cfg.output_roadlist
    road_list.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved road_list to: {out_path}")
    return road_list


# -------------------------------
# main batch flow
# -------------------------------
def main(cfg: MMConfig):
    # prepare paths
    input_folder = Path(cfg.input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"input_folder not found: {input_folder}")
    Path(cfg.output_folder).mkdir(parents=True, exist_ok=True)

    # load roads (WGS84 expected)
    print("Loading roads:", cfg.roads_path)
    roads = gpd.read_file(cfg.roads_path)
    if roads.crs is None:
        roads.set_crs(epsg=4326, inplace=True)
    else:
        roads = roads.to_crs(epsg=4326)

    # build and save road_list once
    road_list_df = build_and_save_road_list(roads, cfg)

    # find csv files in input_folder
    csv_files = sorted([p for p in input_folder.glob("*.csv")])
    if not csv_files:
        print("No CSV files found in input_folder:", input_folder)
        return

    # process files one by one (each file's internal orders can be parallelized)
    all_matched_paths = []
    for csv_path in csv_files:
        try:
            matched_path = process_single_file(csv_path, roads, cfg)
            all_matched_paths.append(matched_path)
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}", file=sys.stderr)

    print("All done. Matched files saved to:", cfg.output_folder)
    print("Road list saved to:", Path(cfg.output_folder) / cfg.output_roadlist)


# -------------------------------
# run
# -------------------------------
if __name__ == "__main__":
    # === 修改下面 cfg 路径为你的实际路径 ===
    cfg = MMConfig(
        roads_path="data/GaiyaData/RoadMatch/roads_chengdu.shp",
        input_folder="data/GaiyaData/raw/2016年成都滴滴轨迹数据",   # 放多个 csv 的文件夹
        output_folder="data/GaiyaData/RoadMatch/processed/match_origin",
        output_roadlist="roads_chengdu_edges.csv",
        search_radius_m=50.0,
        k_candidates=3,
        sigma_z=12.0,
        beta=100.0,
        v_max=30.0,
        clip_buffer_m=2000.0,
        force_epsg=None,
        n_workers=None,  # None -> 自动选择（Unix: cpu_count()-1；Windows: 1 unless ALLOW_MP_WINDOWS=1）
    )
    main(cfg)
