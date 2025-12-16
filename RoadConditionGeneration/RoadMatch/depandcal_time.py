import pandas as pd
from pathlib import Path

input_folder="/mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/RoadMatch/processed/match_origin"
input_folder = Path(input_folder)
csv_files = sorted([p for p in input_folder.glob("*.csv")])
for csv_path in csv_files:
    # === Step 1: 读取数据 ===
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 排序
    df = df.sort_values(["traj_id", "timestamp"]).reset_index(drop=True)

    # === Step 2: 去重逻辑（相同traj_id & edge_id连续重复只保留第一次） ===
    same_edge = (df["edge_id"] == df.groupby("traj_id")["edge_id"].shift())
    df_dedup = df.loc[~same_edge].copy().reset_index(drop=True)

    # === Step 3: 计算停留时间（基于路段切换） ===
    dwell_times = []

    for traj_id, group in df.groupby("traj_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        # 找到去重后的序列
        edges = group["edge_id"].tolist()
        times = group["timestamp"].tolist()

        # 遍历路段切换
        for i in range(len(edges)):
            edge = edges[i]
            start_time = times[i]

            if i < len(edges) - 1:  # 中间路段：用下一条edge的开始时间减去当前开始时间
                end_time = times[i + 1]
            else:  # 最后一条：用最后一次出现该edge的时间
                end_time = group[group["edge_id"] == edge]["timestamp"].iloc[-1]

            dwell_time = (end_time - start_time).total_seconds()
            dwell_times.append([traj_id, edge, start_time, dwell_time])

    # 转换为 DataFrame
    dwell_df = pd.DataFrame(dwell_times, columns=["traj_id", "edge_id", "timestamp", "dwell_time"])

    # === Step 4: 合并路段名称等信息 ===
    df_dedup = df_dedup.merge(dwell_df, on=["traj_id", "edge_id", "timestamp"], how="left")
    df_dedup = df_dedup.dropna(subset=["edge_id"])
    df_dedup["edge_id"] = df_dedup["edge_id"].astype(int)

    # === Step 5: 保存结果 ===
    out_path = f"/mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/RoadMatch/processed/{csv_path.stem}_dedup_with_dwell.csv"
    df_dedup.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ 已生成 {out_path}")
