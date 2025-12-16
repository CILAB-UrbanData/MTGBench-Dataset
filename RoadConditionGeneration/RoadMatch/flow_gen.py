from pathlib import Path

import pandas as pd


DATA_DIR = Path("/mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/RoadMatch/processed")
OUTPUT_DIR = Path('/mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/TrGNN/flow_processed')


def _format_edge_id(value):
    if isinstance(value, (int, float)):
        if pd.notna(value) and float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def process_file(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['traj_id', 'edge_id', 'timestamp']]
    df['time_bin'] = df['timestamp'].dt.floor('10min')

    flow = df.groupby(['time_bin', 'edge_id', 'traj_id']).size().reset_index(name='count')
    flow = flow.groupby(['time_bin', 'edge_id']).size().reset_index(name='flow')

    flow_pivot = flow.pivot(index='time_bin', columns='edge_id', values='flow').fillna(0).astype(int)
    flow_pivot.columns = flow_pivot.columns.map(_format_edge_id)

    output_path = OUTPUT_DIR / f"{csv_path.stem}_flow_10min.csv"
    flow_pivot.to_csv(output_path)

    print(f"处理完成，输出文件：{output_path.name}")


def main() -> None:
    csv_files = sorted(
        csv_path
        for csv_path in DATA_DIR.glob('*.csv')
        if not csv_path.name.endswith('_flow_10min.csv')
    )
    if not csv_files:
        print(f"未在目录 {DATA_DIR} 找到任何 CSV 文件。")
        return

    for csv_path in csv_files:
        print(f"开始处理：{csv_path.name}")
        process_file(csv_path)


if __name__ == "__main__":
    main()
