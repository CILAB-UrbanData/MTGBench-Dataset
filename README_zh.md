<img src="fig/MTG-logo-hires-transparent.png" width="260" align="left" />
<br clear="left" /><br>

------

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Pytorch](https://img.shields.io/badge/Pytorch-2.2.1%2B-blue)](https://pytorch.org/) 

# MTGBench-Dataset

[数据集下载](https://cilab-urbandata.github.io/) | [论文仓库](https://github.com/CILAB-UrbanData/MTGBench) | [会议论文]() | [English](https://github.com/CILAB-UrbanData/MTGBench-Dataset/blob/master/README.md)

MTGBench-Dataset 提供了以下信息：各种原始数据集的预处理程序，不同信息的数据的规范格式，各个数据集下载好后如何放置到 [论文仓库](https://github.com/CILAB-UrbanData/MTGBench) 中

## 标准数据格式

### 轨迹路网数据

### 区域订单数据

## MTGBench 中数据集结构

```text
MTGBench/
|- data/                # Dataset files
    |- sf/
        |- raw/         # contain road map and traj file
        |- MDTP/        # contain order file from traj
    |- Gaiya/           # so as the above
    |- porto/
    |- NYC
|- exp/
|- models/
|- ...
