<img src="fig/MTG-logo-hires-transparent.png" width="260" align="left" />
<br clear="left" /><br>

------

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Pytorch](https://img.shields.io/badge/Pytorch-2.2.1%2B-blue)](https://pytorch.org/) 

# MTGBench-Dataset

[数据集下载](https://cilab-urbandata.github.io/) | [论文仓库](https://github.com/CILAB-UrbanData/MTGBench) | [会议论文]() | [English](https://github.com/CILAB-UrbanData/MTGBench-Dataset/blob/master/README.md)

MTGBench-Dataset 提供了以下信息：各种原始数据集的预处理程序，不同信息的数据的规范格式，各个数据集下载好后如何放置到 [论文仓库](https://github.com/CILAB-UrbanData/MTGBench) 中

## 标准数据格式

### 轨迹路网数据

主要参考了 [DRPK](https://github.com/derekwtian/DRPK) 中匹配好的轨迹数据的存储，举例如下

```angular2html
edmugrip,1212038019:edmugrip,"[(137.96932188674592, 1212038019, -122.39606, 37.792731), (41.63089590260138, 1212038170, -122.40162, 37.793008)]","[[6557, 1212037972.113, 2.943, 71.205], [8965, 1212038043.318, 2.943, 15.766], [10763, 1212038059.085, 2.943, 3.908], [10761, 1212038062.993, 2.943, 34.729], [1607, 1212038097.721, 2.943, 36.551], [11780, 1212038134.272, 2.943, 72.254], [3634, 1212038206.526, 2.943, 36.875], [8612, 1212038243.402, 5.118, 55.304], [6290, 1212038298.705, 5.118, 20.926], [1610, 1212038319.631, 5.118, 20.414], [3402, 1212038340.045, 5.118, 20.371], [3404, 1212038360.416, 4.261, 24.524], [1612, 1212038384.94, 4.261, 24.546], [1614, 1212038409.486, 4.261, 34.233]]","[(-122.39636, 37.79236, 1212038019, 2.942585), (-122.3978, 37.79133, 1212038077, 5.11841), (-122.40007, 37.79343, 1212038137, 4.260853), (-122.40167, 37.79341, 1212038170, 4.260853)]"
...
```
每一行数据由四部分组成：车牌号（表示移动物体的id），轨迹编号，偏置信息，路网编号和时间戳序列，具体以如下列出。

```angular2html
offsets: (137.96932188674592, 1212038019, -122.39606, 37.792731) means 
         (the distance between the start point of source segment and the source location,
          departure time,
          the longitude of source location,
          the latitude of source location,)
          The destination location has similar information.

segment sequence: [6557, 1212037972.113, 2.943, 71.205] means 
                  [segment id,
                   the timestamp when trajectory enter this segment,
                   the average speed when the trajectory go through this segment,
                   the duration for the trajectory through this segment]
```
如果原始信息缺少某些部分（如车牌号常常没有），我们采取的策略是用相似的信息替换或者用空字符占位。

### 区域订单数据

区域订单数据的标准格式主要参考 [NYC_taxi](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) ，根据任务需要只保留四个订单数据的基本信息，开始时间，开始区域，结束时间，结束区域。并附上具体的区域划分文件（如果区域划分只是简单的去一个矩形bbox进行矩形划分则没必要附上区域文件）

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
