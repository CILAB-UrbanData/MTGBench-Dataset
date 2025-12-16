<img src="fig/MTG-logo-hires-transparent.png" width="260" align="left" />
<br clear="left" /><br>

------

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1%2B-blue)](https://pytorch.org/)

# MTGBench-Dataset

[Dataset Download](https://cilab-urbandata.github.io/) |
[Paper Repository](https://github.com/CILAB-UrbanData/MTGBench) |
[Conference Paper]() |
[中文说明](https://github.com/CILAB-UrbanData/MTGBench-Dataset/blob/master/README_zh.md)

MTGBench-Dataset provides:
- Preprocessing scripts for various raw mobility and transportation datasets  
- Unified and standardized data formats for heterogeneous data sources  
- Instructions on how to organize downloaded datasets into the
  [MTGBench paper repository](https://github.com/CILAB-UrbanData/MTGBench)

This repository focuses on **data standardization and preparation**, enabling
fair and reproducible benchmarking across multiple mobility tasks.

---

## Standard Data Formats

### Trajectory–Road Network Data

The trajectory–road network data format mainly follows the matched trajectory
storage design in
[DRPK](https://github.com/derekwtian/DRPK).
An example is shown below:

```text
edmugrip,1212038019:edmugrip,"[(137.96932188674592, 1212038019, -122.39606, 37.792731), (41.63089590260138, 1212038170, -122.40162, 37.793008)]","[[6557, 1212037972.113, 2.943, 71.205], [8965, 1212038043.318, 2.943, 15.766], [10763, 1212038059.085, 2.943, 3.908], [10761, 1212038062.993, 2.943, 34.729], [1607, 1212038097.721, 2.943, 36.551], [11780, 1212038134.272, 2.943, 72.254], [3634, 1212038206.526, 2.943, 36.875], [8612, 1212038243.402, 5.118, 55.304], [6290, 1212038298.705, 5.118, 20.926], [1610, 1212038319.631, 5.118, 20.414], [3402, 1212038340.045, 5.118, 20.371], [3404, 1212038360.416, 4.261, 24.524], [1612, 1212038384.94, 4.261, 24.546], [1614, 1212038409.486, 4.261, 34.233]]","[(-122.39636, 37.79236, 1212038019, 2.942585), (-122.3978, 37.79133, 1212038077, 5.11841), (-122.40007, 37.79343, 1212038137, 4.260853), (-122.40167, 37.79341, 1212038170, 4.260853)]"
...
```

Each row consists of four components:

* Vehicle ID (e.g., license plate or unique object identifier)

* Trajectory ID

* Offset information

* Road segment sequence with timestamps

Detailed explanations are given below.
```text
offsets: (137.96932188674592, 1212038019, -122.39606, 37.792731) means
         (
           distance from the start point of the source road segment to the source location,
           departure time,
           longitude of the source location,
           latitude of the source location
         )
         The destination location follows the same format.

segment sequence: [6557, 1212037972.113, 2.943, 71.205] means
                  [
                    road segment ID,
                    timestamp when the trajectory enters this segment,
                    average speed on this segment,
                    duration spent on this segment
                  ]
```

If certain attributes are missing in the raw data (e.g., vehicle IDs are often
unavailable), we adopt the following strategies:

* Replace them with semantically similar identifiers when possible

* Otherwise, use empty placeholders to preserve format consistency
  
### Region-Based Order Data

The standard format for region-based order data mainly follows the [NYC Taxi Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

For MTGBench tasks, we retain only the four essential attributes of each order:

* Start time

* Origin region

* End time

* Destination region

In addition, a region partition file is provided when necessary.
If regions are defined by a simple rectangular bounding box grid, the region
definition file may be omitted.

## Tutorial

All datasets should be placed in the following directory structure inside the
MTGBench repository:
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
```
This unified structure ensures compatibility across different benchmarks,
models, and experimental pipelines in MTGBench.