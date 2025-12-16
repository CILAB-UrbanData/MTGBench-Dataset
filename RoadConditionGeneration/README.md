## Road Match

Since many trajectory data (Gaiya Chengdu, Porto) was collected in GPS format, we have to change it into road format. 

We perform a two-stage preprocessing workflow to obtain clean, segment-level trajectory sequences. First, we apply a batched HMM-based map-matching procedure to all raw GPS files, where each trip is processed independently. For every order, candidate road-segment projections are generated within a local search radius, emission and transition probabilities are computed using geometric distance and network-constrained shortest-path metrics, and the optimal sequence of road segments is inferred via Viterbi decoding. All matched trajectories and a normalized road-segment list are exported for downstream use. 

Second, we refine the matched sequences by removing consecutive duplicate segments within each trajectory and computing per-segment dwell time. After sorting by timestamp, consecutive identical edges are collapsed, and the time spent on each retained edge is obtained using the difference between the current and next segment’s timestamps. The resulting deduplicated, time-annotated segment sequences are stored as the final input for model training. 

All these codes are available in the ` RoadMatch/ ` , which takes Gaiya Chengdu data as example. Run ` match.py, depandcal_time.py ` step by step. By the way, ` CutfromSichuantoChengDu.py ` uses a bbox to extract Chengdu road network from Sichuan province road network which is downloaded from OSM. We have cut it for you.

## Convert to Standard Form

We preprocess the raw data to resolve a range of practical issues (e.g., timestamp normalization, format inconsistencies, missing/invalid records, and other dataset-specific quirks). Therefore, we do not release the raw-data preprocessing scripts. Instead, you can directly download the processed datasets from the [Dataset Download page](https://cilab-urbandata.github.io/). The code for converting the processed data into our standard format is provided in `converted2Standard/`.
