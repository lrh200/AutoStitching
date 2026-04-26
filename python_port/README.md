# Python Rewrite (autostitch-py)

This folder contains a Python rewrite of the topology-guided mosaicking pipeline.

## Inputs

The CLI defaults are already set to your requested paths:

- `params_f = D:\forest_good3_hejiangdong\forest_good3_hejiangdong\1_initial\params\forest_good3_hejiangdong_calibrated_external_camera_parameters.txt`
- `tp_f = D:\forest_good3_hejiangdong\forest_good3_hejiangdong\1_initial\params\forest_good3_hejiangdong_tp_pix4d.txt`
- `img_dir = E:/southeastcode/image_mosaic/UAV_dateset_three_image_mosaic/dataset/HeJiaDong`

### Expected input patterns

1. `*_tp_pix4d.txt` (Pix4D tie-point observations)
   - block pattern:
     - line with image name
     - multiple lines: `point_id x y scale`
   - same `point_id` appearing in two images means a valid cross-image correspondence.
   - parser builds:
     - pair co-visibility score (shared point count)
     - per-pair 2D-2D correspondences for geometric verification.

2. `*_calibrated_external_camera_parameters.txt` (external camera params)
   - row pattern: `imageName X Y Z Omega Phi Kappa`
   - parser extracts position `X,Y,Z` and tolerates wrapped/broken rows.

## Run

```bash
cd python_port
python -m autostitch_py.cli \
  --params_f "D:\\forest_good3_hejiangdong\\forest_good3_hejiangdong\\1_initial\\params\\forest_good3_hejiangdong_calibrated_external_camera_parameters.txt" \
  --tp_f "D:\\forest_good3_hejiangdong\\forest_good3_hejiangdong\\1_initial\\params\\forest_good3_hejiangdong_tp_pix4d.txt" \
  --img_dir "E:/southeastcode/image_mosaic/UAV_dateset_three_image_mosaic/dataset/HeJiaDong"
```

Add `--is_time_consecutive` to force the time-consecutive topology mode.
This ablation rewrite removes linear initialization and directly runs global 8DoF optimization.

You can also run the file directly (fixes `attempted relative import with no known parent package`):

```bash
python python_port/autostitch_py/cli.py \
  --params_f "D:\\forest_good3_hejiangdong\\forest_good3_hejiangdong\\1_initial\\params\\forest_good3_hejiangdong_calibrated_external_camera_parameters.txt" \
  --tp_f "D:\\forest_good3_hejiangdong\\forest_good3_hejiangdong\\1_initial\\params\\forest_good3_hejiangdong_tp_pix4d.txt" \
  --img_dir "E:/southeastcode/image_mosaic/UAV_dateset_three_image_mosaic/dataset/HeJiaDong"
```

## Outputs

Generated in `output_dir` (default `python_port/output`):

- `similarity_mat.txt`
- `report.json`
- `mosaic.png`

## Notes

- Feature extraction/matching uses ORB + geometric verification (F-matrix + homography).
- If Pix4D tie-point correspondences are present, they are used as the primary matching basis.
- Topology search supports both sequential and unordered modes.
- Priors from external camera params and Pix4D tiepoint file are fused into guiding costs.
- Alignment stage is ablation mode: **direct global 8DoF optimization** (`ba_iters`, `ba_sample_step`, `ba_lambda`) without affine linear initialization.
