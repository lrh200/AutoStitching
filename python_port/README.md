# Python Rewrite (autostitch-py)

This folder contains a Python rewrite of the topology-guided mosaicking pipeline.

## Inputs

The CLI defaults are already set to your requested paths:

- `params_f = D:\forest_good3_hejiangdong\forest_good3_hejiangdong\1_initial\params\forest_good3_hejiangdong_calibrated_external_camera_parameters.txt`
- `tp_f = D:\forest_good3_hejiangdong\forest_good3_hejiangdong\1_initial\params\forest_good3_hejiangdong_tp_pix4d.txt`
- `img_dir = E:/southeastcode/image_mosaic/UAV_dateset_three_image_mosaic/dataset/HeJiaDong`

## Run

```bash
cd python_port
python -m autostitch_py.cli \
  --params_f "D:\\forest_good3_hejiangdong\\forest_good3_hejiangdong\\1_initial\\params\\forest_good3_hejiangdong_calibrated_external_camera_parameters.txt" \
  --tp_f "D:\\forest_good3_hejiangdong\\forest_good3_hejiangdong\\1_initial\\params\\forest_good3_hejiangdong_tp_pix4d.txt" \
  --img_dir "E:/southeastcode/image_mosaic/UAV_dateset_three_image_mosaic/dataset/HeJiaDong"
```

Add `--is_time_consecutive` to force the time-consecutive topology mode.

## Outputs

Generated in `output_dir` (default `python_port/output`):

- `similarity_mat.txt`
- `report.json`
- `mosaic.png`

## Notes

- Feature extraction/matching uses ORB + geometric verification (F-matrix + homography).
- Topology search supports both sequential and unordered modes.
- Priors from external camera params and Pix4D tiepoint file are fused into guiding costs.
