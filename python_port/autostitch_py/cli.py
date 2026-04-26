from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from autostitch_py.pipeline import MosaicPipeline, PipelineConfig
else:
    from .pipeline import MosaicPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Python rewrite of AutoStitching")
    p.add_argument(
        "--params_f",
        default=r"D:\forest_good3_hejiangdong\forest_good3_hejiangdong\1_initial\params\forest_good3_hejiangdong_calibrated_external_camera_parameters.txt",
    )
    p.add_argument(
        "--tp_f",
        default=r"D:\forest_good3_hejiangdong\forest_good3_hejiangdong\1_initial\params\forest_good3_hejiangdong_tp_pix4d.txt",
    )
    p.add_argument(
        "--img_dir",
        default=r"E:/southeastcode/image_mosaic/UAV_dateset_three_image_mosaic/dataset/HeJiaDong",
    )
    p.add_argument("--is_time_consecutive", action="store_true")
    p.add_argument("--output_dir", default="./python_port/output")
    p.add_argument("--ba_iters", type=int, default=30)
    p.add_argument("--ba_sample_step", type=int, default=3)
    p.add_argument("--ba_lambda", type=float, default=1e-3)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = PipelineConfig(
        params_f=args.params_f,
        tp_f=args.tp_f,
        img_dir=args.img_dir,
        is_time_consecutive=args.is_time_consecutive,
        output_dir=args.output_dir,
        ba_iters=args.ba_iters,
        ba_sample_step=args.ba_sample_step,
        ba_lambda=args.ba_lambda,
    )
    report = MosaicPipeline(cfg).run()
    print("Done:", report)


if __name__ == "__main__":
    main()
