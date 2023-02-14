import json
import os
from dataclasses import dataclass, field
from typing import List

class ExperimentTypes:
    CLASSIFY_OVER_UNDER = 'cou'
    TRACE_PREDICTION = 'trp'

ALLOWED_EXPT_TYPES = [ExperimentTypes.CLASSIFY_OVER_UNDER,
                      ExperimentTypes.TRACE_PREDICTION]

def get_dataset_dir(expt_type):
    if expt_type == ExperimentTypes.TRACE_PREDICTION:
        return '../data/sim_data/trace_dataset_complex'
    elif expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
        return '../data/sim_data/under_over_centered_hard2'

def is_crop_task(expt_type):
    return expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER

def is_point_pred(expt_type):
    return expt_type == ExperimentTypes.TRACE_PREDICTION

def save_config_params(path, expt_config):
    with open(os.path.join(path, 'config.json'), 'w') as f:
        dct = {}
        for k in dir(expt_config):
            if k[0] != '_':
                dct[k] = getattr(expt_config, k)
        json.dump(dct, f, indent=4)
        f.close()
    with open(os.path.join(path, 'expt_class.txt'), 'w') as f:
        f.write(str(expt_config.__class__.__name__))
        f.close()

def load_config_class(path):
    with open(os.path.join(path, 'config.json'), 'r') as f:
        dct = json.load(f)
        f.close()
    return BaseConfig(**dct)

@dataclass
class BaseConfig:
    expt_type: str = ExperimentTypes.TRACE_PREDICTION
    dataset_dir: List[str] = field(default_factory=lambda: [get_dataset_dir(ExperimentTypes.TRACE_PREDICTION)])
    dataset_weights: List[float] = field(default_factory=lambda: [1.0])
    dataset_real: List[bool] = field(default_factory=lambda: [False])
    img_height: int = 100
    img_width: int = 100
    crop_width: int = 80
    num_keypoints: int = 1
    gauss_sigma: int = 2
    classes: int = 1
    epochs: int = 150
    batch_size: int = 4
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 1
    eval_checkpoint_freq: int = 1
    min_checkpoint_freq: int = 10
    resnet_type: str = '50'
    pretrained: bool = False
    oversample: bool = False
    oversample_rate: float = 0.8
    oversample_factor: float = 1.0
    rot_cond: bool = False
    sharpen: bool = False
    learning_rate: float = 1.0e-5
    contrast: bool = False
    mark_crossing: bool = False
    expand_spline: bool = False

@dataclass
class TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp(BaseConfig):
    crop_width: int = 32
    cond_point_dist_px: int = 12
    condition_len: int = 3
    pred_len: int = 1
    img_height: int = 96
    img_width: int = 96
    resnet_type: str = 'UNet34'
    batch_size: int = 64
    dataset_dir: List[str] = field(default_factory=lambda: ['../data/sim_data/trace_dataset_hard_2', '../data/sim_data/annotations_hard_knots_3', '../data/sim_data/trace_dataset_hard_adjacent_1', '../data/real_data/real_data_for_tracer'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.3, 0.15, 0.35, 0.2])
    dataset_real: List[bool] = field(default_factory=lambda: [False, False, False, True])
    oversample: bool = True
    oversample_rate: float = 0.95
    rot_cond: bool = True
    epochs: int = 125
    sharpen: bool = True

@dataclass
class UNDER_OVER_RNet34_lr1e4_medley_03Hard2_wReal_B16_recentered_mark_crossing_smaller(BaseConfig):
    expt_type: str = ExperimentTypes.CLASSIFY_OVER_UNDER
    dataset_dir: str = field(default_factory=lambda: ['../data/sim_data/under_over_hard1_10_recenter', '../data/sim_data/under_over_hard2_10_recenter', '../data/real_data/under_over_REAL_centered/'])
    dataset_weights: List[float] = field(default_factory=lambda: [0.7, 0.3, 0.2])
    dataset_real: List[bool] = field(default_factory=lambda: [False, False, True])
    classes: int = 1
    img_height: int = 20
    img_width: int = 20
    crop_width: int = 10
    num_keypoints: int = 1
    gauss_sigma: int = 1
    epochs: int = 50
    batch_size: int = 16
    cond_point_dist_px: int = 20
    condition_len: int = 5
    pred_len: int = 0
    eval_checkpoint_freq: int = 1
    min_checkpoint_freq: int = 10
    resnet_type: str = '34'
    pretrained: bool = True
    rot_cond: bool = True
    sharpen: bool = True
    learning_rate: float = 1.0e-4
    mark_crossing: bool = True

def get_class_name(cls):
    return cls.__name__

ALL_EXPERIMENTS_LIST = [BaseConfig,
TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp,
UNDER_OVER_RNet34_lr1e4_medley_03Hard2_wReal_B16_recentered_mark_crossing_smaller]

ALL_EXPERIMENTS_CONFIG = {get_class_name(expt): expt for expt in ALL_EXPERIMENTS_LIST}
