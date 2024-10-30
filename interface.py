import os
from typing import Literal

import torch
from tqdm import tqdm

from svc.seed_vc.inference import SeedVCArgs, infer, load_models
from utils import get_all_wav_files


class SeedVC:
    def __init__(self, speaker_name: Literal["databaker_male", "sisi"]):
        self.target: str = None
        self.diffusion_steps: int = 100
        self.length_adjust: float = 1.0
        self.inference_cfg_rate: float = 0.7
        self.n_quantizers: int = 3
        self.f0_condition: bool = True
        self.auto_f0_adjust: bool = True
        self.semi_tone_shift: int = 0
        self._init_config(speaker_name)

    def run(self, input_dir: str, output_dir: str):
        arg_list = self.prepare_args_list(input_dir, output_dir)
        models = load_models(arg_list[0])
        os.makedirs(output_dir, exist_ok=True)
        for i in tqdm(range(len(arg_list))):
            with torch.no_grad():
                infer(arg_list[i], models)

    def prepare_args_list(self, input_dir: str, out_dir: str):
        filelist = get_all_wav_files(input_dir)
        arg_list = []
        for i in tqdm(range(len(filelist))):
            source = filelist[i]
            args = SeedVCArgs(
                target=self.target,
                source=source,
                output=out_dir,
                diffusion_steps=self.diffusion_steps,
                length_adjust=self.length_adjust,
                inference_cfg_rate=self.inference_cfg_rate,
                n_quantizers=self.n_quantizers,
                f0_condition=self.f0_condition,
                auto_f0_adjust=self.auto_f0_adjust,
                semi_tone_shift=self.semi_tone_shift,
            )
            arg_list.append(args)
        return arg_list

    def _init_config(self, speaker_name):
        if speaker_name == "databaker_male":
            self.target = "src/svc/seed_vc/reference_audio/databaker_male_102377.wav"
            self.auto_f0_adjust = False
            self.semi_tone_shift = -12
        elif speaker_name == "sisi":
            self.target = "src/svc/seed_vc/reference_audio/sisi_VO01_07.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "waner":
            self.target = "src/svc/seed_vc/reference_audio/waner.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "xier":
            self.target = "src/svc/seed_vc/reference_audio/xier.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "yawen":
            self.target = "src/svc/seed_vc/reference_audio/yawen.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "zia":
            self.target = "src/svc/seed_vc/reference_audio/zia.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "zina":
            self.target = "src/svc/seed_vc/reference_audio/zina.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "zixue":
            self.target = "src/svc/seed_vc/reference_audio/zixue.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "chenlily":
            self.target = "src/svc/seed_vc/reference_audio/chenlily.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "chenyifa":
            self.target = "src/svc/seed_vc/reference_audio/chenyifa.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "tanyilun":
            self.target = "src/svc/seed_vc/reference_audio/tanyilun.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "yangmi":
            self.target = "src/svc/seed_vc/reference_audio/yangmi.wav"
            self.auto_f0_adjust = True
        elif speaker_name == "zhaoliying":
            self.target = "src/svc/seed_vc/reference_audio/zhaoliying.wav"
            self.auto_f0_adjust = True
