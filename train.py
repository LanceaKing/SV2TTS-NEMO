import os
import torch
import pytorch_lightning as pl
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import Tacotron2Model
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from sv2tts import SV2TTSModel


@hydra_runner(config_name='sv2tts')
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get('exp_manager', None))
    model = SV2TTSModel(cfg=cfg.model, trainer=trainer)
    from_pretrained_modules(model, cfg.get('pretrained_modules', None), cfg.get('checkpoint_modules', None))
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    freeze_modules(model, cfg.get('freeze_modules', None))
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


def from_pretrained_modules(model, module_list, path=None):
    if module_list is None or len(module_list) == 0:
        return
    if path is None:
        restored_model = Tacotron2Model.from_pretrained('tts_en_tacotron2', map_location='cpu', strict=True)
        for name in module_list:
            src = getattr(restored_model, name)
            dst = getattr(model, name)
            dst.load_state_dict(src.state_dict())
        del restored_model
    else:
        src_state_dict = torch.load(path)['state_dict']
        dst_state_dict = model.state_dict()
        for key in dst_state_dict.keys():
            if key[:key.index('.')] in module_list:
                dst_state_dict[key] = src_state_dict[key]
        model.load_state_dict(dst_state_dict)
        del src_state_dict


def freeze_modules(model, module_list):
    if module_list is None or len(module_list) == 0:
        return
    for name in module_list:
        m = getattr(model, name)
        m.freeze()


if __name__ == '__main__':
    main()
