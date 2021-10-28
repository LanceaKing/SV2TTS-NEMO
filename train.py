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
    model = fill_pretrained_modules(model)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


def fill_pretrained_modules(model):
    restored_model = Tacotron2Model.from_pretrained('tts_en_tacotron2', map_location='cpu', strict=True)
    for name in ['text_embedding', 'encoder', 'postnet']:
        src = getattr(restored_model, name)
        dst = getattr(model, name)
        dst.load_state_dict(src.state_dict())
    del restored_model
    model.freeze()
    model.decoder.unfreeze()
    return model


if __name__ == '__main__':
    main()
