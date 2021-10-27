import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.collections.common.callbacks import LogEpochTimeCallback
from sv2tts import SV2TTSModel


@hydra_runner(config_name='sv2tts')
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get('exp_manager', None))
    model = SV2TTSModel(cfg=cfg.model, trainer=trainer)
    model = fill_pretrained_modules(model, cfg.pretrained_modules)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


def fill_pretrained_modules(model, cfg):
    model.freeze()
    module_names = ['text_embedding', 'encoder', 'postnet']
    for name in module_names:
        module_file = cfg.get(name, None)
        if module_file is None:
            continue
        module = getattr(model, name)
        module.load_state_dict(torch.load(module_file))
    model.decoder.unfreeze()
    return model


if __name__ == '__main__':
    main()
