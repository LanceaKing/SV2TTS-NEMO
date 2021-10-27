import pickle as pkl
from pathlib import Path

import pytorch_lightning as pl
import torch
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.models import Tacotron2Model
from nemo.core import typecheck
from nemo.core.config import hydra_runner
from nemo.core.neural_types import (AcousticEncodedRepresentation, AudioSignal,
                                    EmbeddedTextType, LengthsType,
                                    MelSpectrogramType, NeuralType)
from nemo.utils.exp_manager import exp_manager


class SV2TTSModel(Tacotron2Model):

    @property
    def input_types(self):
        input_types = super().input_types
        input_types['speaker_embedding'] = NeuralType(('B', 'D'), AcousticEncodedRepresentation())
        return input_types

    @typecheck()
    def forward(self, *, speaker_embedding, tokens, token_len, audio=None, audio_len=None):
        if audio is not None and audio_len is not None:
            spec_target, spec_target_len = self.audio_to_melspec_precessor(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding=token_embedding, token_len=token_len)
        encoder_embedding = self.concatenate_speaker_embedding(encoder_embedding, speaker_embedding)
        if self.training:
            spec_pred_dec, gate_pred, alignments = self.decoder(
                memory=encoder_embedding, decoder_inputs=spec_target, memory_lengths=token_len
            )
        else:
            spec_pred_dec, gate_pred, alignments, pred_length = self.decoder(
                memory=encoder_embedding, memory_lengths=token_len
            )
        spec_pred_postnet = self.postnet(mel_spec=spec_pred_dec)

        if not self.calculate_loss:
            return spec_pred_dec, spec_pred_postnet, gate_pred, alignments, pred_length
        return spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments

    @staticmethod
    def concatenate_speaker_embedding(encoder_embedding, speaker_embedding):
        speaker_embedding = speaker_embedding.unsqueeze(1)
        speaker_embedding = speaker_embedding.repeat(1, encoder_embedding.size(1), 1)
        encoder_embedding = torch.cat((encoder_embedding, speaker_embedding), dim=2)
        return encoder_embedding

    @typecheck(
        input_types={
            "tokens": NeuralType(('B', 'T'), EmbeddedTextType()),
            "speaker_embedding": NeuralType(('B', 'D'), AcousticEncodedRepresentation()),
        },
        output_types={"spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType())},
    )
    def generate_spectrogram(self, *, tokens, speaker_embedding):
        self.eval()
        self.calculate_loss = False
        token_len = torch.tensor([len(i) for i in tokens]).to(self.device)
        tensors = self(tokens=tokens, token_len=token_len, speaker_embedding=speaker_embedding)
        spectrogram_pred = tensors[1]

        if spectrogram_pred.shape[0] > 1:
            # Silence all frames past the predicted end
            mask = ~get_mask_from_lengths(tensors[-1])
            mask = mask.expand(spectrogram_pred.shape[1], mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            spectrogram_pred.data.masked_fill_(mask, self.pad_value)

        return spectrogram_pred

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len, speaker_embedding = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, _ = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len, speaker_embedding=speaker_embedding
        )

        loss, _ = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
        )

        output = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': {'loss': loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len, speaker_embedding = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len, speaker_embedding=speaker_embedding
        )

        loss, gate_target = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
        )
        return {
            "val_loss": loss,
            "mel_target": spec_target,
            "mel_postnet": spec_pred_postnet,
            "gate": gate_pred,
            "gate_target": gate_target,
            "alignments": alignments,
        }


class SV2TTSDataset(AudioToCharDataset):

    def __init__(self, speaker_embeddings_filepath, *args, **kwargs):
        with open(speaker_embeddings_filepath, 'rb') as f:
            self.speaker_embeddings = pkl.load(f)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        output = super().__getitem__(index)
        sample = self.manifest_processor.collection[index]
        p = Path(sample.audio_file)
        uniq_name = '@'.join(p.relative_to(p.parents[2]).parts)
        return *output, torch.from_numpy(self.speaker_embeddings[uniq_name])

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id)

    @property
    def output_types(self):
        output_types = super().output_types
        output_types['speaker_embedding'] = NeuralType(('B', 'D'), AcousticEncodedRepresentation())
        return output_types


def _speech_collate_fn(batch, pad_id):
    _, audio_lengths, _, tokens_lengths, _ = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens, speaker_embeddings = [], [], []
    for sig, sig_len, tokens_i, tokens_i_len, emb in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)
        speaker_embeddings.append(emb)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)
    speaker_embeddings = torch.stack(speaker_embeddings)

    return audio_signal, audio_lengths, tokens, tokens_lengths, speaker_embeddings
