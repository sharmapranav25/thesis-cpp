"""Thin adapters around upstream architectures; no pretrained downloads."""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream' / 'parseq'))
sys.path.insert(0, str(ROOT / 'upstream' / 'OpenOCR'))


class Recognizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.name = cfg['model']
        self.charset = cfg['charset']
        self.char_ids = {c: i for i, c in enumerate(self.charset)}
        self.max_length = cfg['max_label_length']
        self.bos = len(self.charset)
        self.eos = self.bos + 1
        self.pad = self.bos + 2
        if self.name in ['PARSeq', 'ViTSTR', 'CRNN']:
            from strhub.models.utils import create_model
            self.net = create_model(
                self.name.lower(), pretrained=False,
                charset_train=self.charset, charset_test=self.charset,
                max_label_length=self.max_length, img_size=cfg['image_size'],
                batch_size=cfg['batch_size'], lr=cfg['learning_rate'],
                weight_decay=cfg['weight_decay'], warmup_pct=cfg['warmup_fraction'])
            # Native training_step computes the loss; this runner owns logging.
            self.net.log = lambda *args, **kwargs: None
            if self.name == 'PARSeq':
                self.net.rng = np.random.default_rng(cfg['seed'])
        elif self.name in ['SVTR', 'SVTRv2']:
            from openrec.modeling import build_model
            arch = copy.deepcopy(cfg['architecture'])
            arch['Decoder']['out_channels'] = len(self.charset) + 1
            if self.name == 'SVTR':
                arch['Encoder']['img_size'] = cfg['image_size']
                arch['Encoder']['out_char_num'] = cfg['image_size'][1] // 4
            self.net = build_model(arch)
        elif self.name == 'SATRN':
            from satrn_vendor.base import Dictionary
            from satrn_vendor.shallow_cnn import ShallowCNN
            from satrn_vendor.satrn_encoder import SATRNEncoder
            from satrn_vendor.nrtr_decoder import NRTRDecoder
            self.backbone = ShallowCNN(input_channels=3, hidden_dim=512)
            self.encoder = SATRNEncoder(n_layers=12, n_head=8, d_k=64, d_v=64,
                                       d_model=512, d_inner=2048, dropout=0.1)
            self.decoder = NRTRDecoder(n_layers=6, d_embedding=512, n_head=8,
                                       d_model=512, d_inner=2048, d_k=64, d_v=64,
                                       dictionary=Dictionary(self.charset),
                                       max_seq_len=self.max_length + 1)
            self.backbone.init_weights()
            self.encoder.init_weights()
            self.decoder.init_weights()
        elif self.name == 'TrOCR':
            from transformers import (DeiTConfig, DeiTModel, TrOCRConfig,
                                      TrOCRForCausalLM, VisionEncoderDecoderModel)
            enc_cfg = DeiTConfig(image_size=cfg['image_size'][0], patch_size=16,
                                 hidden_size=384, num_hidden_layers=12,
                                 num_attention_heads=6, intermediate_size=1536)
            dec_cfg = TrOCRConfig(vocab_size=len(self.charset)+3, d_model=256,
                                 decoder_layers=6, decoder_attention_heads=8,
                                 decoder_ffn_dim=1024, max_position_embeddings=512,
                                 pad_token_id=self.pad, bos_token_id=self.bos,
                                 eos_token_id=self.eos, decoder_start_token_id=self.bos,
                                 is_decoder=True, add_cross_attention=True,
                                 use_cache=False)
            self.net = VisionEncoderDecoderModel(
                encoder=DeiTModel(enc_cfg, add_pooling_layer=False),
                decoder=TrOCRForCausalLM(dec_cfg))
            self.net.config.decoder_start_token_id = self.bos
            self.net.config.pad_token_id = self.pad
            self.net.config.eos_token_id = self.eos
        else:
            raise ValueError(f'Unknown model: {self.name}')

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Lightning tracks device metadata in addition to Tensor devices.
        # Parent nn.Module.to() alone does not update a nested LightningModule.
        if self.name in ['PARSeq', 'ViTSTR', 'CRNN']:
            self.net.to(*args, **kwargs)
        return self

    def targets(self, labels, device):
        seq = [[self.bos] + [self.char_ids[c] for c in s] + [self.eos] for s in labels]
        out = torch.full((len(seq), max(map(len, seq))), self.pad, dtype=torch.long, device=device)
        for i, values in enumerate(seq):
            out[i, :len(values)] = torch.tensor(values, device=device)
        return out

    def loss(self, images, labels):
        if self.name in ['PARSeq', 'ViTSTR', 'CRNN']:
            if self.name == 'CRNN':
                logits, loss, _ = self.net.forward_logits_loss(images, labels)
                self.check_ctc(labels, logits.size(1))
                return loss
            return self.net.training_step((images, labels), 0)
        if self.name in ['SVTR', 'SVTRv2']:
            logits = self.net(images)
            self.check_ctc(labels, logits.size(1))
            target = torch.tensor([self.char_ids[c]+1 for s in labels for c in s],
                                  device=images.device, dtype=torch.long)
            lengths = torch.tensor([len(s) for s in labels], dtype=torch.long)
            input_lengths = torch.full((len(labels),), logits.size(1), dtype=torch.long)
            return F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1), target,
                              input_lengths, lengths, blank=0, zero_infinity=False)
        target = self.targets(labels, images.device)
        if self.name == 'SATRN':
            memory = self.encoder(self.backbone(images))
            hidden = self.decoder._attention(target[:, :-1], memory)
            logits = self.decoder.classifier(hidden)
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   target[:, 1:].reshape(-1), ignore_index=self.pad)
        labels_t = target[:, 1:].clone()
        labels_t[labels_t == self.pad] = -100
        return self.net(pixel_values=images, labels=labels_t).loss

    @staticmethod
    def check_ctc(labels, steps):
        for text in labels:
            needed = len(text) + sum(a == b for a,b in zip(text,text[1:]))
            if needed > steps:
                raise ValueError(f'CTC requires {needed} time steps for {text!r}, only {steps} available; increase image width.')

    def strings(self, ids):
        result = []
        for row in ids.tolist():
            chars = []
            for value in row:
                if value == self.eos:
                    break
                if value < len(self.charset):
                    chars.append(self.charset[value])
            result.append(''.join(chars))
        return result

    @torch.no_grad()
    def predict(self, images):
        if self.name in ['PARSeq', 'ViTSTR', 'CRNN']:
            return self.net.tokenizer.decode(self.net(images).softmax(-1))[0]
        if self.name in ['SVTR', 'SVTRv2']:
            result = []
            for row in self.net(images).argmax(-1).tolist():
                chars, prev = [], None
                for value in row:
                    if value != prev and value != 0:
                        chars.append(self.charset[value-1])
                    prev = value
                result.append(''.join(chars))
            return result
        ids = torch.full((images.size(0),1), self.bos, device=images.device, dtype=torch.long)
        done = torch.zeros(images.size(0),device=images.device,dtype=torch.bool)
        if self.name == 'SATRN':
            memory = self.encoder(self.backbone(images))
        else:
            encoder_output = self.net.encoder(pixel_values=images)
        for _ in range(self.max_length + 1):
            if self.name == 'SATRN':
                logits = self.decoder.classifier(self.decoder._attention(ids, memory))
            else:
                logits = self.net(encoder_outputs=encoder_output,
                                  decoder_input_ids=ids, use_cache=False).logits
            # BOS/PAD are input-only tokens, never valid output characters.
            logits = logits[:, -1].clone()
            logits[:, self.bos] = -torch.inf
            logits[:, self.pad] = -torch.inf
            nxt = logits.argmax(-1)
            nxt = torch.where(done, self.eos, nxt)
            ids = torch.cat([ids,nxt[:,None]],dim=1)
            done |= nxt == self.eos
            if done.all():
                break
        return self.strings(ids[:,1:])
