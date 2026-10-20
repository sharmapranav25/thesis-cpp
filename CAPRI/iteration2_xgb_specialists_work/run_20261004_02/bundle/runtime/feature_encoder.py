"""The trained iteration-2 encoder with every prediction head removed."""
import torch
import torch.nn.functional as F
from feature_bank.iter2.model import Iter2Model, token_mask
from feature_bank.train.model import col_mask

# CTC is a separate sequence reader, not an image feature head. Its weights are
# retained in an optional sidecar; the exported feature encoder runs no reader.
REMOVED = ('eff_head', 'txt_head', 'length', 'font_cat', 'font_stroke',
           'font_ord', 'font_italic', 'col_pool', 'ctc_up', 'ctc')


class Iter2FeatureEncoder(Iter2Model):
    def __init__(self, **config):
        super().__init__(**config)
        for name in REMOVED:
            delattr(self, name)

    def forward(self, image, cols, include_tokens=False):
        f = self.cnn(image, cols)
        batch = f.shape[0]
        e = F.avg_pool2d(self.eff_block(f, col_mask(cols, f.shape[-1], 4)), (4, 2))
        rows, columns = e.shape[-2:]
        mask = token_mask(cols, rows, columns)
        et = self.eff_norm(e.permute(0, 2, 3, 1) + self.eff_pos.view(1, rows, 1, -1))
        cnn_features = self.eff_pool(et.reshape(batch, rows * columns, -1), mask)
        t = F.avg_pool2d(f, (4, 2))
        x = self.proj(t.permute(0, 2, 3, 1)).reshape(batch, rows * columns, -1)
        x = x + self.width_emb(cols.clamp(max=self.max_cols)).unsqueeze(1)
        for block in self.blocks:
            x = block(x, mask, rows, columns)
        x = self.norm(x)
        result = dict(C=cnn_features, T=self.txt_pool(x, mask))
        if include_tokens:
            result.update(tokens=x, rows=rows, columns=columns, key_mask=mask)
        return result


def load_encoder(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = Iter2FeatureEncoder(**saved['config'])
    model.load_state_dict(saved['state_dict'], strict=True)
    model.requires_grad_(False)
    return model.to(device).eval(), saved
