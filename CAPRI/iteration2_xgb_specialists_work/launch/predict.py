"""Run only the exported feature encoder and its fitted XGBoost heads."""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from PIL import Image
from feature_bank.iter2 import data as D
from feature_bank.iter2.generate import TEXT_HEADS, effect_heads
if __package__:
    from .feature_encoder import load_encoder
    from .fit_xgb import boosted, decode_font, load_models, read
else:
    from feature_encoder import load_encoder
    from fit_xgb import boosted, decode_font, load_models, read


class XGBoostDescriber:
    def __init__(self, bundle, device=None):
        self.bundle = Path(bundle).resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder, self.metadata = load_encoder(self.bundle/'encoder.pt', self.device)
        self.tasks = read(self.bundle/'heads.json')['tasks']
        assert len(self.tasks) == 107, 'Booster fitting is incomplete'
        self.models = load_models(self.bundle.parent, self.tasks, bundle=self.bundle)
        self.lookup = {task['name']: task for task in self.tasks}

    @torch.inference_mode()
    def predict(self, image_path):
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            width = max(8, min(960, round(image.width*96/image.height)))
            pixels = np.asarray(image.resize((width, 96), Image.Resampling.LANCZOS), dtype=np.uint8)
        columns = (width+7)//8
        padded = ((columns*8+D.MARGIN+7)//8)*8
        tensor = torch.from_numpy(np.pad(pixels, ((0, 0), (0, padded-width), (0, 0)), mode='edge')
                                 .transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device).float()/255
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
            encoded = self.encoder(tensor, torch.tensor([columns], device=self.device))
        features = {key: value.float().cpu().numpy().astype(np.float16) for key, value in encoded.items()}
        values = boosted(features, self.tasks, self.models)
        result = dict(source=str(Path(image_path).resolve()), effects={}, text_features={})
        for group, names, block in [('eff', effect_heads(), 'effects'), ('txt', TEXT_HEADS, 'text_features')]:
            for name in names:
                key = f'{group}_presence_{name}'
                probability = float(values[key][0])
                threshold = self.lookup[key]['threshold']
                ordinal = values.get(f'{group}_level_{name}')
                result[block][name] = dict(probability=probability, present=probability >= threshold,
                    level=int(ordinal[0]) if ordinal is not None else None, threshold=threshold)
        result.update(font=decode_font(values), length=int(values['length'][0]), neural_feature_heads_used=False)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out')
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = XGBoostDescriber(args.bundle, args.device).predict(args.image)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
