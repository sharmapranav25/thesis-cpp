from boot import ROOT,U
import html

def main():
    out=ROOT/'SimpleCaptcha_run_01';r=U.read(out/'results.json');inventory=U.read(ROOT/'simple_captcha_dataset/inventory.json')
    review=U.read(out/'visual_review.json') if (out/'visual_review.json').exists() else {'conclusion':'Visual review pending. No quality pass claimed.'}
    rows=[];scores=r['scores']['fresh'];base=scores['v2']['typography']
    for arm in ['v2','old','mask','typography','full']:
        s=scores[arm]['typography'];rows.append('<tr><td>'+arm+'</td>'+''.join(f'<td>{s[k]:.3f} ({100*(base[k]-s[k])/max(1e-10,base[k]):+.1f}%)</td>' for k in ['text','background','audit'])+f'<td>{s["usable"]:.1%}</td></tr>')
    content=f'''<!doctype html><meta charset="utf-8"><title>MVP v3 — Simple Captcha</title><style>body{{font:17px system-ui;max-width:1500px;margin:35px auto;padding:20px;background:#f8fafc;color:#182030}}img{{max-width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}p{{line-height:1.5}}</style>
    <h1>MVP v3: Simple Captcha</h1><p><strong>{html.escape(review['conclusion'])}</strong></p>
    <p>User-selected <a href="{inventory['source']}">Kaggle Simple Captcha</a>: {inventory['image_entries']} image entries, {inventory['unique_images']} unique pixel hashes. Only images were extracted. Target answers, CSV labels and generator code were not read; local image names are pixel hashes.</p>
    <p>No VLMs and no retraining for this dataset. Same mask checkpoint, descriptors, search phases and selection rules as the preceding MVP v3 tests. The original v1 and v2 stages establish the starting point. Fit 128, selection 64; final v3 holdout 256 images, disjoint from both preceding test sets. All controls frozen before final holdout scoring.</p>
    <table><tr><th>Arm</th><th>Text distance</th><th>Background distance</th><th>Independent audit</th><th>Synthetic usable</th></tr>{''.join(rows)}</table>
    <p>Positive percentages mean lower distance versus v2. These are appearance proxies, not OCR accuracy. Full v3 uses additional search budget. First eight held-out examples, without cherry-picking; generated strings differ from targets.</p><img src="fresh_preview.png"><h2>Predicted text masks</h2><p>Target / old mask / new mask. Real target ground truth masks are unknown.</p><img src="fresh_masks.png"><p><a href="results.json">Full results</a> · <a href="frozen_policies.json">Frozen controls</a> · <a href="audit.json">Verification</a> · <a href="history.json">Candidate history</a></p>'''
    (out/'report.html').write_text(content,encoding='utf-8')

if __name__=='__main__':main()
