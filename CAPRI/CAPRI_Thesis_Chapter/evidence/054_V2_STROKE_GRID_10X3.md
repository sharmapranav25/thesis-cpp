# Stroke10x3v2: raw RGB multicolour trial

Explicit user request: retrain with10columns x3rows, no feature pop, multiple colours. Fresh synthetic seeds111m/112m/113m;2400 train,600 selection,600 locked images plus clean curriculum counterpart. No real fitting. Detector sees30 non-overlapping RGB patches; shared weights, not30 separate classifiers. Replicate padding extends128x64 to130x66; each patch13x22; logits stitched back and padding cropped. No extra1x1 input or postprocessing. Shape stage receives the stitched whole-image mask. Patch seams and lost long-range context are explicit risks.

Random full-range RGB backgrounds/text/strokes, both luminance polarities, colour gradients and rectangular hard negatives. Stroke colour separated from base colour; exact final visible synthetic target masks exclude negligible pixel changes. Four clean detector epochs then20 text-clutter epochs; shape16epochs. Old model/runs untouched. Synthetic masks supervise training only. Real inference uses predicted masks.

Same fixed gates: positiveIoU>=.70, absent false-positive rate<=.05, predicted-mask shape accuracy>=.85. Same threshold.5 and12pixel detection rule. Fresh font holdouts: first6train, first7selection, final2locked. No automatic promotion, no runtimeVLM, no external papers newly used. Changed architecture and colours together mean this is a requested combined experiment, not an isolated proof of grid benefit. Finite run, automation remains paused.


## Completed outcome

```json
{
  "selection": {
    "positive_iou": 0.32220473885536194,
    "none_false_positive_rate": 0.8299999833106995,
    "detection_recall": 0.9925000071525574,
    "oracle_shape_accuracy": 0.9149999618530273,
    "end_to_end_shape_accuracy": 0.5575000047683716,
    "all_class_accuracy": 0.4283333420753479
  },
  "locked": {
    "positive_iou": 0.295436829328537,
    "none_false_positive_rate": 0.8449999690055847,
    "detection_recall": 0.9899999499320984,
    "oracle_shape_accuracy": 0.9350000023841858,
    "end_to_end_shape_accuracy": 0.5249999761581421,
    "all_class_accuracy": 0.40166667103767395
  },
  "diagnostic_gate_pass": false,
  "production_promoted": false
}
```

No production promotion. This trial changes both view and colour distribution; old/new scores are not an isolated grid comparison.
