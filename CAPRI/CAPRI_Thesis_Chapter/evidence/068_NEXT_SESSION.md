# CAPRI v2 handoff

User explicitly requested font variety plus Meta-Sim-inspired refinement packaged together. Start README.md and run verify. Default similar-font top3 weighted sampling, family-specific refinement auto-enabled for exact five hybrid cached profiles. Prior CAPRI package is historical baseline. No neural classifiers were retrained for this change. Existing three-seed jobs finished before refinement trial; verify actual monitors before starting new work.

Important: adapters trained on single-font recipes. Combination with variety has smoke/invariant checks only; downstream decoding improvement remains untested. No cross-family adapter reuse. New-family profiling does CV and learned inference, but no adapter is fitted automatically. Exact family/profile/kind match required for refinement.

No runtimeVLM, no target generator/fonts/labels. Preserve original/vendor/history, realfit/evaluation boundaries and truthful gates. Papers/failed outcomes in docs and canonical thesis logs. User stopped continuous loop earlier; do not restart automation. Length/alphabet remain chosen synthetic buckets, not inferred real characteristics. Production use and PARSeq training are not authorized by package creation alone.
