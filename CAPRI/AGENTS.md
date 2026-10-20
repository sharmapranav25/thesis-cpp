# Workspace maintenance

- For shoot-rate policy development, learn character confusions and attempt/skip rules on synthetic validation data, freeze the policy and real attempt decisions, then use real labels only to score. Do not choose thresholds by sweeping real test accuracy. Keep historical real-test confidence sweeps explicitly distinguished from this synthetic-derived procedure.

- For user-requested DS2 status updates, include real DarkBay accuracy for the latest fully saved checkpoint and compare with DS1 at the same epoch and seed (correct/900, accuracy, difference). Reuse saved interim evaluations; if missing, run `tools/evaluate_ds2_checkpoint.py --seed <seed> --epoch <epoch>` without changing ongoing training. Clearly distinguish the in-progress epoch from the tested completed epoch. These requested diagnostics must not influence checkpoint selection or curriculum. Automated heartbeat checks remain quiet on ordinary progress.

- Read `README.md` and `docs/CODE_AND_MODELS.md` before changing the active controller workflow.
- Maintain active code in `src/capri_controller`, selected models in `models/registry.json`, and machine paths in `configs/local.json`. Use `capri.py` for new runs.
- Put new run outputs under `runs/<descriptive-name>`. Preserve original experiment artifacts and frozen release packages unless the user explicitly asks to change them.
- After code, model-selection or location changes, add an entry to `docs/MAINTENANCE_LOG.md` explaining what changed, why, validation and remaining limits. Refresh the workspace catalog with `capri.py inventory` when relevant.
- Target input is pixels only. Do not introduce VLMs, human target strings or manually chosen target-family lengths into this workflow. Retain the three-controller mixture and provenance unless the user changes the design.
