"""Explicit stage commands. Production generation/training require later readiness receipts."""
from .common import run_path, read, event, StageLock


def main(args):
    run = run_path(args.run)
    if args.stage in ('timed-pilot','timed-pilot-worker'):
        from .timed_pilot import supervise,run as worker
        (supervise if args.stage=='timed-pilot' else worker)(run)
        return
    if args.stage == 'profile-stopped':
        import json
        from .performance_probe import measure
        print(json.dumps(measure(run)))
        return
    if args.stage in ('status','supervise'):
        import json
        from .supervision import inspect
        print(json.dumps(inspect(run,save=args.stage=='supervise')))
        return
    with StageLock(run, args.stage):
        event(run, args.stage, 'running')
        try:
            if args.stage == 'audit-dev-gallery':
                from .dev_gallery_data import audit_gallery
                result = audit_gallery(run)
            elif args.stage == 'dev-gallery-reader-tests':
                from .dev_gallery_data import check_reader
                result = check_reader(run)
            elif args.stage == 'dev-gallery-tests':
                from .dev_gallery import check_gallery
                result = check_gallery(run,args.workers)
            elif args.stage == 'generate-dev-gallery':
                from .dev_gallery import generate_gallery
                result = generate_gallery(run,args.workers)
            elif args.stage == 'joint-phase-tests':
                from .joint_phase_checks import check_joint_phase
                result = check_joint_phase(run)
            elif args.stage == 'reference-sampling-tests':
                from .reference_sampling_checks import check_reference_sampling
                result = check_reference_sampling(run)
            elif args.stage == 'font-training-tests':
                from .font_training_checks import check_font_training
                result = check_font_training(run)
            elif args.stage == 'observe-training':
                from .runtime_observer import observe
                result = observe(run,args.count)
            elif args.stage == 'migrate-storage':
                from .storage_migration import migrate
                result = migrate(run)
            elif args.stage == 'amp-recovery-tests':
                from .amp_recovery_checks import check_amp_recovery
                result = check_amp_recovery(run)
            elif args.stage == 'joint-evaluation-tests':
                from .joint_evaluation_checks import check_joint_evaluation
                result = check_joint_evaluation(run)
            elif args.stage == 'font-extraction-tests':
                from .font_extraction_checks import check_font_extraction
                result = check_font_extraction(run)
            elif args.stage == 'phase-engine-tests':
                from .phase_engine_checks import check_phase_engine
                result = check_phase_engine(run)
            elif args.stage == 'specialist-selection-tests':
                from .selection_checks import check_selection
                result = check_selection(run)
            elif args.stage == 'reference-data-tests':
                from .reference_data import check_reference_data
                result = check_reference_data(run)
            elif args.stage == 'specialist-contract-tests':
                from .specialist_checks import check_specialists
                result = check_specialists(run)
            elif args.stage == 'acquire-dinov2':
                from .dinov2_assets import acquire
                result = acquire(run)
            elif args.stage == 'training-tests':
                from .training import check_training
                result = check_training(run)
            elif args.stage == 'train-baseline':
                if (run/'amp_recovery/amendment.json').exists():
                    from .amp_recovery import resume
                    result = resume(run,args.workers)
                else:
                    from .training import train_baseline
                    result = train_baseline(run,args.workers)
            elif args.stage == 'acquire-fonts':
                from .fonts import acquire
                result = acquire(run, args.workers)
            elif args.stage == 'archive-fonts':
                from .fonts import archives
                result = archives(run)
            elif args.stage == 'historical-fonts':
                from .fonts import historical
                result = historical(run)
            elif args.stage == 'distro-fonts':
                from .distro_fonts import acquire
                result = acquire(run)
            elif args.stage == 'index-fonts':
                from .fonts import index
                result = index(run)
            elif args.stage == 'renderer-tests':
                from .checks import check_renderer
                result = check_renderer(run)
            elif args.stage == 'pilot':
                from .pilot import generate
                result = generate(run, args.count, args.workers)
            elif args.stage == 'checkpoint-tests':
                from .checkpoints import check_resume
                result = check_resume(run)
            elif args.stage == 'storage-tests':
                from .production import check_storage
                result = check_storage(run,args.workers)
            elif args.stage == 'data-tests':
                from .data_checks import check_data
                result = check_data(run)
            elif args.stage == 'checkpoint-cuda-tests':
                from .checkpoints import check_cuda_resume
                result = check_cuda_resume(run)
            elif args.stage == 'objective-tests':
                from .objective_checks import check_objectives
                result = check_objectives(run)
            elif args.stage == 'audit-corpus':
                from .corpus_audit import audit
                result = audit(run)
            elif args.stage == 'affinity-tests':
                from .affinity_checks import check_affinity
                result = check_affinity(run)
            elif args.stage == 'diagnose-render-failure':
                from .recovery import diagnose
                result = diagnose(run)
            elif args.stage == 'recovery-tests':
                from .recovery import check_recovery
                result = check_recovery(run)
            elif args.stage == 'model-tests':
                from .model_checks import check_model
                result = check_model(run)
            elif args.stage == 'reference-storage-tests':
                from .reference_generation import check_reference_storage
                result = check_reference_storage(run)
            elif args.stage == 'generate-font-references':
                if (run/'reference_recovery/amendment.json').exists():
                    from .reference_recovery import resume
                    result = resume(run,args.workers)
                else:
                    from .reference_generation import generate_references
                    result = generate_references(run,args.workers)
            elif args.stage == 'reference-recovery-tests':
                from .reference_recovery import check_recovery
                result = check_recovery(run)
            elif args.stage == 'audit-font-references':
                from .reference_audit import audit_references
                result = audit_references(run)
            elif args.stage == 'reference-diagnose':
                from .reference_recovery import diagnose
                result = diagnose(run)
            elif args.stage == 'reference-pilot':
                from .reference_render import reference_pilot
                result = reference_pilot(run,args.count,args.workers)
            elif args.stage == 'retrieval-tests':
                from .retrieval_checks import check_retrieval
                result = check_retrieval(run)
            elif args.stage == 'prepare-font-references':
                from .specimens import prepare_train_references
                result = prepare_train_references(run)
            elif args.stage == 'appearance-tests':
                from .appearance_checks import check_appearance
                result = check_appearance(run)
            elif args.stage == 'model-resume-tests':
                from .model_resume_checks import check_model_resume
                result = check_model_resume(run)
            elif args.stage == 'prepare-corpus':
                from .readiness import prepare
                result = prepare(run,args.count)
            elif args.stage == 'generate-corpus':
                from .common import sha
                protocol = read(run / 'corpus_protocol.json')
                for path, digest in protocol.get('data_asset_hashes', {}).items():
                    if sha(path) != digest:
                        raise ValueError('Frozen data asset changed: ' + path)
                from .production import generate
                result = generate(run,args.workers)
            else:
                raise ValueError(args.stage)
            event(run, args.stage, 'completed', result=result)
        except Exception as exc:
            if args.stage == 'generate-corpus' and (run/'corpus/progress.json').exists():
                from .common import write, now
                progress = read(run/'corpus/progress.json')
                progress.update(state='failed', stopped_at=now(), error=repr(exc)[:400])
                write(run/'corpus/progress.json', progress)
            event(run, args.stage, 'failed', error=repr(exc))
            raise
