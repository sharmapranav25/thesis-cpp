"""Checkpoint-boundary CAPRI relocation; retain logical paths and verify bytes."""
import json
import os
import subprocess
import time
from pathlib import Path
import psutil
from .common import ROOT,read,write,sha,now

DESTINATION=Path('E:/pranav/CAPRI/runs/capri_decoder_500k_20261007')
JOURNAL=ROOT/'runs/capri_storage_migration_20261007'


def migrate(run):
    run=Path(run);expected=ROOT/'runs/capri_decoder_500k_20261007'
    if run!=expected or run.resolve()!=expected or DESTINATION.resolve()!=DESTINATION:
        raise ValueError('Migration requires the original unlinked C run and exact E destination')
    plan=read(JOURNAL/'plan.json')
    if plan['source']!=str(expected) or plan['destination']!=str(DESTINATION):raise ValueError('Migration path mismatch')
    if plan['minimum_checkpoint'] not in (10000,15000):raise ValueError('Unexpected checkpoint boundary')
    def report(state,**fields):write(JOURNAL/'progress.json',dict(updated_at=now(),pid=os.getpid(),state=state,**fields))
    def ps(action):
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                        str(Path(__file__).with_name('storage_cutover.ps1')),'-Action',action],check=True)
    try:
        report('waiting_for_checkpoint',minimum_checkpoint=plan['minimum_checkpoint'])
        while True:
            pointer=read(run/'training_recovery/baseline_localization/latest.json')
            if pointer['epoch']>=plan['minimum_checkpoint']:break
            lock=read(run/'train-baseline.lock.json')
            process=psutil.Process(lock['pid'])
            if abs(process.create_time()-lock['process_created'])>=1:raise RuntimeError('Trainer identity changed while waiting')
            time.sleep(2)
        checkpoint=Path(pointer['path'])
        if not checkpoint.resolve().is_relative_to(run.resolve()) or sha(checkpoint)!=pointer['sha256']:
            raise ValueError('Starting checkpoint failed verification')
        receipt=read(checkpoint.parent/'receipt.json')
        if receipt['sha256']!=pointer['sha256']:raise ValueError('Checkpoint receipt mismatch')
        lock=read(run/'train-baseline.lock.json')
        try:process=psutil.Process(lock['pid'])
        except psutil.NoSuchProcess:
            # Retry a failed, pre-switch migration only when this worker's earlier
            # journal proves it stopped this exact trainer at this checkpoint.
            stopped=read(JOURNAL/'stopped_trainer.json');ready=read(JOURNAL/'cutover_ready.json')
            if (stopped['lock']!=lock or stopped['checkpoint']!=pointer or ready['checkpoint']!=pointer
                or not ready['verified'] or ready['source']!=str(run) or ready['destination']!=str(DESTINATION)
                or ready['manifest_sha256']!=sha(JOURNAL/'verified_manifest.json')):
                raise RuntimeError('No verified prior migration stop for this trainer')
            report('resuming_quiescent_copy_verification',checkpoint=pointer)
        else:
            if abs(process.create_time()-lock['process_created'])>=1 or 'train-baseline' not in process.cmdline():
                raise RuntimeError('Refusing to stop an unverified trainer')
            report('stopping_at_verified_checkpoint',checkpoint=pointer,trainer_pid=process.pid)
            write(JOURNAL/'stopped_trainer.json',dict(checkpoint=pointer,lock=lock,progress=read(run/'training/baseline_localization/progress.json')))
            process.suspend();children=process.children(recursive=True)
            for child in children:child.terminate()
            process.kill();_,alive=psutil.wait_procs(children+[process],timeout=10)
            for child in alive:child.kill()
            psutil.wait_procs(alive,timeout=5)
        # The initial copy was non-destructive; now synchronize the quiescent run.
        report('final_copy',checkpoint=pointer)
        result=subprocess.run(['robocopy',str(run),str(DESTINATION),'/E','/COPY:DAT','/DCOPY:DAT','/R:2','/W:1','/MT:8',
                               '/XJ','/NFL','/NDL','/NP','/LOG:'+str(JOURNAL/'final_copy.log')])
        if result.returncode>=8:raise RuntimeError('Final copy failed: '+str(result.returncode))
        report('verifying_all_files',checkpoint=pointer)
        rows=[];files=sorted(p for p in run.rglob('*') if p.is_file())
        for i,path in enumerate(files):
            if path.is_symlink():raise ValueError('Unexpected source symlink')
            rel=path.relative_to(run);target=DESTINATION/rel;before=path.stat();source_hash=sha(path)
            if not target.is_file() or target.stat().st_size!=before.st_size or sha(target)!=source_hash:
                raise ValueError('Migration byte mismatch: '+str(rel))
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during verification')
            rows.append(dict(path=str(rel),bytes=before.st_size,sha256=source_hash))
            if i%500==0:report('verifying_all_files',files_verified=i+1,total_files=len(files),checkpoint=pointer)
        if {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()}!={r['path'] for r in rows}:
            raise RuntimeError('Source file set changed during verification')
        write(JOURNAL/'verified_manifest.json',dict(created_at=now(),source=str(run),destination=str(DESTINATION),files=rows))
        write(JOURNAL/'cutover_ready.json',dict(verified=True,source=str(run),destination=str(DESTINATION),
              manifest_sha256=sha(JOURNAL/'verified_manifest.json'),files=len(rows),bytes=sum(r['bytes'] for r in rows),checkpoint=pointer))
        report('switching_directory_link',checkpoint=pointer);ps('Switch')
        if run.resolve()!=DESTINATION or sha(checkpoint)!=pointer['sha256']:raise ValueError('Directory link/checkpoint verification failed')
        from .common import run_path
        if run_path(str(DESTINATION))!=run:raise ValueError('Physical path did not map to frozen logical run')
        report('restarting_from_checkpoint',checkpoint=pointer);ps('Resume')
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            progress=read(run/'training/baseline_localization/progress.json')
            if progress.get('state')=='failed':
                launch=read(JOURNAL/'restart.json')
                if progress.get('pid')!=lock['pid']:raise RuntimeError('Restarted trainer failed; C backup retained')
            new_lock=read(run/'train-baseline.lock.json') if (run/'train-baseline.lock.json').exists() else {}
            if (new_lock.get('pid')!=lock['pid'] and progress.get('pid')==new_lock.get('pid') and
                progress.get('state')=='training' and progress.get('optimizer_updates',0)>pointer['epoch']):
                current=psutil.Process(new_lock['pid'])
                if abs(current.create_time()-new_lock['process_created'])<1:break
            time.sleep(5)
        else:raise RuntimeError('Timed out verifying resumed trainer; C backup retained')
        write(JOURNAL/'resume_verified.json',dict(verified_at=now(),checkpoint=pointer,lock=new_lock,progress=progress,
                                                physical_root=str(run.resolve()),manifest_sha256=sha(JOURNAL/'verified_manifest.json')))
        # Only the redundant, byte-verified C copy is removed, by bounded native PowerShell.
        report('removing_verified_c_backup',checkpoint=pointer);ps('Cleanup')
        report('completed',checkpoint=pointer,physical_root=str(DESTINATION),logical_root=str(run),
               migrated_files=len(rows),migrated_bytes=sum(r['bytes'] for r in rows),trainer_pid=new_lock['pid'])
        return read(JOURNAL/'progress.json')
    except Exception as exc:
        report('failed',error=repr(exc),note='Copies preserved; inspect before retry. No automatic deletion on failure.');raise
