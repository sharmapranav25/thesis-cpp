"""Low-rate read-only telemetry of an existing trainer; never loads a model."""
import statistics
import subprocess
import time
from pathlib import Path
import psutil
from .common import ROOT,read,write,now


def observe(destination,seconds=30):
    if not 5<=seconds<=60:raise ValueError('Observer duration must be 5..60 seconds')
    source=ROOT/read(ROOT/'configs/local.json')['decoder_run'];destination=Path(destination)
    if destination.resolve().is_relative_to(source.resolve()):raise ValueError('Keep observer writes outside the migrating run')
    lock=read(source/'train-baseline.lock.json');trainer=psutil.Process(lock['pid'])
    if abs(trainer.create_time()-lock['process_created'])>=1:raise ValueError('Trainer identity mismatch')
    processes=[trainer]+trainer.children(recursive=True)
    def snapshot(process):
        with process.oneshot():
            cpu=process.cpu_times();io=process.io_counters()
            return dict(pid=process.pid,cpu_seconds=cpu.user+cpu.system,read_bytes=io.read_bytes,
                        write_bytes=io.write_bytes,rss=process.memory_info().rss)
    before={p.pid:snapshot(p) for p in processes};progress_before=read(source/'training/baseline_localization/progress.json')
    samples=[];start=time.monotonic();reason='duration_complete'
    while time.monotonic()-start<seconds:
        if not trainer.is_running():reason='trainer_exited';break
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.sm',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        if result.returncode:raise RuntimeError('GPU telemetry query failed')
        values=[float(x.strip()) for x in result.stdout.strip().splitlines()[0].split(',')]
        samples.append(dict(elapsed=time.monotonic()-start,gpu_percent=values[0],memory_controller_percent=values[1],
                            memory_used_MiB=values[2],power_W=values[3],sm_clock_MHz=values[4]))
        time.sleep(min(1.,max(0.,seconds-(time.monotonic()-start))))
    elapsed=time.monotonic()-start;usage=[]
    for p in processes:
        try:after=snapshot(p)
        except psutil.NoSuchProcess:continue
        old=before[p.pid];usage.append(dict(pid=p.pid,role='trainer' if p.pid==trainer.pid else 'loader',
            mean_cpu_cores=(after['cpu_seconds']-old['cpu_seconds'])/elapsed,
            read_MiB_per_second=(after['read_bytes']-old['read_bytes'])/elapsed/2**20,
            write_MiB_per_second=(after['write_bytes']-old['write_bytes'])/elapsed/2**20,rss_MiB=after['rss']/2**20))
    progress_after=read(source/'training/baseline_localization/progress.json')
    report=dict(created_at=now(),source=str(source),trainer_pid=trainer.pid,elapsed_seconds=elapsed,stop_reason=reason,
        process_usage=usage,gpu_samples=samples,progress_before=progress_before,progress_after=progress_after,
        gpu_mean_percent=statistics.mean(s['gpu_percent'] for s in samples) if samples else None,
        gpu_median_percent=statistics.median(s['gpu_percent'] for s in samples) if samples else None,
        limitation='Coarse driver/process telemetry, not an operator profiler or causal bottleneck diagnosis. Periodic GPU samples can alias the training cycle; process CPU/read counters include runtime and filesystem caching. No training state changed.')
    write(destination/'runtime_observation.json',report)
    return {k:v for k,v in report.items() if k not in ('gpu_samples','progress_before','progress_after')}
