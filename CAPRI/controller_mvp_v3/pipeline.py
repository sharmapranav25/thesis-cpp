from boot import ROOT,U
import subprocess,sys

def main():
    for family in ['Outlined','CryptBB','City','DarkBay']:
        U.dump(ROOT/'pipeline_status.json',dict(stage='running',family=family))
        with (ROOT/(family+'_stdout.log')).open('w') as stdout,(ROOT/(family+'_stderr.log')).open('w') as stderr:
            result=subprocess.run([sys.executable,str(ROOT/'experiment3.py'),family],stdout=stdout,stderr=stderr)
        if result.returncode:
            U.dump(ROOT/'pipeline_status.json',dict(stage='failed',family=family,exit_code=result.returncode));raise RuntimeError(family+' failed')
    U.dump(ROOT/'pipeline_status.json',dict(stage='complete'))

if __name__=='__main__':main()
