"""Fixed bounded measurement queue, no training or intervention entry point."""
import sys,subprocess,json,socket,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from common import D,save,sha
from measure import synthetic_checks
seeds=[int(s) for s in sys.argv[1:]]
assert set(seeds).issubset({7,17,29,43,61}) and len(seeds)==len(set(seeds))
synthetic_checks()
for seed in seeds:
    i=1
    while (D/'logs'/f'measure_seed{seed}_attempt{i}.log').exists():i+=1
    log=D/'logs'/f'measure_seed{seed}_attempt{i}.log';log.parent.mkdir(exist_ok=True)
    with log.open('x') as f:
        result=subprocess.run([str(ROOT/'.venv-highres/bin/python'),'-u',str(Path(__file__).parent/'measure.py'),'--seed',str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        save(D/'workers'/f'{socket.gethostname()}.failed_attempt{i}.json',{'seed':seed,'returncode':result.returncode});raise SystemExit(result.returncode)
save(D/'workers'/f'{socket.gethostname()}.finished.json',{'seeds':seeds,'pid':os.getpid()})
