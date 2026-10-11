"""Complete reproducible pipeline; run from the repository root."""
import argparse
from .collect import collect
from .prepare import prepare
from .train import train

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skip-download',action='store_true');args=parser.parse_args()
    if not args.skip_download:collect()
    prepare();train()
    from .report import report
    report()
