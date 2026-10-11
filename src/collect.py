"""Reproducible download of a versioned Booking.com research snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

RECORD = 'https://zenodo.org/api/records/17560799'
FILENAME = 'resultados_booking_20251108_175437.csv'

def collect(refresh=False):
    root=Path('data/raw'); root.mkdir(parents=True,exist_ok=True)
    manifest_path=Path('data/source_manifest.json')
    old=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    path=root/FILENAME
    if not path.exists() or refresh:
        metadata=requests.get(RECORD,timeout=45); metadata.raise_for_status()
        record=metadata.json()
        entry=next(f for f in record['files'] if f['key']==FILENAME)
        response=requests.get(entry['links']['self'],timeout=90); response.raise_for_status()
        content=response.content
        if 'md5:'+hashlib.md5(content).hexdigest()!=entry['checksum']:
            raise ValueError('Zenodo checksum mismatch')
        digest=hashlib.sha256(content).hexdigest()
        if old.get('sha256') and old['sha256']!=digest and not refresh:
            raise ValueError('Snapshot differs from the committed hash; use --refresh deliberately')
        temp=path.with_suffix('.part'); temp.write_bytes(content); temp.replace(path)
        manifest={'record_url':RECORD,'doi':record['metadata']['doi'],
            'title':record['metadata']['title'],'creators':record['metadata']['creators'],
            'license':record['metadata']['license']['id'], 'publication_date':record['metadata']['publication_date'],
            'download_url':entry['links']['self'],'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_md5':entry['checksum'],'sha256':digest,'bytes':len(content)}
    else:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if old.get('sha256') and old['sha256']!=digest:
            raise ValueError('Local raw snapshot differs from committed manifest')
        if not old:
            # A pre-existing manually supplied file must still be verified against Zenodo.
            return collect(refresh=True)
        manifest=old
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)
    return path

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--refresh',action='store_true')
    collect(parser.parse_args().refresh)
