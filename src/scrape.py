"""Optional Booking search HTML scraper. Stop on robots restrictions or blocks."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit,urlunsplit
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

USER_AGENT='HotelPriceResearch/1.0 (academic; single cached search request)'

def parse_search(html):
    soup=BeautifulSoup(html,'html.parser')
    cards=soup.select('[data-testid="property-card"]')
    if not cards:raise ValueError('No Booking property cards: JavaScript rendering, a block or a changed layout')
    rows=[]
    for card in cards:
        name=card.select_one('[data-testid="title"]')
        price=card.select_one('[data-testid="price-and-discounted-price"]')
        link=card.select_one('a[data-testid="title-link"]')
        if name is None or price is None or link is None:continue
        split=urlsplit(link.get('href',''))
        if split.hostname!='www.booking.com':continue
        rows.append({'name':name.get_text(' ',strip=True),'price_text':price.get_text(' ',strip=True),
            'canonical_url':urlunsplit(('https','www.booking.com',split.path,'',''))})
    if not rows:raise ValueError('No complete property/price/URL rows; refusing silent partial collection')
    return rows

def scrape(url):
    split=urlsplit(url)
    if split.scheme!='https' or split.hostname!='www.booking.com':raise ValueError('Only HTTPS Booking.com URLs are allowed')
    root=Path('data/raw/html');root.mkdir(parents=True,exist_ok=True)
    path=root/(hashlib.sha256(url.encode()).hexdigest()+'.html')
    Path('reports').mkdir(exist_ok=True)
    record={'attempted_at_utc':datetime.now(timezone.utc).isoformat(),'url':url,'user_agent':USER_AGENT}
    try:
        if path.exists():
            rows=parse_search(path.read_text(encoding='utf-8'));record['from_cache']=True
        else:
            session=requests.Session();session.headers['User-Agent']=USER_AGENT
            robot=session.get('https://www.booking.com/robots.txt',timeout=30);record['robots_status']=robot.status_code
            robot.raise_for_status()
            rules=RobotFileParser();rules.parse(robot.text.splitlines())
            allowed=rules.can_fetch(USER_AGENT,url);record['robots_allowed']=allowed
            if not allowed:raise PermissionError('robots.txt disallows this search; use the licensed published snapshot')
            time.sleep(max(3,rules.crawl_delay(USER_AGENT) or 0))
            response=session.get(url,timeout=45);record['http_status']=response.status_code
            response.raise_for_status()
            rows=parse_search(response.text)
            path.write_text(response.text,encoding='utf-8')
        record.update(status='success',rows=len(rows))
        Path('data/raw/live_search.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
        return rows
    except (requests.RequestException,ValueError,PermissionError) as error:
        record.update(status='unavailable',reason=str(error))
        raise
    finally:
        Path('reports/live_collection_attempt.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('url');args=parser.parse_args()
    try:print(f'Collected {len(scrape(args.url))} offers')
    except (requests.RequestException,ValueError,PermissionError) as error:
        parser.exit(2,f'{error}\nHistorical analysis uses the published Zenodo snapshot.\n')
