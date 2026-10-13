"""Destination identity: accept any Moroccan city and configured global cities."""
import json
import unicodedata
from pathlib import Path

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config/destinations.json').read_text(encoding='utf-8'))

def key(value):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFKD', str(value).strip().casefold())
                           if not unicodedata.combining(c)).split())

def destination(country, city):
    country = str(country).strip().upper()
    city = str(city).strip()
    if not city or city.casefold() in {'nan', 'none'}:
        raise ValueError('City must be provided')
    if country == 'MA':
        canonical = CONFIG['aliases'].get(key(city))
        canonical = canonical or next((c for c in CONFIG['morocco_shortcuts'] if key(c) == key(city)), city)
        return country, canonical
    match = next((d for d in CONFIG['global'] if d['country'] == country and key(d['city']) == key(city)), None)
    if match is None:
        raise ValueError(f'Destination is outside configured coverage: {country}/{city}')
    return country, match['city']
