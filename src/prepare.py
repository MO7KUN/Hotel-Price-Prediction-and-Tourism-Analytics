"""Normalize Spanish listing fields and retain single-room search quotes."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, urlunsplit
import numpy as np
import pandas as pd
from .collect import FILENAME

AMENITIES = {
    'mentions_wifi':r'\bwi[ -]?fi\b', 'mentions_pool':r'\bpiscinas?\b',
    'mentions_spa':r'\bspa\b', 'mentions_fitness':r'\b(gimnasio|fitness)\b',
    'mentions_parking':r'\b(parking|aparcamiento)\b', 'mentions_restaurant':r'\brestaurantes?\b',
    'mentions_air_conditioning':r'\baire acondicionado\b',
    'mentions_breakfast':r'\bdesayunos?\b', 'mentions_pets':r'\b(mascotas?|pet[ -]friendly)\b'}
NUMERIC=['adults','distance_center_km','distance_beach_km','beach_distance_missing',
    'log_review_count','sustainability_certification','description_missing']+list(AMENITIES)
CATEGORICAL=['district']
FEATURES=NUMERIC+CATEGORICAL
CLUSTER_FEATURES=['log_median_price','distance_center_km','log_review_count','amenity_mentions_count']

def plain(value):
    if pd.isna(value):return ''
    return ''.join(c for c in unicodedata.normalize('NFKD',str(value).lower()) if not unicodedata.combining(c))

def parse_eur(value):
    text=str(value).replace('\u00a0',' ').strip()
    if '€' not in text:
        raise ValueError(f'Expected explicit EUR price: {text!r}')
    if '-' in text or '−' in text:raise ValueError('Negative price is invalid')
    token=re.sub(r'[^\d.,]','',text)
    # The source uses Spanish grouping: 1.234,50 or 1.234; comma is decimal.
    if ',' in token:token=token.replace('.','').replace(',','.')
    elif re.fullmatch(r'\d{1,3}(?:\.\d{3})+',token):token=token.replace('.','')
    amount=float(token)
    if not np.isfinite(amount) or amount<=0:raise ValueError('Price must be positive and finite')
    return amount

def parse_distance(value):
    text=plain(value).replace(',','.')
    match=re.search(r'(\d+(?:\.\d+)?)\s*(km|m)\b',text)
    if match is None:return np.nan
    return float(match[1])/(1000 if match[2]=='m' else 1)

def parse_reviews(value):
    match=re.search(r'([\d.,\s]+)\s*comentarios',str(value),flags=re.I)
    if match is None:return np.nan
    return float(re.sub(r'\D','',match[1]))

def url_fields(url):
    split=urlsplit(url)
    if split.hostname!='www.booking.com' or not split.path.startswith('/hotel/'):
        raise ValueError('Expected a Booking.com hotel URL')
    query=parse_qs(split.query)
    canonical=urlunsplit(('https','www.booking.com',split.path,'',''))
    get=lambda name:query.get(name,[None])[0]
    return {'property_id':hashlib.sha256(split.path.encode()).hexdigest()[:16],
        'canonical_url':canonical,'checkin':get('checkin'),'checkout':get('checkout'),
        'url_adults':get('group_adults'),'url_rooms':get('no_rooms'),'url_children':get('group_children'),
        'child_ages':','.join(query.get('age',[]))}

def prepare(path=None):
    source=pd.read_csv(path or Path('data/raw')/FILENAME)
    source.columns=[plain(c) for c in source.columns]
    urls=pd.DataFrame(source.link.map(url_fields).tolist())
    data=pd.concat([source.reset_index(drop=True),urls],axis=1)
    data['checkin']=pd.to_datetime(data.checkin,errors='raise')
    data['checkout']=pd.to_datetime(data.checkout,errors='raise')
    data['nights']=(data.checkout-data.checkin).dt.days
    data['adults']=pd.to_numeric(data.adultos,errors='raise')
    data['rooms']=pd.to_numeric(data.habitaciones,errors='raise')
    data['children']=pd.to_numeric(data.ninos,errors='raise')
    for column,url_column in [('adults','url_adults'),('children','url_children')]:
        if not data[column].eq(pd.to_numeric(data[url_column],errors='raise')).all():
            raise ValueError(f'Source versus URL occupancy mismatch: {column}')
    room_mismatch=~data.rooms.eq(pd.to_numeric(data.url_rooms,errors='raise'))
    excluded_room_mismatch=int(room_mismatch.sum())
    data=data.loc[~room_mismatch].copy()
    if not data.nights.gt(0).all():raise ValueError('Nonpositive stay duration')
    data['quoted_price_eur']=data.precio.map(parse_eur)
    data['price_per_night_eur']=data.quoted_price_eur/data.nights
    data['distance_center_km']=data.distancia_centro.map(parse_distance)
    data['distance_beach_km']=data.distancia_playa.map(parse_distance)
    data['beach_distance_missing']=data.distance_beach_km.isna().astype(int)
    data['review_count']=data.comentarios.map(parse_reviews)
    data['log_review_count']=np.log1p(data.review_count)
    data['district']=data.ubicacion.str.replace(r',\s*Barcelona\s*$','',regex=True).str.strip()
    data['name']=data.nombre
    data['sustainability_certification']=data.certificaciones.map(lambda x:int('sostenibilidad' in plain(x)))
    data['description_missing']=data.descripcion.isna().astype(int)
    descriptions=data.descripcion.map(plain)
    for name,pattern in AMENITIES.items():
        data[name]=descriptions.map(lambda text:int(bool(re.search(pattern,text))))
    data['amenity_mentions_count']=data[list(AMENITIES)].sum(axis=1)
    keys=['property_id','checkin','checkout','adults','rooms','children']
    duplicates=int(data.duplicated(keys).sum())
    data=data.drop_duplicates(keys).copy()
    all_rows=len(data)
    # Requested multi-room counts do not prove the number of rooms in the displayed offer.
    # Some one-adult two-room searches have the same quote as one-room searches.
    # Never divide by requested rooms. Restrict the model to one-room searches instead.
    excluded=int(data.rooms.ne(1).sum())
    data=data.loc[data.rooms.eq(1)].copy()
    assert not data.duplicated(keys).any()
    assert data.price_per_night_eur.gt(0).all()
    assert set(FEATURES).isdisjoint({'price_per_night_eur','quoted_price_eur','name','property_id','canonical_url'})
    audit={'source_rows':len(source),'excluded_room_count_mismatch':excluded_room_mismatch,
        'duplicate_search_rows_removed':duplicates,'excluded_multiroom_rows':excluded,
        'retained_rows':len(data),'properties':int(data.property_id.nunique()),'districts':int(data.district.nunique()),
        'checkin_dates':sorted(data.checkin.dt.strftime('%Y-%m-%d').unique().tolist()),
        'checkout_dates':sorted(data.checkout.dt.strftime('%Y-%m-%d').unique().tolist()),
        'adults':sorted(data.adults.unique().tolist()),'children':sorted(data.children.unique().tolist()),
        'rooms':sorted(data.rooms.unique().tolist()),'nights':sorted(data.nights.unique().tolist()),
        'child_ages':sorted(data.child_ages.unique().tolist()),
        'price_summary_eur':data.price_per_night_eur.describe(percentiles=[.05,.25,.5,.75,.95]).to_dict(),
        'missing_percent':(100*data[FEATURES].isna().mean()).round(3).to_dict(),
        'unavailable_features':['official_star_category','guest_review_score','room_type','tax_inclusion','cancellation_policy','multiple_stay_dates'],
        'group_sizes':data.groupby('property_id').size().value_counts().sort_index().to_dict()}
    columns=['property_id','name','canonical_url','checkin','checkout','nights','rooms','children','child_ages',
        'quoted_price_eur','price_per_night_eur','review_count','amenity_mentions_count']+FEATURES
    output=Path('data/processed'); output.mkdir(parents=True,exist_ok=True)
    data[columns].to_csv(output/'hotel_quotes.csv',index=False)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit,indent=2,ensure_ascii=False),flush=True)
    return data[columns]

if __name__=='__main__':prepare()
