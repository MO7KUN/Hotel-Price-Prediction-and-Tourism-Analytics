"""Train destination-specific snapshot models and export identical browser inference.

No city multipliers, synthetic prices, cross-market fallbacks or implicit FX rates.
The existing Barcelona research workflow is preserved independently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from .destinations import CONFIG, destination, key

FEATURES = ['adults', 'distance_center_km', 'log_review_count', 'amenity_mentions_count']
SCOPE = ['country', 'city_key', 'currency', 'checkin', 'checkout', 'rooms', 'children', 'child_ages']
REQUIRED = ['country', 'city', 'currency', 'property_id', 'checkin', 'checkout', 'rooms',
            'children', 'child_ages', 'adults', 'distance_center_km', 'review_count', 'amenity_mentions_count']

def normalize(frame, training=False):
    required = REQUIRED + (['price_per_night', 'source_url', 'source_license'] if training else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    data = frame.copy()
    for name in ['country', 'city', 'currency', 'property_id'] + (['source_url', 'source_license'] if training else []):
        if data[name].isna().any() or data[name].astype(str).str.strip().eq('').any():
            raise ValueError(f'Empty {name}')
    destinations = [destination(c, n) for c, n in zip(data.country, data.city)]
    data['country'] = [d[0] for d in destinations]
    data['city'] = [d[1] for d in destinations]
    data['city_key'] = data.city.map(key)
    data['currency'] = data.currency.str.strip().str.upper()
    if not data.currency.str.fullmatch('[A-Z]{3}').all():
        raise ValueError('Currency must be an explicit three-letter code; normalize FX upstream')
    for name in ['checkin', 'checkout']:
        dates = pd.to_datetime(data[name], format='%Y-%m-%d', errors='raise')
        if dates.isna().any():
            raise ValueError(f'Missing {name}')
        data[name] = dates.dt.strftime('%Y-%m-%d')
    if not (pd.to_datetime(data.checkout) > pd.to_datetime(data.checkin)).all():
        raise ValueError('Checkout must follow checkin')
    for name in ['adults', 'rooms', 'children', 'distance_center_km', 'review_count', 'amenity_mentions_count']:
        data[name] = pd.to_numeric(data[name], errors='raise')
        if not np.isfinite(data[name]).all() or (data[name] < 0).any():
            raise ValueError(f'Expected finite nonnegative {name}')
    for name in ['adults', 'rooms', 'children', 'review_count', 'amenity_mentions_count']:
        if not (data[name] == np.floor(data[name])).all():
            raise ValueError(f'Expected integer {name}')
        data[name] = data[name].astype(int)
    if not data.adults.ge(1).all() or not data.rooms.ge(1).all():
        raise ValueError('At least one adult and room are required')
    # Child ages are part of the quote contract, not an ignored input.
    data['child_ages'] = data.child_ages.fillna('').astype(str)
    def ages(value, count):
        numbers = [float(a) for a in value.split(',') if a.strip()]
        if any(not np.isfinite(a) or a != int(a) for a in numbers):
            raise ValueError('Child ages must be whole years')
        parsed = sorted(int(a) for a in numbers)
        if len(parsed) != count or any(a < 0 or a > 17 for a in parsed):
            raise ValueError('Child ages must match children and be between 0 and 17')
        return ','.join(map(str, parsed))
    data['child_ages'] = [ages(a, n) for a, n in zip(data.child_ages, data.children)]
    data['log_review_count'] = np.log1p(data.review_count)
    if training:
        data['price_per_night'] = pd.to_numeric(data.price_per_night, errors='raise')
        if not np.isfinite(data.price_per_night).all() or not data.price_per_night.gt(0).all():
            raise ValueError('Observed nightly room prices must be finite and positive')
        identifiers = ['country', 'city_key', 'property_id', 'checkin', 'checkout', 'rooms', 'adults', 'children', 'child_ages', 'currency']
        if data.duplicated(identifiers).any():
            raise ValueError('Duplicate property/quote configuration; resolve repeated captures upstream')
    return data

def scope_id(row):
    values = [str(int(row[k])) if k in {'rooms', 'children'} else str(row[k]) for k in SCOPE]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()[:20]

def build(path, output='.'):
    root = Path(output)
    data = normalize(pd.read_csv(path, dtype={'child_ages': str}), training=True)
    models, exports, coverage = {}, {}, []
    for _, group in data.groupby(SCOPE, dropna=False, sort=True):
        row = group.iloc[0]
        identifier = scope_id(row)
        record = {k: (row[k].item() if isinstance(row[k], np.generic) else row[k]) for k in SCOPE}
        record.update(id=identifier, city=row.city, quotes=len(group), properties=int(group.property_id.nunique()),
                      sources=group[['source_url', 'source_license']].drop_duplicates().to_dict('records'))
        properties = np.array(sorted(group.property_id.unique()))
        if len(properties) < 30:
            record.update(status='needs_data', reason='At least 30 distinct properties per stay configuration are required')
            coverage.append(record)
            continue
        np.random.default_rng(42).shuffle(properties)
        test_ids = properties[:max(6, int(len(properties) * .2))]
        train = group.loc[~group.property_id.isin(test_ids)]
        test = group.loc[group.property_id.isin(test_ids)]
        model = RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=2)
        model.fit(train[FEATURES], np.log1p(train.price_per_night))
        prediction = np.maximum(np.expm1(model.predict(test[FEATURES])), 0)
        baseline = np.full(len(test), train.price_per_night.median())
        domains = {name: [float(train[name].min()), float(train[name].max())] for name in FEATURES}
        record.update(status='ready', test_properties=len(test_ids), test_quotes=len(test),
                      train_property_ids=sorted(str(p) for p in train.property_id.unique()),
                      test_property_ids=sorted(str(p) for p in test.property_id.unique()),
                      mae=float(mean_absolute_error(test.price_per_night, prediction)),
                      baseline_mae=float(mean_absolute_error(test.price_per_night, baseline)),
                      domains=domains, adults=sorted(int(a) for a in train.adults.unique()))
        # Publish the exact evaluated estimator, without refitting on the test hotels.
        models[identifier] = model
        exports[identifier] = [{'left': t.tree_.children_left.tolist(), 'right': t.tree_.children_right.tolist(),
                               'feature': t.tree_.feature.tolist(), 'threshold': t.tree_.threshold.tolist(),
                               'value': t.tree_.value[:, 0, 0].tolist()} for t in model.estimators_]
        coverage.append(record)
    payload = {'schema_version': 1, 'sklearn_version': sklearn.__version__,
               'training_csv_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               'features': FEATURES, 'destinations': CONFIG, 'coverage': coverage, 'trees': exports,
               'note': 'Historical snapshot estimates; exact city, currency, dates and room/child configuration must be covered. Not live booking offers.'}
    for folder in ['artifacts', 'reports', 'dashboard']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    joblib.dump({'models': models, 'coverage': coverage}, root / 'artifacts/multicity.joblib', compress=3)
    (root / 'reports/destination_coverage.json').write_text(json.dumps({k:v for k,v in payload.items() if k != 'trees'}, ensure_ascii=False, indent=2), encoding='utf-8')
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    (root / 'dashboard/destinations-data.js').write_text('window.HOTEL_DESTINATIONS = ' + encoded + ';\n', encoding='utf-8')
    return payload

def predict(frame, artifacts='artifacts'):
    data = normalize(frame)
    bundle = joblib.load(Path(artifacts) / 'multicity.joblib')
    output = frame.copy()
    prices = []
    for _, row in data.iterrows():
        identifier = scope_id(row)
        record = next((r for r in bundle['coverage'] if r['id'] == identifier and r['status'] == 'ready'), None)
        if record is None:
            raise ValueError(f'No trained price coverage for {row.country}/{row.city}, {row.checkin}–{row.checkout}, {row.currency}, requested room/child configuration')
        for feature, (low, high) in record['domains'].items():
            if not low <= row[feature] <= high:
                raise ValueError(f'Input outside observed training range: {feature}')
        if row.adults not in record['adults']:
            raise ValueError('Adult count not observed during training')
        prices.append(max(float(np.expm1(bundle['models'][identifier].predict(pd.DataFrame([row[FEATURES]]))[0])), 0))
    output['prediction_per_night'] = prices
    output['prediction_currency'] = data.currency.to_numpy()
    return output

def seed():
    """Explicitly label the existing Barcelona observations; never relabel a city."""
    data = pd.read_csv('data/processed/hotel_quotes.csv', dtype={'child_ages': str})
    data['country'], data['city'], data['currency'] = 'ES', 'Barcelona', 'EUR'
    data['price_per_night'] = data.price_per_night_eur
    data['source_url'] = 'https://doi.org/10.5281/zenodo.17560799'
    data['source_license'] = 'CC-BY-NC-SA-4.0'
    path = Path('data/processed/destination_quotes.csv')
    data[REQUIRED + ['price_per_night', 'source_url', 'source_license']].to_csv(path, index=False)
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Normalized observations CSV')
    parser.add_argument('--output', default='.')
    parser.add_argument('--seed-barcelona', action='store_true')
    args = parser.parse_args()
    if not args.input and not args.seed_barcelona:
        parser.error('Provide --input or --seed-barcelona')
    payload = build(seed() if args.seed_barcelona else args.input, args.output)
    print(f"Built {len(payload['trees'])} covered destination/stay models; {len(payload['coverage'])} configurations audited")
