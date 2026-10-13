import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.destinations import destination
from src.multicity import build, normalize, predict, scope_id

ROOT = Path(__file__).resolve().parents[1]

def example():
    return pd.read_csv(ROOT / 'examples/destination_hotel.csv', dtype={'child_ages': str})

def test_any_moroccan_city_and_aliases():
    assert destination('ma', 'Ain Aouda') == ('MA', 'Ain Aouda')
    assert destination('MA', ' Fès ') == ('MA', 'Fes')
    assert destination('MA', 'Tanger') == ('MA', 'Tangier')
    assert destination('FR', ' paris ') == ('FR', 'Paris')
    with pytest.raises(ValueError): destination('FR', 'Unknown')
    with pytest.raises(ValueError): destination('MA', '')

@pytest.mark.parametrize('column,value', [('review_count', -1), ('adults', 2.5), ('children', 1.5),
    ('child_ages', '0.5'), ('child_ages', ''), ('checkout', '2026-01-23'), ('currency', 'dirhams')])
def test_quote_contract_rejects_invalid_values(column, value):
    frame = example(); frame[column] = value
    with pytest.raises(ValueError): normalize(frame)

@pytest.mark.parametrize('changes', [dict(country='MA', city='Marrakech'), dict(city='Madrid'),
    dict(currency='MAD'), dict(checkin='2026-02-24', checkout='2026-02-25'), dict(rooms=2),
    dict(children=0, child_ages=''), dict(adults=7), dict(distance_center_km=1000)])
def test_missing_city_currency_date_occupancy_and_domain_are_rejected(changes):
    frame = example()
    for name, value in changes.items(): frame[name] = value
    with pytest.raises(ValueError): predict(frame, ROOT / 'artifacts')

def test_numeric_scope_identity_is_stable():
    frame = normalize(example())
    row = frame.iloc[0].copy(); expected = scope_id(row)
    row['rooms'] = 1.0; row['children'] = 1.0
    assert scope_id(row) == expected

def test_batch_routes_geographic_inputs_to_destination_model():
    from src.predict import predict as public_predict
    frame = example()
    assert public_predict(frame, ROOT / 'artifacts').prediction_currency.tolist() == ['EUR']
    frame['city'] = 'Marrakech'; frame['country'] = 'MA'
    with pytest.raises(ValueError, match='No trained price coverage'):
        public_predict(frame, ROOT / 'artifacts')

def test_sparse_cities_do_not_get_a_fallback(tmp_path):
    frame = example(); frame['country'] = 'MA'; frame['city'] = 'Rabat'
    frame['price_per_night'] = 500; frame['source_url'] = 'https://example.org/test-fixture'
    frame['source_license'] = 'test-only'
    path = tmp_path / 'quotes.csv'; frame.to_csv(path, index=False)
    result = build(path, tmp_path)
    assert result['coverage'][0]['status'] == 'needs_data'
    assert result['trees'] == {}
    with pytest.raises(ValueError, match='No trained price coverage'):
        predict(frame, tmp_path / 'artifacts')

def test_browser_and_python_prediction_parity():
    node = shutil.which('node')
    if node is None: pytest.skip('Node required to validate browser inference')
    frame = pd.read_csv(ROOT / 'data/processed/destination_quotes.csv', dtype={'child_ages': str})
    payload = json.loads((ROOT / 'reports/destination_coverage.json').read_text(encoding='utf-8'))
    record = next(r for r in payload['coverage'] if r['status'] == 'ready')
    supported = frame.adults.isin(record['adults'])
    numeric = normalize(frame)
    for name, (low, high) in record['domains'].items(): supported &= numeric[name].between(low, high)
    frame = frame.loc[supported].head(30)
    expected = predict(frame, ROOT / 'artifacts').prediction_per_night.to_numpy()
    values = normalize(frame)[['adults','distance_center_km','log_review_count','amenity_mentions_count']].to_numpy().tolist()
    script = "global.window={};require('./dashboard/destinations-data.js');const {forestPrice}=require('./dashboard/destinations.js');const p=window.HOTEL_DESTINATIONS;const rows=JSON.parse(process.argv[1]);process.stdout.write(JSON.stringify(rows.map(r=>forestPrice(p.trees[p.coverage[0].id],r))));"
    actual = json.loads(subprocess.check_output([node, '-e', script, json.dumps(values)], cwd=ROOT, text=True))
    np.testing.assert_allclose(actual, expected, rtol=1e-10)

def test_city_training_keeps_markets_and_properties_separate(tmp_path):
    # Deliberately synthetic unit-test fixtures; never exported as project observations.
    rows = []
    for city, price in [('Rabat', 400), ('Fes', 900)]:
        for prop in range(35):
            for adults in [1, 2]:
                row = example().iloc[0].to_dict()
                row.update(country='MA', city=city, currency='MAD', property_id=f'hotel-{prop}',
                           adults=adults, price_per_night=price + prop + adults,
                           source_url='https://example.org/unit-test-only', source_license='test-only')
                rows.append(row)
    data = pd.DataFrame(rows)
    path = tmp_path / 'quotes.csv'; data.to_csv(path, index=False)
    output = build(path, tmp_path)
    assert len(output['trees']) == 2
    for record in output['coverage']:
        assert set(record['train_property_ids']).isdisjoint(record['test_property_ids'])
        assert record['properties'] == 35 and record['test_quotes'] == 14
    request = data.groupby('city').head(1)
    result = predict(request, tmp_path / 'artifacts').set_index('city')
    assert result.loc['Fes', 'prediction_per_night'] > result.loc['Rabat', 'prediction_per_night'] + 400
