from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.prepare import parse_eur,parse_distance,parse_reviews,url_fields,FEATURES
from src.model import validate_input,metrics
from src.scrape import parse_search
from src.train import assign_splits

@pytest.mark.parametrize('value,expected',[('€ 105',105),('€ 1.234',1234),('€ 1.234,50',1234.5),('€ 105,50',105.5)])
def test_eur_units(value,expected):assert parse_eur(value)==expected

@pytest.mark.parametrize('value',['$ 105','€ -1','€ 0'])
def test_invalid_eur(value):
    with pytest.raises(ValueError):parse_eur(value)

@pytest.mark.parametrize('value,expected',[('a 2,7 km del centro',2.7),('a 300 m del centro',.3),('A 750 m de la playa',.75)])
def test_distance_units(value,expected):assert parse_distance(value)==expected

def test_unknown_distance_remains_missing():assert np.isnan(parse_distance(np.nan))
def test_review_count_is_not_score():assert parse_reviews('7.898 comentarios')==7898

def test_url_sanitization_and_dates():
    result=url_fields('https://www.booking.com/hotel/es/example.es.html?sid=private&checkin=2026-01-24&checkout=2026-01-25&no_rooms=1&group_adults=2&group_children=1&age=0')
    assert result['canonical_url']=='https://www.booking.com/hotel/es/example.es.html'
    assert result['checkin']=='2026-01-24'
    assert result['child_ages']=='0'

def test_block_page_is_rejected():
    with pytest.raises(ValueError):parse_search('<html>Access denied</html>')

def test_booking_parser_fixture():
    html='<div data-testid="property-card"><div data-testid="title">Fixture</div><span data-testid="price-and-discounted-price">€ 105</span><a data-testid="title-link" href="https://www.booking.com/hotel/es/fixture.html?sid=removed"></a></div>'
    assert parse_search(html)==[{'name':'Fixture','price_text':'€ 105','canonical_url':'https://www.booking.com/hotel/es/fixture.html'}]

def test_property_disjoint_and_reproducible_splits():
    frame=pd.DataFrame({'property_id':[f'hotel{i}' for i in range(100) for _ in range(3)]})
    labels=assign_splits(frame)
    assert labels.equals(assign_splits(frame))
    assert frame.assign(split=labels).groupby('property_id').split.nunique().max()==1
    assert set(labels)=={'train','validation','calibration','test'}

def test_no_target_or_identity_features():
    assert set(FEATURES).isdisjoint({'price_per_night_eur','quoted_price_eur','name','property_id','canonical_url','positioning'})

def test_known_metrics():assert metrics([100,200],[110,180])['mae_eur']==15

def test_missing_input_columns_are_rejected():
    with pytest.raises(ValueError,match='Missing input columns'):validate_input(pd.DataFrame({'adults':[2]}))

def test_invalid_occupancy_is_rejected():
    frame=pd.DataFrame([{feature:0 for feature in FEATURES}]);frame['district']='Example';frame['adults']=5
    with pytest.raises(ValueError,match='Supported observed range'):validate_input(frame)

def test_committed_cohort_integrity():
    path=Path('data/processed/hotel_quotes.csv')
    if not path.exists():pytest.skip('Run preparation first')
    data=pd.read_csv(path)
    assert data.rooms.eq(1).all() and data.nights.eq(1).all()
    assert data.price_per_night_eur.gt(0).all()
    assert data.canonical_url.str.contains('?',regex=False).sum()==0
    assert not data.duplicated(['property_id','adults','checkin','checkout']).any()
    if Path('reports/property_splits.csv').exists():
        split=pd.read_csv('reports/property_splits.csv')
        assert split.groupby('property_id').split.nunique().max()==1
