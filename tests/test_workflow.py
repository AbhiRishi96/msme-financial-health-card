"""Behavior checks with independent fixtures, not supplied source rows."""
import copy
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import create_app
from scoring import ROOT, RAW, TARGETS, SCORES, ANOMALY_COLUMNS, Business, assess, make_features, load_bundle, card_html


@pytest.fixture(scope='module')
def example():
    return json.loads((ROOT/'examples/businesses.json').read_text())[0]


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp('db')/'demo.sqlite3'
    with TestClient(create_app(path)) as c:
        c.database = path
        yield c


def test_no_label_leakage(example):
    frame = pd.DataFrame([example | {c: 999 for c in TARGETS}])
    features = make_features(frame)
    assert not set(TARGETS+['MSME_ID']) & set(features)
    for record in [load_bundle()[k] for k in ['scores','eligibility','risk','limit']]:
        assert not set(TARGETS+['MSME_ID']) & set(record['columns'])


def test_missing_history_and_signed_surplus(example):
    business = Business.model_validate(example)
    frame = make_features(pd.DataFrame([business.model_dump()]))
    assert frame.EMI_On_Time_Rate_Pct.isna().all()
    assert frame.EMI_Not_Applicable.iloc[0] == 1
    changed = example | {'Monthly_Bank_Debits_INR': 900000}
    assert make_features(pd.DataFrame([changed])).Bank_Surplus_INR.iloc[0] < 0


@pytest.mark.parametrize('change', [
    {'Financial_Health_Score':80}, {'Probability_of_Default':.05},
    {'Monthly_UPI_Inflow_INR':-1}, {'GST_Filing_Timeliness_Pct':101},
    {'EMI_On_Time_Rate_Pct':90}, {'Has_Existing_Loan':'Yes'},
    {'Annual_Turnover_INR':float('inf')}, {'Annual_Turnover_INR':float('nan')},
])
def test_invalid_payloads(example, change):
    with pytest.raises(ValidationError): Business.model_validate(example | change)


def test_unseen_categories_zero_denominators(example):
    changed = example | {'Industry':'Novel industry','Business_Type':'New legal form',
                         'Monthly_Bank_Credits_INR':0,'Monthly_Bank_Debits_INR':0,'Monthly_GST_Sales_INR':0}
    card = assess(changed)
    assert all(np.isfinite(list(card['scores'].values())))
    assert any('Zero denominator' in s for s in card['risk_indicators'])
    json.dumps(card, allow_nan=False)


def test_explanations_and_print_escape(example):
    card = assess(example | {'Industry':'<script>alert(1)</script>'})
    for explanation in card['explanations'].values():
        assert np.isclose(explanation['base_value'] + sum(r['contribution'] for r in explanation['features']), explanation['raw_prediction'], atol=1e-5)
    assert all(0 <= s <= 100 for s in card['scores'].values())
    card['recommendations']=['<script>alert(1)</script>']
    html = card_html(card)
    assert '<script>' not in html and '&lt;script&gt;' in html


def test_anomaly_perturbation(example):
    base = make_features(pd.DataFrame([example]))
    stressed = base.copy()
    # Controlled input stress; this is a sensitivity check, not labeled accuracy.
    stressed.loc[:,['Bank_Surplus_Margin','Cashflow_Stability_Index','Vendor_Payment_Timeliness_Pct']] = [-2,.1,10]
    stressed.loc[:,['Avg_Invoice_Payment_Delay_Days','Payroll_Burden','EMI_Burden']] = [180,3,2]
    model=load_bundle()['anomaly']
    assert -model.score_samples(stressed[ANOMALY_COLUMNS])[0] > -model.score_samples(base[ANOMALY_COLUMNS])[0]


def test_api_parity_history_and_reopen(client, example):
    payload = example | {'MSME_ID':'TEST_HISTORY'}
    assert client.post('/businesses',json=payload).status_code == 201
    assert client.get('/businesses/TEST_HISTORY/health-card').status_code == 409
    first = client.post('/businesses/TEST_HISTORY/assessments').json()
    second = client.post('/businesses/TEST_HISTORY/assessments').json()
    direct = assess(payload)
    assert first['scores'] == second['scores'] == direct['scores']
    assert first['credit'] == direct['credit']
    assert first['assessment_id'] != second['assessment_id']
    assert len(client.get('/businesses/TEST_HISTORY/assessments').json()) == 2
    credit=client.get('/businesses/TEST_HISTORY/credit-recommendation').json()
    assert credit['credit'] == second['credit']
    assert client.post('/businesses',json=payload | {'Revenue_Growth_Rate_Pct':-20}).status_code == 201
    assert client.get('/businesses/TEST_HISTORY/health-card').json()['newer_snapshot_available']
    assert next(b for b in client.get('/portfolio').json()['businesses'] if b['MSME_ID']=='TEST_HISTORY')['needs_assessment']
    # Independent connection proves persisted history survives request connections.
    with sqlite3.connect(client.database) as con:
        stored = json.loads(con.execute('SELECT payload FROM assessments WHERE id=?',(first['assessment_id'],)).fetchone()[0])
    assert stored['scores']==first['scores']


def test_api_validation_and_filters(client, example):
    assert client.get('/businesses/MISSING/health-card').status_code == 404
    assert client.post('/businesses/MISSING/assessments').status_code == 404
    assert client.post('/businesses',json=example | {'Credit_Eligible':'Yes'}).status_code == 422
    assert client.post('/businesses',json={'MSME_ID':'incomplete'}).status_code == 422
    response=client.get('/portfolio',params={'segment':'NTC'}).json()
    assert all(b['segment']=='NTC' for b in response['businesses'])
    assert response['summary']['count']==len(response['businesses'])


def test_ineligible_limit_gate():
    records=json.loads((ROOT/'examples/businesses.json').read_text())
    card=assess(next(r for r in records if r['MSME_ID']=='DEMO_NTC_STRESSED'))
    assert not card['credit']['eligible']
    assert card['credit']['recommended_limit_inr']==0


def test_direct_assess_endpoint(client, example):
    payload = example | {'MSME_ID': 'TEST_INSTANT'}
    resp = client.post('/assess', json=payload)
    assert resp.status_code == 201
    card = resp.json()
    assert card['MSME_ID'] == 'TEST_INSTANT'
    assert 'scores' in card and 'Financial_Health_Score' in card['scores']
    assert not card['newer_snapshot_available']
    assert client.get('/businesses/TEST_INSTANT/health-card').status_code == 200
    assert len(client.get('/businesses/TEST_INSTANT/assessments').json()) == 1
