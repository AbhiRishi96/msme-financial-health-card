"""Hosted demo uses the same scoring routes with isolated visitor workspaces."""
import json
from scoring import ROOT
from streamlit.testing.v1 import AppTest


def test_hosted_assessment_and_session_isolation(monkeypatch):
    monkeypatch.setenv('MSME_EMBEDDED_API', '1')
    first = AppTest.from_file(ROOT / 'dashboard.py', default_timeout=60).run()
    assert not first.exception and not first.error
    first.radio(key='page').set_value('Add Business').run()
    record = json.loads(first.text_area[0].value)
    record['MSME_ID'] = 'CLOUD_SESSION_CHECK'
    first.text_area[0].set_value(json.dumps(record)).run()
    next(b for b in first.button if b.label == '2. Validate and save snapshot').click().run()
    first.radio(key='page').set_value('Health Card').run()
    next(s for s in first.selectbox if s.label == 'Business').set_value('CLOUD_SESSION_CHECK').run()
    next(b for b in first.button if b.label == 'Generate new assessment').click().run()
    assert not first.exception and not first.error
    assert any(t.label == 'Why this result?' for t in first.tabs)
    second = AppTest.from_file(ROOT / 'dashboard.py', default_timeout=60).run()
    second.radio(key='page').set_value('Health Card').run()
    assert not second.exception and not second.error
    assert 'CLOUD_SESSION_CHECK' not in next(s for s in second.selectbox if s.label == 'Business').options
