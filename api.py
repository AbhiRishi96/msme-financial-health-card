"""Local REST demo with immutable snapshots and assessments."""
from contextlib import asynccontextmanager, closing
from pathlib import Path
import json
import os
import sqlite3
import uuid

from fastapi import FastAPI, HTTPException
from scoring import Business, ROOT, assess, load_bundle


def create_app(db_path=None, model_path=None):
    database = Path(db_path or os.environ.get('MSME_DB_PATH', ROOT / 'local/demo.sqlite3'))
    model_path = str(model_path or ROOT / 'artifacts/models.joblib')

    def connect():
        con = sqlite3.connect(database, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA foreign_keys=ON')
        return con

    def latest_snapshot(con, business_id):
        row = con.execute('SELECT * FROM snapshots WHERE business_id=? ORDER BY id DESC LIMIT 1', (business_id,)).fetchone()
        if row is None: raise HTTPException(404, 'Business not found')
        return row

    def save_assessment(con, snapshot, bundle):
        card = assess(json.loads(snapshot['payload']), bundle)
        card.update(assessment_id=str(uuid.uuid4()), snapshot_id=snapshot['id'])
        con.execute('INSERT INTO assessments(id,business_id,snapshot_id,created_at,payload) VALUES(?,?,?,?,?)',
                    (card['assessment_id'], snapshot['business_id'], snapshot['id'], card['created_at'], json.dumps(card, allow_nan=False)))
        return card

    @asynccontextmanager
    async def lifespan(app):
        database.parent.mkdir(parents=True, exist_ok=True)
        app.state.bundle = load_bundle(model_path)
        with closing(connect()) as con, con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY, business_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS snapshot_business ON snapshots(business_id,id);
                CREATE TABLE IF NOT EXISTS assessments(id TEXT PRIMARY KEY, business_id TEXT NOT NULL,
                    snapshot_id INTEGER NOT NULL REFERENCES snapshots(id), created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS assessment_business ON assessments(business_id,created_at);
            ''')
            # Independent example scenarios only, never supplied CSV rows.
            if not con.execute('SELECT 1 FROM snapshots LIMIT 1').fetchone():
                for example in json.loads((ROOT / 'examples/businesses.json').read_text()):
                    business = Business.model_validate(example)
                    con.execute('INSERT INTO snapshots(business_id,payload) VALUES(?,?)', (business.MSME_ID, business.model_dump_json()))
                    save_assessment(con, latest_snapshot(con, business.MSME_ID), app.state.bundle)
        yield

    app = FastAPI(title='MSME Financial Health Card', version='1.0.0', lifespan=lifespan,
                  description='Local synthetic-data demonstration. No real lending decisions or live integrations.')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model_version': app.state.bundle['version']}

    @app.post('/businesses', status_code=201)
    def ingest(business: Business):
        with closing(connect()) as con, con:
            cursor = con.execute('INSERT INTO snapshots(business_id,payload) VALUES(?,?)', (business.MSME_ID, business.model_dump_json()))
            return {'MSME_ID': business.MSME_ID, 'snapshot_id': cursor.lastrowid, 'status': 'Stored; generate a new assessment.'}

    @app.post('/assess', status_code=201)
    def assess_business(business: Business):
        """One-step instant scoring: persists snapshot and assessment, returning complete health card immediately."""
        with closing(connect()) as con, con:
            cursor = con.execute('INSERT INTO snapshots(business_id,payload) VALUES(?,?)', (business.MSME_ID, business.model_dump_json()))
            snapshot = {'id': cursor.lastrowid, 'business_id': business.MSME_ID, 'payload': business.model_dump_json()}
            card = save_assessment(con, snapshot, app.state.bundle)
            card['newer_snapshot_available'] = False
            return card

    @app.get('/businesses/{business_id}')
    def get_business(business_id: str):
        with closing(connect()) as con:
            row = latest_snapshot(con, business_id)
            return {'snapshot_id': row['id'], 'business': json.loads(row['payload'])}

    @app.post('/businesses/{business_id}/assessments', status_code=201)
    def generate(business_id: str):
        with closing(connect()) as con, con:
            return save_assessment(con, latest_snapshot(con, business_id), app.state.bundle)

    @app.get('/businesses/{business_id}/health-card')
    def get_card(business_id: str):
        with closing(connect()) as con:
            snapshot = latest_snapshot(con, business_id)
            row = con.execute('SELECT payload FROM assessments WHERE business_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1', (business_id,)).fetchone()
            if row is None: raise HTTPException(409, 'No assessment yet; generate one first.')
            card = json.loads(row['payload'])
            card['newer_snapshot_available'] = card['snapshot_id'] != snapshot['id']
            return card

    @app.get('/businesses/{business_id}/assessments')
    def assessment_history(business_id: str):
        with closing(connect()) as con:
            latest_snapshot(con, business_id)
            return [dict(row) for row in con.execute('SELECT id,snapshot_id,created_at FROM assessments WHERE business_id=? ORDER BY created_at DESC,rowid DESC', (business_id,))]

    @app.get('/businesses/{business_id}/credit-recommendation')
    def credit(business_id: str):
        card = get_card(business_id)
        return {'MSME_ID': business_id, 'assessment_id': card['assessment_id'], 'model_version': card['model_version'],
                'credit': card['credit'], 'reasons': card['risk_indicators'] + card['recommendations'],
                'explanation': card['explanations']['eligibility'], 'disclaimer': card['disclaimer'],
                'newer_snapshot_available': card['newer_snapshot_available']}

    @app.get('/portfolio')
    def portfolio(segment: str | None = None, industry: str | None = None, risk: str | None = None):
        with closing(connect()) as con:
            rows = con.execute('''SELECT s.id,s.business_id,s.payload,
                (SELECT a.payload FROM assessments a WHERE a.business_id=s.business_id ORDER BY a.created_at DESC,a.rowid DESC LIMIT 1) AS assessment
                FROM snapshots s WHERE s.id=(SELECT MAX(s2.id) FROM snapshots s2 WHERE s2.business_id=s.business_id)
                ORDER BY s.business_id''').fetchall()
        businesses = []
        for row in rows:
            raw = json.loads(row['payload']); card = json.loads(row['assessment']) if row['assessment'] else None
            stale = card is not None and card['snapshot_id'] != row['id']
            item = {'MSME_ID': row['business_id'], 'segment': raw['Customer_Segment'], 'industry': raw['Industry'],
                    'health_score': card['scores']['Financial_Health_Score'] if card and not stale else None,
                    'risk': card['credit']['risk_category'] if card and not stale else 'Unassessed',
                    'eligible': card['credit']['eligible'] if card and not stale else None,
                    'limit_inr': card['credit']['recommended_limit_inr'] if card and not stale else None,
                    'anomaly': card['anomaly']['flagged'] if card and not stale else None,
                    'needs_assessment': card is None or stale}
            if (segment and item['segment'] != segment) or (industry and item['industry'] != industry) or (risk and item['risk'] != risk): continue
            businesses.append(item)
        scored = [b for b in businesses if b['health_score'] is not None]
        return {'businesses': businesses, 'summary': {'count': len(businesses), 'assessed_count': len(scored),
                'mean_health': sum(b['health_score'] for b in scored) / len(scored) if scored else None,
                'eligible_count': sum(b['eligible'] for b in scored), 'anomaly_count': sum(b['anomaly'] for b in scored)}}

    return app


app = create_app()
