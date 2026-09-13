# REST API

Base URL: `http://127.0.0.1:8000`. OpenAPI: `/openapi.json`; Swagger: `/docs`. Local demo only, no authentication or live source integrations.

| Method and path | Behavior |
|---|---|
| `POST /businesses` | Validate all 34 snapshot fields (ID + 33 inputs), persist a new snapshot, return 201 and snapshot ID. Repeat ID creates another version. |
| `POST /assess` | One-step instant scoring: validate 34 fields, persist snapshot & assessment, return 201 with complete health card immediately. |
| `POST /businesses/{id}/assessments` | Score latest snapshot, save immutable assessment, return 201 with complete card. Each request creates a separate assessment. |
| `GET /businesses/{id}` | Retrieve latest input snapshot. |
| `GET /businesses/{id}/health-card` | Latest assessment with the exact snapshot scored; flags a newer unassessed snapshot. |
| `GET /businesses/{id}/credit-recommendation` | Eligibility, probability of synthetic eligibility, risk, indicative limit, explanations and review reasons. |
| `GET /businesses/{id}/assessments` | Assessment history IDs, snapshot IDs and timestamps. |
| `GET /portfolio` | Latest-state summaries and rows; optional exact-match `segment`, `industry`, `risk` filters. Stale assessments are excluded from scored aggregates. |
| `GET /health` | Readiness and loaded model version. |

Input contract: `Business` in scoring.py. [Independent request examples](examples/businesses.json) contain every field. Categories outside trained industry/business-type/location values are accepted and handled by models; fixed segment/loan/frequency enums are validated. No-loan EMI payment rate must be null. Unknown fields, target columns, missing required fields, negative amounts, nonfinite numbers, and inconsistent NTC history are rejected with HTTP 422.

Unknown business: 404. Existing business without an assessment: 409. Zero amounts are valid; unavailable ratios lead to a data-quality review indication. Existing borrowers can have unavailable EMI history, which is explicitly flagged. Reported history is checked against business age with one month rounding tolerance.

Health card contains seven 0–100 scores; raw model explanations before bounding; credit eligibility, risk probabilities and indicative limit; anomaly score and demo threshold; model version; risk indicators; recommendations; input snapshot; and immutable assessment/snapshot identifiers.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/businesses/DEMO_NTC_RETAIL/health-card
curl -X POST http://localhost:8000/businesses/DEMO_NTC_RETAIL/assessments
curl 'http://localhost:8000/portfolio?segment=NTC'
```

The service does not accept supplied outcome labels during ingestion. It never retrains from its predictions. Public datasets are not uploaded by the dashboard. Database writes use parameterized SQL and retain old versions. Consent verification and real-source adapters are documented future boundaries, not simulated approvals in this demo.
