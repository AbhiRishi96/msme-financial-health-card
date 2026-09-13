# Free hosted demo

Live app: https://cars24-msme-financial-health.streamlit.app/

Deployed on Streamlit Community Cloud (https://share.streamlit.io).

| Setting | Value |
|---|---|
| Repository | AbhiRishi96/msme-financial-health-card |
| Branch | main |
| Main file | cloud/streamlit_app.py |
| Python version (Advanced settings) | 3.14 |
| Suggested app URL | msme-financial-health-demo |
| Secrets | None |

Use the free Community Cloud service. No paid backend or custom domain is required.
The directory-specific requirements file keeps notebook tooling out of the hosted runtime.
Use the pinned model-library versions; the saved artifact was trained with them.

The hosted entrypoint runs the same FastAPI routes in-process through Starlette's
ASGI client. It does not expose a separate public REST endpoint. The normal local
uvicorn workflow remains available for REST demonstration.

Each visitor session gets a temporary SQLite database seeded with the six synthetic
examples. Inputs and assessments persist within that session and are cleaned up
when Streamlit releases it. A reload or server restart can reset the workspace.
Use synthetic inputs only. The supplied assignment dataset is not deployed.

Free instances can sleep or restart. Open the link before presenting and allow
cold startup time. Hosting success is confirmed only after the deployed URL has
been opened and the assessment flow tested.
