"""Streamlit client; predictions and persistence belong to REST service."""
import json
import os
import uuid
import pandas as pd
import plotly.express as px
import requests
import httpx
import streamlit as st
from scoring import ROOT, card_html

st.set_page_config(page_title='CARS24 | MSME Financial Health', page_icon=':material/monitoring:', layout='wide')
API = os.environ.get('MSME_API_URL', 'http://127.0.0.1:8000')
EMBEDDED_API = os.environ.get('MSME_EMBEDDED_API') == '1'


def close_embedded_service(resource):
    resource[0].close()


@st.cache_resource(scope='session', on_release=close_embedded_service)
def embedded_service(workspace_id):
    from contextlib import ExitStack
    from tempfile import TemporaryDirectory
    from fastapi.testclient import TestClient
    from api import create_app

    with ExitStack() as stack:
        folder = stack.enter_context(TemporaryDirectory(prefix='msme-demo-'))
        client = stack.enter_context(TestClient(create_app(db_path=f'{folder}/demo.sqlite3')))
        return stack.pop_all(), client


def api(path, method='GET', payload=None):
    try:
        if EMBEDDED_API:
            workspace_id = st.session_state.setdefault('_workspace_id', uuid.uuid4().hex)
            response = embedded_service(workspace_id)[1].request(method, path, json=payload)
        else:
            response = requests.request(method, API + path, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, httpx.HTTPError) as error:
        response = getattr(error, 'response', None)
        detail = response.text if response is not None else str(error)
        st.error('API request failed: ' + detail[:600])
        if not EMBEDDED_API:
            st.info('Start the local API with: uvicorn api:app --host 127.0.0.1 --port 8000')
        st.stop()


st.html("""
<style>
[data-testid="stAppViewContainer"] {background:#f5f7fb;}
[data-testid="stSidebar"] {background:white;border-right:1px solid #e4e9f2;}
.block-container {padding-top:2.2rem;max-width:1440px;}
h1,h2,h3 {color:#14213d;letter-spacing:-.035em;}
h1 {font-size:2rem !important;} h3 {font-size:1.15rem !important;}
[data-testid="stMetric"] {background:white;border:1px solid #e4e9f2;border-radius:12px;padding:18px 20px;}
[data-testid="stMetricValue"] {font-size:1.8rem;color:#14213d;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:12px;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:8px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:10px 12px;border-radius:8px;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#edf3ff;color:#2455ed;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button {border-radius:8px;}
.brand {font-size:28px;font-weight:850;letter-spacing:-1.2px;color:#14213d;margin-bottom:0;}
.brand-sub {font-size:12px;color:#64748b;margin-bottom:28px;}
.demo-note {font-size:12px;color:#64748b;margin:0 0 24px;}
</style>
""")
px.defaults.template = 'plotly_white'
px.defaults.color_discrete_sequence = ['#315cf5', '#7e9cf9', '#172747', '#14a38b']
with st.sidebar:
    st.image(str(ROOT / 'assets/cars24-logo.svg'), width=190)
    st.html('<div class="brand-sub">MSME Financial Health</div>')
    page = st.radio('Workspace', ['Portfolio', 'Health Card', 'Model Evidence', 'Add Business'], key='page')
    st.divider()
    st.caption('ASSESSMENT WORKSPACE')
    st.caption('Explore businesses, review financial evidence, and understand each recommendation.')
    st.caption('Interview demonstration · Not an official CARS24 service')
    if EMBEDDED_API:
        st.caption('Your demo workspace is temporary. Use synthetic inputs only; changes are cleared when the session ends.')
portfolio = api('/portfolio')
table = pd.DataFrame(portfolio['businesses'])
headings = {
    'Portfolio': ('Portfolio overview', 'Find businesses to review and understand the portfolio at a glance.'),
    'Health Card': ('Business health card', 'Understand financial health, credit suitability, and the evidence behind the result.'),
    'Model Evidence': ('Model evidence', 'See how the models were selected and evaluated on held-out businesses.'),
    'Add Business': ('Add a business snapshot', 'Validate financial inputs, save a snapshot, then generate a health assessment.'),
}
st.title(headings[page][0])
st.caption(headings[page][1])
st.html('<div class="demo-note">Demo data · No live financial connections or observed default outcomes.</div>')
filtered = table.copy()
if page in ['Portfolio', 'Health Card']:
    with st.expander('Search and filter businesses', expanded=page == 'Portfolio'):
        search, seg, ind, risk_col = st.columns([1.4, 1, 1, 1])
        query = search.text_input('Business ID', placeholder='Search business ID', key='search')
        segment = seg.selectbox('Customer segment', ['All'] + sorted(table.segment.unique()) if not table.empty else ['All'], key='segment')
        industry = ind.selectbox('Industry', ['All'] + sorted(table.industry.unique()) if not table.empty else ['All'], key='industry')
        risk_filter = risk_col.selectbox('Risk category', ['All', 'Low', 'Medium', 'High', 'Unassessed'], key='risk_filter')
    if not table.empty:
        if query: filtered = filtered[filtered.MSME_ID.str.contains(query, case=False, regex=False)]
        for column, value in [('segment', segment), ('industry', industry), ('risk', risk_filter)]:
            if value != 'All': filtered = filtered.loc[filtered[column].eq(value)]

if page == 'Portfolio':
    a,b,c,d = st.columns(4)
    a.metric('Businesses', len(filtered))
    b.metric('Mean health', f'{filtered.health_score.mean():.1f}' if not filtered.empty and filtered.health_score.notna().any() else '—')
    c.metric('Indicatively eligible', int(filtered.eligible.fillna(False).sum()) if not filtered.empty else 0)
    d.metric('Anomaly reviews', int(filtered.anomaly.fillna(False).sum()) if not filtered.empty else 0)
    st.subheader('Business portfolio')
    st.caption('Select a row to open its health card. Limits are indicative and subject to review.')
    selection = st.dataframe(filtered, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row', key='portfolio_table', column_config={
        'MSME_ID':'Business','segment':'Segment','industry':'Industry','health_score':st.column_config.NumberColumn('Health',format='%.1f'),
        'risk':'Risk','eligible':'Eligible','limit_inr':st.column_config.NumberColumn('Limit (INR)',format='localized'),
        'anomaly':'Review flag','needs_assessment':'Needs scoring'})
    if selection.selection.rows:
        st.session_state['selected_business'] = filtered.iloc[selection.selection.rows[0]].MSME_ID
        def open_health_card():
            st.session_state['page'] = 'Health Card'
        st.button('Open selected health card', type='primary', icon=':material/arrow_forward:', on_click=open_health_card)
    if filtered.empty: st.info('No businesses match these filters. Try a different search or segment.')
    if not filtered.empty:
        left,right = st.columns(2)
        left.plotly_chart(px.histogram(filtered.dropna(subset=['health_score']), x='health_score', color='segment', nbins=12,
                         title='Health score distribution', range_x=[0,100], labels={'health_score':'Financial health score','segment':'Segment','count':'Businesses'}), width='stretch')
        right.plotly_chart(px.histogram(filtered, x='risk', color='segment', title='Credit risk mix', labels={'risk':'Risk category','segment':'Segment','count':'Businesses'}, category_orders={'risk':['Low','Medium','High']}), width='stretch')

if page == 'Health Card':
    if filtered.empty:
        st.warning('No businesses match current filters.')
    else:
        choices = filtered.MSME_ID.tolist()
        previous = st.session_state.get('selected_business')
        selected = st.selectbox('Business', choices, index=choices.index(previous) if previous in choices else 0)
        needs = bool(filtered.loc[filtered.MSME_ID.eq(selected), 'needs_assessment'].iloc[0])
        if st.button('Generate new assessment', type='primary'):
            api(f'/businesses/{selected}/assessments', 'POST'); st.rerun()
        if needs:
            st.warning('Latest snapshot needs an assessment. Generate one above.')
        else:
            card = api(f'/businesses/{selected}/health-card'); raw = card['snapshot']; credit = card['credit']
            a,b,c,d = st.columns(4)
            a.metric('Financial health', f'{card["scores"]["Financial_Health_Score"]:.1f} / 100')
            b.metric('Risk category', credit['risk_category'])
            c.metric('Credit eligibility', 'Yes' if credit['eligible'] else 'No')
            d.metric('Indicative limit', f'₹{credit["recommended_limit_inr"]:,.0f}')
            st.caption(f'{raw["Customer_Segment"]} · {raw["Industry"]} · {card["model_version"]} · {credit["status"]}')
            summary, finance, explain, card_preview = st.tabs(['Overview & actions', 'Financial evidence', 'Why this result?', 'Printable Health Card'])
            with summary:
                scores = pd.DataFrame([{'Dimension': k.replace('_Score','').replace('_',' '), 'Score':v} for k,v in card['scores'].items() if k!='Financial_Health_Score'])
                st.plotly_chart(px.bar(scores, x='Score', y='Dimension', orientation='h', range_x=[0,100], title='Six financial dimensions', color_discrete_sequence=['#315cf5'], text='Score'), width='stretch')
                for flag in card['risk_indicators']: st.warning(flag)
                left,right = st.columns(2)
                with left:
                    st.subheader('Strengths supporting the score')
                    for row in card.get('strengths',[]): st.write(f'{row["feature"].replace("_"," ")}: +{row["contribution"]:.2f} points')
                with right:
                    st.subheader('Factors reducing the score')
                    for row in card.get('concerns',[]):
                        label=row['feature'].replace('_',' ')
                        if row['feature']=='EMI_On_Time_Rate_Pct' and raw['Has_Existing_Loan']=='No': label='EMI history (not applicable: no loan)'
                        st.write(f'{label}: {row["contribution"]:.2f} model points')
                st.subheader('Recommended actions')
                for text in card['recommendations']: st.write('• ' + text)
            with finance:
                st.subheader('1. Revenue Analysis & Channel Reconciliation')
                st.caption('Compare reported turnover against annualized banking, GST, and UPI flows to verify revenue consistency.')
                col_rev1, col_rev2 = st.columns([1.6, 1])
                annual_gst = raw['Monthly_GST_Sales_INR'] * 12
                annual_bank = raw['Monthly_Bank_Credits_INR'] * 12
                annual_upi = raw['Monthly_UPI_Inflow_INR'] * 12
                rev_df = pd.DataFrame({
                    'Channel': ['Reported Turnover', 'Annualized GST Sales', 'Annualized Bank Credits', 'Annualized UPI Inflows'],
                    'INR': [raw['Annual_Turnover_INR'], annual_gst, annual_bank, annual_upi]
                })
                col_rev1.plotly_chart(px.bar(rev_df, x='Channel', y='INR', color='Channel', text_auto='.2s',
                                             title='Multi-Channel Revenue Reconciliation (Annualized)',
                                             color_discrete_sequence=['#1e293b', '#2563eb', '#3b82f6', '#93c5fd']), width='stretch')
                with col_rev2:
                    st.metric('Reported Growth Rate', f'{raw["Revenue_Growth_Rate_Pct"]:+.1f}%')
                    st.metric('Seasonality Index', f'{raw["Seasonality_Index"]:.2f}', help='0 = steady year-round, 1 = extreme seasonal concentration')
                    alignment = annual_gst / annual_bank if annual_bank > 0 else 0
                    st.metric('GST-to-Bank Alignment', f'{alignment:.2f}x', help='Ratio of reported GST sales to bank inflows')

                st.divider()
                st.subheader('2. Cash Flow & Net Surplus Analysis')
                col_cf1, col_cf2 = st.columns([1.6, 1])
                bank_surplus = raw['Monthly_Bank_Credits_INR'] - raw['Monthly_Bank_Debits_INR']
                cf_df = pd.DataFrame({
                    'Flow': ['Bank Credits (In)', 'Bank Debits (Out)', 'Net Surplus'],
                    'INR': [raw['Monthly_Bank_Credits_INR'], raw['Monthly_Bank_Debits_INR'], bank_surplus],
                    'Type': ['Inflow', 'Outflow', 'Surplus' if bank_surplus >= 0 else 'Deficit']
                })
                col_cf1.plotly_chart(px.bar(cf_df, x='Flow', y='INR', color='Type', text_auto='.2s',
                                            title='Monthly Banking Cash Flow & Net Surplus',
                                            color_discrete_map={'Inflow': '#2563eb', 'Outflow': '#64748b', 'Surplus': '#15803d', 'Deficit': '#dc2626'}), width='stretch')
                with col_cf2:
                    st.metric('Monthly Net Bank Surplus', f'₹{bank_surplus:,.0f}')
                    st.metric('Average Bank Balance', f'₹{raw["Average_Bank_Balance_INR"]:,.0f}')
                    st.metric('Overdraft Usage Ratio', f'{raw["Overdraft_Usage_Ratio"]*100:.1f}%')
                    st.metric('Cashflow Stability Index', f'{raw["Cashflow_Stability_Index"]:.2f}', help='Scale 0–1; higher indicates smoother balance maintenance')

                st.divider()
                st.subheader('3. GST Performance & Tax Compliance')
                col_gst1, col_gst2 = st.columns([1.6, 1])
                gst_margin = raw['Monthly_GST_Sales_INR'] - raw['Monthly_GST_Purchases_INR']
                gst_df = pd.DataFrame({
                    'Item': ['GST Sales', 'GST Purchases', 'Value Added Margin'],
                    'INR': [raw['Monthly_GST_Sales_INR'], raw['Monthly_GST_Purchases_INR'], gst_margin]
                })
                col_gst1.plotly_chart(px.bar(gst_df, x='Item', y='INR', color='Item', text_auto='.2s',
                                             title='Monthly GST Sales vs. Purchases (Value-Add Margin)',
                                             color_discrete_sequence=['#0d9488', '#14b8a6', '#2dd4bf']), width='stretch')
                with col_gst2:
                    st.metric('GST Filing Timeliness', f'{raw["GST_Filing_Timeliness_Pct"]:.1f}%',
                              delta='Compliant' if raw['GST_Filing_Timeliness_Pct'] >= 80 else 'Review Required')
                    st.metric('Return Frequency', raw['GST_Return_Frequency'])
                    margin_pct = (gst_margin / raw['Monthly_GST_Sales_INR'] * 100) if raw['Monthly_GST_Sales_INR'] > 0 else 0
                    st.metric('Gross GST Margin %', f'{margin_pct:.1f}%')

                st.divider()
                st.subheader('4. UPI Transaction Snapshot & Dynamics')
                st.caption('Dated UPI history was not supplied, so this section compares current aggregate measures rather than a time trend.')
                col_upi1, col_upi2 = st.columns([1.6, 1])
                upi_df = pd.DataFrame({
                    'Stream': ['UPI Inflows', 'UPI Outflows'],
                    'INR': [raw['Monthly_UPI_Inflow_INR'], raw['Monthly_UPI_Outflow_INR']]
                })
                col_upi1.plotly_chart(px.bar(upi_df, x='Stream', y='INR', color='Stream', text_auto='.2s',
                                             title='Monthly UPI Digital Payment Flow',
                                             color_discrete_sequence=['#4f46e5', '#818cf8']), width='stretch')
                with col_upi2:
                    st.metric('Daily UPI Transactions', f'{raw["UPI_Daily_Txn_Count"]:g} txns/day')
                    st.metric('Average Ticket Size', f'₹{raw["UPI_Avg_Ticket_Size_INR"]:,.0f}')
                    st.metric('Transaction Volatility Index', f'{raw["Transaction_Volatility_Index"]:.2f}', help='Lower is more predictable')

                st.divider()
                st.subheader('5. Payroll Consistency & Workforce Stability')
                col_pay1, col_pay2, col_pay3, col_pay4 = st.columns(4)
                col_pay1.metric('Monthly Payroll', f'₹{raw["Monthly_Payroll_INR"]:,.0f}')
                col_pay2.metric('Salary Consistency', f'{raw["Salary_Consistency_Pct"]:.1f}%')
                col_pay3.metric('Employee Attrition', f'{raw["Employee_Attrition_Rate_Pct"]:.1f}%',
                               delta='Elevated' if raw['Employee_Attrition_Rate_Pct'] > 20 else 'Stable', delta_color='inverse')
                avg_salary = (raw['Monthly_Payroll_INR'] / raw['Employee_Count']) if raw['Employee_Count'] > 0 else 0
                col_pay4.metric('Avg. Monthly Salary', f'₹{avg_salary:,.0f}')
                st.caption('Note: Derived from single-snapshot alternative digital financial data; transaction flows may overlap across banking and UPI rails.')
            with explain:
                task = st.selectbox('What would you like to understand?', ['health','eligibility','risk','positive_limit'], format_func=lambda x: {'health':'Financial health score','eligibility':'Credit eligibility','risk':'Risk category','positive_limit':'Credit limit before eligibility check'}[x])
                explanation = card['explanations'][task]
                values = pd.DataFrame(explanation['features'][:12]).sort_values('contribution')
                values['label'] = values.feature.str.replace('_', ' ')
                st.plotly_chart(px.bar(values,x='contribution',y='label',orientation='h',color='source',title=f'Top contributions · {explanation["output_units"]}'),width='stretch')
                st.caption(f'Baseline {explanation["base_value"]:.4f} + all contributions = {explanation["reconstructed"]:.4f}. Raw model output before display bounds / eligibility gate.')
                st.dataframe(pd.DataFrame(explanation['source_contributions'].items(),columns=['Source','Contribution']),hide_index=True)
                st.dataframe(values[['feature','value','contribution','source']].astype({'value':str}),hide_index=True,width='stretch')
                st.caption(explanation['note']+' Displayed feature values: '+explanation['value_units']+'.')
            with card_preview:
                st.subheader('CARS24 Executive Financial Health Card')
                st.caption('Complete printable credit evaluation memo. Renders directly in browser and supports print / PDF export.')
                import streamlit.components.v1 as components
                components.html(card_html(card), height=850, scrolling=True)
            left,right = st.columns(2)
            left.download_button('Download health card JSON',json.dumps(card,indent=2),f'{selected}.json','application/json')
            right.download_button('Download printable HTML',card_html(card),f'{selected}.html','text/html')
            st.caption('Assessment '+card['assessment_id']+' · '+card['created_at'])

if page == 'Model Evidence':
    st.subheader('Validation you can inspect')
    st.caption('35,000 training / 7,500 validation / 7,500 test businesses. All supplied outcomes are excluded from inputs.')
    with st.expander('Which models are used, and why?'):
        st.markdown('**Financial scores:** CatBoost multi-output regression. **Eligibility and risk:** logistic regression. **Positive credit limit:** CatBoost regression. **Anomaly review:** IsolationForest.\n\nCandidates competed on validation data; simpler models were preferred within predefined tolerances. SHAP explains individual predictions. Synthetic target accuracy does not establish real default prediction.')
    path=ROOT/'reports/test_metrics.json'
    if path.exists():
        metrics=json.loads(path.read_text());a,b,c=st.columns(3)
        a.metric('Eligibility macro-F1',f'{metrics["eligibility_macro_f1"]:.3f}')
        b.metric('Risk macro-F1',f'{metrics["risk_macro_f1"]:.3f}')
        c.metric('Held-out businesses',metrics['test_rows'])
        st.dataframe(pd.read_csv(ROOT/'reports/test_scores.csv'),hide_index=True,width='stretch')
        with st.expander('Model comparison chart'):
            st.image(str(ROOT/'reports/model_results.png'))
        st.dataframe(pd.read_csv(ROOT/'reports/subgroup_metrics.csv').iloc[:,:9],hide_index=True,width='stretch')
        st.caption('Synthetic labels only. Subgroup errors and review rates are diagnostics, not fairness certification. No anomaly ground truth exists.')
    else: st.info('Execute modeling.ipynb to generate model evidence.')

if page == 'Add Business':
    st.subheader('1. Provide financial inputs')
    st.caption('Start from an example, edit the business ID and financial inputs, then save. Existing IDs create a new snapshot while retaining prior assessments.')
    st.info('After saving, open Health Card and select Generate new assessment.')
    example=json.loads((ROOT/'examples/businesses.json').read_text())[0]
    text=st.text_area('Business JSON',json.dumps(example,indent=2),height=350)
    if st.button('2. Validate and save snapshot'):
        try: payload=json.loads(text)
        except json.JSONDecodeError as error: st.error(str(error))
        else:
            result=api('/businesses','POST',payload);st.success(result['status']);st.rerun()
