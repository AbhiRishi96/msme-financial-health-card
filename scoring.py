"""Shared input contract, features, inference, and explanations. No training here."""
from datetime import datetime, timezone
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Literal
import json

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent
SCORES = ['Business_Stability_Score', 'Cashflow_Score', 'Revenue_Consistency_Score',
          'Payment_Behaviour_Score', 'Business_Growth_Score', 'Compliance_Score',
          'Financial_Health_Score']
TARGETS = SCORES + ['Probability_of_Default', 'Credit_Risk_Category', 'Credit_Eligible',
                    'Recommended_Credit_Limit_INR']
DISCLAIMER = ('Synthetic-data demonstration. Predictions reproduce supplied labels; '
              'they are not validated default probabilities or actual lending decisions. '
              'Dated transaction history is unavailable.')


class Business(BaseModel):
    """One current business snapshot. All source fields required; EMI may be null."""
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, str_strip_whitespace=True)
    MSME_ID: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    Customer_Segment: Literal['NTC', 'NTB', 'Existing-to-Credit']
    Business_Type: str = Field(min_length=1, max_length=80)
    Industry: str = Field(min_length=1, max_length=80)
    Location_Category: str = Field(min_length=1, max_length=80)
    Years_in_Operation: float = Field(gt=0, le=200)
    Employee_Count: int = Field(ge=1)
    Annual_Turnover_INR: float = Field(ge=0)
    Monthly_GST_Sales_INR: float = Field(ge=0)
    Monthly_GST_Purchases_INR: float = Field(ge=0)
    GST_Filing_Timeliness_Pct: float = Field(ge=0, le=100)
    GST_Return_Frequency: Literal['Monthly', 'Quarterly']
    Monthly_UPI_Inflow_INR: float = Field(ge=0)
    Monthly_UPI_Outflow_INR: float = Field(ge=0)
    UPI_Avg_Ticket_Size_INR: float = Field(ge=0)
    UPI_Daily_Txn_Count: float = Field(ge=0)
    Transaction_Volatility_Index: float = Field(ge=0, le=1)
    Monthly_Bank_Credits_INR: float = Field(ge=0)
    Monthly_Bank_Debits_INR: float = Field(ge=0)
    Average_Bank_Balance_INR: float = Field(ge=0)
    Cashflow_Stability_Index: float = Field(ge=0, le=1)
    Has_Existing_Loan: Literal['Yes', 'No']
    Monthly_Loan_EMI_INR: float = Field(ge=0)
    EMI_On_Time_Rate_Pct: float | None = Field(ge=0, le=100)
    Overdraft_Usage_Ratio: float = Field(ge=0, le=1)
    Monthly_Payroll_INR: float = Field(ge=0)
    Salary_Consistency_Pct: float = Field(ge=0, le=100)
    Employee_Attrition_Rate_Pct: float = Field(ge=0, le=100)
    Avg_Invoice_Payment_Delay_Days: float = Field(ge=0)
    Customer_Concentration_Ratio: float = Field(ge=0, le=1)
    Vendor_Payment_Timeliness_Pct: float = Field(ge=0, le=100)
    Seasonality_Index: float = Field(ge=0, le=1)
    Revenue_Growth_Rate_Pct: float = Field(ge=-100)
    Credit_History_Months: int = Field(ge=0)

    @model_validator(mode='after')
    def check_loan(self):
        if self.Has_Existing_Loan == 'No' and (self.Monthly_Loan_EMI_INR != 0 or self.EMI_On_Time_Rate_Pct is not None):
            raise ValueError('No existing loan requires zero EMI and null EMI payment rate.')
        if self.Customer_Segment == 'NTC' and (self.Has_Existing_Loan == 'Yes' or self.Credit_History_Months > 0):
            raise ValueError('NTC requires no existing loan and zero credit history.')
        if self.Credit_History_Months > self.Years_in_Operation * 12 + 1:
            raise ValueError('Credit history exceeds business age (one month rounding tolerance).')
        return self


RAW = list(Business.model_fields)[1:]
CATEGORIES = ['Customer_Segment', 'Business_Type', 'Industry', 'Location_Category',
              'GST_Return_Frequency', 'Has_Existing_Loan']
LOG_COLUMNS = [c for c in RAW if c.endswith('_INR')] + ['Employee_Count', 'UPI_Daily_Txn_Count']
ANOMALY_COLUMNS = ['Bank_Surplus_Margin', 'Balance_Coverage', 'EMI_Burden', 'GST_Margin',
                   'Payroll_Burden', 'GST_Bank_Alignment', 'Transaction_Volatility_Index',
                   'Cashflow_Stability_Index', 'Avg_Invoice_Payment_Delay_Days',
                   'Customer_Concentration_Ratio', 'Vendor_Payment_Timeliness_Pct',
                   'Revenue_Growth_Rate_Pct', 'Salary_Consistency_Pct', 'Overdraft_Usage_Ratio']


def make_features(frame):
    """Explicit allowlist; zero denominators yield missing ratios, never infinity."""
    x = frame.loc[:, RAW].copy()
    for c in CATEGORIES:
        x[c] = x[c].fillna('Unknown').astype(str)
    for c in set(RAW) - set(CATEGORIES):
        x[c] = pd.to_numeric(x[c], errors='raise')
    x['EMI_Not_Applicable'] = x.Has_Existing_Loan.eq('No').astype(int)
    credits = x.Monthly_Bank_Credits_INR.replace(0, np.nan)
    x['Bank_Surplus_INR'] = x.Monthly_Bank_Credits_INR - x.Monthly_Bank_Debits_INR
    x['Bank_Surplus_Margin'] = x.Bank_Surplus_INR / credits
    x['Balance_Coverage'] = x.Average_Bank_Balance_INR / x.Monthly_Bank_Debits_INR.replace(0, np.nan)
    x['EMI_Burden'] = x.Monthly_Loan_EMI_INR / credits
    x['GST_Margin'] = (x.Monthly_GST_Sales_INR - x.Monthly_GST_Purchases_INR) / x.Monthly_GST_Sales_INR.replace(0, np.nan)
    x['UPI_Net_INR'] = x.Monthly_UPI_Inflow_INR - x.Monthly_UPI_Outflow_INR
    x['Payroll_Burden'] = x.Monthly_Payroll_INR / credits
    x['GST_Bank_Alignment'] = x.Monthly_GST_Sales_INR / credits
    return x


def log_numeric(frame):
    """Stateless transform for linear baselines; signed net values stay signed."""
    x = frame.copy()
    columns = [c for c in LOG_COLUMNS if c in x]
    x[columns] = np.log1p(x[columns])
    return x


def source_group(feature):
    feature = feature.split('__')[-1].removeprefix('missingindicator_')
    if 'GST' in feature or feature == 'Compliance_Score': return 'GST'
    if 'UPI' in feature: return 'UPI'
    if any(k in feature for k in ['Payroll', 'Salary', 'Employee']): return 'Payroll'
    if any(k in feature for k in ['Loan', 'EMI', 'Credit_History', 'Overdraft']): return 'Credit history'
    if any(k in feature for k in ['Bank', 'Balance', 'Cashflow']): return 'Banking'
    if any(k in feature for k in ['Invoice', 'Vendor', 'Concentration']): return 'Invoices'
    return 'Business profile'


def predict_record(record, features, probability=False):
    x = features[record['columns']]
    return record['model'].predict_proba(x) if probability else record['model'].predict(x)


def credit_limit(record, features):
    # Training target is log1p(INR). Guard numerical overflow; flag OOD separately.
    return np.expm1(np.clip(predict_record(record, features), 0, 40))


def explain_record(record, features, output_index=None):
    """Library SHAP, in model-output units; all feature contributions retained."""
    model = record['model']; x = features[record['columns']]
    if record['family'] == 'catboost':
        from catboost import Pool
        values = np.asarray(model.get_feature_importance(Pool(x, cat_features=CATEGORIES), type='ShapValues'))[0]
        if values.ndim == 2:
            values = values[output_index]
        base, contribution = float(values[-1]), values[:-1]
        names = list(x.columns)
        raw = np.asarray(model.predict(x, prediction_type='RawFormulaVal')).reshape(-1)
        prediction = float(raw[output_index] if output_index is not None else raw[0])
        observed = {c: None if pd.isna(x.iloc[0][c]) else x.iloc[0][c] for c in names}
        observed = {k: v.item() if isinstance(v, np.generic) else v for k, v in observed.items()}
    else:
        import shap
        transform, estimator = model[:-1], model[-1]
        z = transform.transform(x)
        background = np.atleast_2d(record['background_mean'])
        explanation = shap.LinearExplainer(estimator, background)(z)
        values = np.asarray(explanation.values)[0]
        base = np.asarray(explanation.base_values)[0]
        if values.ndim == 2:
            values, base = values[:, output_index], base[output_index]
        contribution, base = values, float(base)
        names = transform.get_feature_names_out().tolist()
        prediction = np.asarray(estimator.decision_function(z) if hasattr(estimator, 'decision_function') else estimator.predict(z)).reshape(-1)
        prediction = float(prediction[output_index] if output_index is not None else prediction[0])
        observed = {c: float(z[0, i]) for i, c in enumerate(names)}
    reconstruction = float(base + contribution.sum())
    if not np.isclose(prediction, reconstruction, rtol=1e-5, atol=1e-5):
        raise ValueError('SHAP reconstruction failed.')
    rows = [{'feature': n, 'contribution': float(v), 'value': observed[n], 'source': source_group(n)} for n, v in zip(names, contribution)]
    rows.sort(key=lambda r: abs(r['contribution']), reverse=True)
    groups = {}
    for row in rows:
        groups[row['source']] = groups.get(row['source'], 0) + row['contribution']
    return {'base_value': base, 'raw_prediction': prediction, 'reconstructed': reconstruction,
            'features': rows, 'source_contributions': groups,
            'value_units': 'source values' if record['family'] == 'catboost' else 'transformed model inputs',
            'note': 'Contributions explain model behavior, not causality. Correlated inputs may share attribution.'}


@lru_cache(maxsize=2)
def load_bundle(path=str(ROOT / 'artifacts' / 'models.joblib')):
    """Load only trusted local artifacts; joblib can execute code."""
    return joblib.load(path)


def assess(business, bundle=None, explain=True):
    business = business if isinstance(business, Business) else Business.model_validate(business)
    bundle = load_bundle() if bundle is None else bundle
    raw = business.model_dump(); x = make_features(pd.DataFrame([raw]))
    scores = np.clip(predict_record(bundle['scores'], x)[0], 0, 100)
    eligible_probability = float(predict_record(bundle['eligibility'], x, True)[0, 1])
    eligible = eligible_probability >= bundle['eligibility_threshold']
    risk_prob = predict_record(bundle['risk'], x, True)[0]
    risk_index = int(np.argmax(risk_prob))
    risk = str(bundle['risk']['model'].classes_[risk_index])
    limit = float(credit_limit(bundle['limit'], x)[0]) if eligible else 0.0
    anomaly_score = float(-bundle['anomaly'].score_samples(x[ANOMALY_COLUMNS])[0])
    anomaly = anomaly_score >= bundle['anomaly_threshold']
    notes, flags = [], []
    if business.Has_Existing_Loan == 'No': notes.append('EMI payment history: not applicable; no existing loan.')
    elif business.EMI_On_Time_Rate_Pct is None: flags.append('Existing borrower has missing EMI payment history.')
    if x[ANOMALY_COLUMNS].isna().any(axis=None): flags.append('Zero denominator: affected ratios unavailable.')
    if anomaly: flags.append('Unusual financial profile; manual review suggested. Not proof of fraud.')
    if (eligible and risk == 'High') or (not eligible and risk == 'Low'):
        flags.append('Eligibility and risk models disagree; manual review suggested.')
    numeric = x.select_dtypes('number').iloc[0]
    outside = [c for c in numeric.index if c in bundle['feature_bounds'] and pd.notna(numeric[c]) and
               not bundle['feature_bounds'][c][0] <= numeric[c] <= bundle['feature_bounds'][c][1]]
    if outside: flags.append('Outside training range: ' + ', '.join(outside))
    recommendations = []
    surplus = float(x.Bank_Surplus_INR.iloc[0])
    if surplus < 0:
        recommendations.append(f'Bank outflows exceed inflows (monthly deficit ₹{abs(surplus):,.0f}). Review recurring expenses and cash reserves.')
    if business.Avg_Invoice_Payment_Delay_Days > 30:
        recommendations.append(f'Invoice payment delay averages {business.Avg_Invoice_Payment_Delay_Days:g} days. Review collections cycle and credit terms.')
    if business.GST_Filing_Timeliness_Pct < 80:
        recommendations.append(f'GST filing timeliness is {business.GST_Filing_Timeliness_Pct:.1f}% (below 80% benchmark). Strengthen tax compliance discipline.')
    if business.Customer_Concentration_Ratio > .6:
        recommendations.append(f'Top customers represent {business.Customer_Concentration_Ratio*100:.0f}% of revenue. Mitigate revenue concentration risks.')
    if business.Overdraft_Usage_Ratio > .7:
        recommendations.append(f'Overdraft utilization is {business.Overdraft_Usage_Ratio*100:.0f}%. High credit line reliance increases refinancing sensitivity.')
    if business.Salary_Consistency_Pct < 80:
        recommendations.append(f'Salary consistency is {business.Salary_Consistency_Pct:.1f}%. Irregular wage payments indicate operational cash flow strain.')
    if business.Employee_Attrition_Rate_Pct > 20:
        recommendations.append(f'Employee attrition is {business.Employee_Attrition_Rate_Pct:.1f}%. High workforce turnover poses operational continuity risks.')
    if business.Revenue_Growth_Rate_Pct < 0:
        recommendations.append(f'Reported revenue contracted by {abs(business.Revenue_Growth_Rate_Pct):.1f}%. Review market demand and product margins.')
    if business.Transaction_Volatility_Index > .4:
        recommendations.append('High transaction volatility indicates uneven cash inflows. Maintain liquidity buffers.')
    if not recommendations:
        recommendations.append('Maintain filing discipline, payment timeliness, and regular cash-flow review.')
    result = {'MSME_ID': business.MSME_ID, 'created_at': datetime.now(timezone.utc).isoformat(),
              'model_version': bundle['version'], 'scores': {k: round(float(v), 2) for k, v in zip(SCORES, scores)},
              'credit': {'eligible': bool(eligible), 'eligibility_probability': eligible_probability,
                         'risk_category': risk, 'risk_probabilities': dict(zip(map(str, bundle['risk']['model'].classes_), map(float, risk_prob))),
                         'recommended_limit_inr': round(limit), 'status': 'Review required' if flags else 'Indicative recommendation'},
              'anomaly': {'score': anomaly_score, 'threshold': bundle['anomaly_threshold'], 'flagged': bool(anomaly)},
              'risk_indicators': flags, 'data_quality_notes': notes, 'recommendations': recommendations,
              'snapshot': raw, 'disclaimer': DISCLAIMER}
    if explain:
        explanations = {'health': explain_record(bundle['scores'], x, 6),
                        'eligibility': explain_record(bundle['eligibility'], x),
                        'risk': explain_record(bundle['risk'], x, risk_index),
                        'positive_limit': explain_record(bundle['limit'], x)}
        for key, units in [('health', 'score points'), ('eligibility', 'log odds of eligibility'),
                           ('risk', 'raw class margin'), ('positive_limit', 'log1p INR before eligibility gate')]:
            explanations[key]['output_units'] = units
        result['explanations'] = explanations
        health = explanations['health']['features']
        result['strengths'] = [r for r in health if r['contribution'] > 0][:3]
        result['concerns'] = [r for r in health if r['contribution'] < 0][:3]
    return result


@lru_cache(maxsize=1)
def get_card_template():
    return (ROOT / 'templates/health_card_template.html').read_text(encoding='utf-8')


def card_html(card):
    """Render executive printable Financial Health Card using external HTML template."""
    def e(v): return escape(str(v))
    scores = card['scores']
    overall = scores.get('Financial_Health_Score', 0)
    badge_color = '#15803d' if overall >= 75 else ('#1d4ed8' if overall >= 60 else ('#b45309' if overall >= 45 else '#b91c1c'))
    badge_tier = 'EXCELLENT' if overall >= 75 else ('GOOD' if overall >= 60 else ('FAIR' if overall >= 45 else 'HIGH RISK'))
    risk_color = '#15803d' if card['credit']['risk_category'] == 'Low' else ('#b45309' if card['credit']['risk_category'] == 'Medium' else '#b91c1c')

    dimension_names = [('Business Stability', 'Business_Stability_Score'),
                       ('Cash Flow', 'Cashflow_Score'),
                       ('Revenue Consistency', 'Revenue_Consistency_Score'),
                       ('Payment Behaviour', 'Payment_Behaviour_Score'),
                       ('Business Growth', 'Business_Growth_Score'),
                       ('Compliance', 'Compliance_Score')]
    dim_cards = []
    for title, key in dimension_names:
        val = scores.get(key, 0)
        bar_color = '#2563eb' if val >= 70 else ('#d97706' if val >= 50 else '#dc2626')
        dim_cards.append(f'''
        <div class="dim-card">
            <div class="dim-title">{e(title)}</div>
            <div class="dim-val">{val:.1f}</div>
            <div class="bar-bg"><div class="bar-fill" style="width:{min(100, max(0, val)):.1f}%;background:{bar_color}"></div></div>
        </div>''')

    flags_html = ''.join(f'<div class="alert-item alert-warning">⚠️ {e(f)}</div>' for f in card.get('risk_indicators', []))
    recs_html = ''.join(f'<div class="alert-item alert-rec">• {e(r)}</div>' for r in card.get('recommendations', []))
    notes_html = ''.join(f'<div class="alert-item alert-note">ℹ️ {e(n)}</div>' for n in card.get('data_quality_notes', []))

    health = card.get('explanations', {}).get('health', {})
    reasons_rows = ''.join(f'''<tr>
        <td style="font-weight:500">{e(r["feature"].replace("_", " "))}</td>
        <td><span class="source-tag">{e(r.get("source", "Feature"))}</span></td>
        <td style="text-align:right;font-weight:600;color:{"#15803d" if r["contribution"]>0 else "#b91c1c"}">{r["contribution"]:+.2f} pts</td>
    </tr>''' for r in health.get('features', [])[:8])

    raw = card.get('snapshot', {})
    template = get_card_template()
    replacements = {
        '{{MSME_ID}}': e(card['MSME_ID']),
        '{{BADGE_COLOR}}': badge_color,
        '{{BADGE_TIER}}': badge_tier,
        '{{RISK_COLOR}}': risk_color,
        '{{RISK_CATEGORY}}': e(card['credit']['risk_category']),
        '{{ELIGIBILITY_STATUS}}': 'Indicatively Eligible' if card['credit']['eligible'] else 'Not Eligible',
        '{{ELIGIBILITY_COLOR}}': '#15803d' if card['credit']['eligible'] else '#b91c1c',
        '{{ELIGIBILITY_PROB}}': f'{card["credit"]["eligibility_probability"]*100:.1f}',
        '{{CREDIT_STATUS}}': e(card['credit']['status']),
        '{{CREDIT_LIMIT}}': f'{card["credit"]["recommended_limit_inr"]:,.0f}',
        '{{OVERALL_SCORE}}': f'{overall:.1f}',
        '{{CUSTOMER_SEGMENT}}': e(raw.get('Customer_Segment', 'MSME')),
        '{{INDUSTRY}}': e(raw.get('Industry', 'Industry')),
        '{{BUSINESS_TYPE}}': e(raw.get('Business_Type', 'Business')),
        '{{TURNOVER}}': f'{raw.get("Annual_Turnover_INR", 0):,.0f}',
        '{{MODEL_VERSION}}': e(card['model_version']),
        '{{DIMENSION_CARDS}}': ''.join(dim_cards),
        '{{RISK_FLAGS}}': flags_html,
        '{{RECOMMENDATIONS}}': recs_html,
        '{{DATA_NOTES}}': notes_html,
        '{{EXPLANATION_ROWS}}': reasons_rows,
        '{{DISCLAIMER}}': e(card['disclaimer']),
        '{{ASSESSMENT_ID}}': e(card.get('assessment_id', 'DEMO-LOCAL')),
        '{{CREATED_AT}}': e(card['created_at'][:19].replace('T', ' ')),
    }
    for k, v in replacements.items():
        template = template.replace(k, v)
    return template
