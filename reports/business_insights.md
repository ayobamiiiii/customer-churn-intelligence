# Customer Churn Intelligence
## Business Insights and Retention Strategy

### 1. Executive Summary

This project analyzes customer churn for a telecommunications company using exploratory data analysis, machine learning, SHAP explainability, and survival analysis.

The dataset contains 7,043 customers, of whom 1,869 churned. The overall churn rate is approximately **26.5%**.

The analysis identifies contract type, customer tenure, customer support, and competitive offerings as important areas for retention investigation.

### 2. Key Findings

**Contract type**

- Month-to-month customers: 42.7% churn rate.
- One-year contract customers: 11.3%.
- Two-year contract customers: 2.8%.

Customers on month-to-month contracts have substantially higher observed churn rates than customers on longer contracts.

**Customer tenure**

- 1–12 months: 47.44% churn rate.
- 13–24 months: 28.71%.
- 25–36 months: 21.63%.
- 37–48 months: 19.03%.
- 49–60 months: 14.42%.
- 61–72 months: 6.61%.

The first year is the highest-risk period in this dataset.

**Recorded churn reasons**

The leading recorded reasons include:

1. Attitude of support personnel: 192 customers.
2. Competitors offering higher download speeds: 189.
3. Competitors offering more data: 162.
4. Unknown reason: 154.
5. Competitors offering better deals: 140.

These responses indicate areas for further investigation; recorded reasons do not establish causation.

### 3. Machine Learning Results

The LightGBM baseline model achieved:

- Accuracy: 78.25%.
- Precision: 59.83%.
- Recall: 55.35%.
- F1 score: 57.50%.
- ROC-AUC: 83.88%.

After adjusting the classification threshold to 0.35:

- Accuracy: 73.13%.
- Precision: 49.68%.
- Recall: 82.62%.
- F1 score: 62.05%.
- ROC-AUC: 83.86%.

The lower threshold detects more customers who churn, but it also generates more false positives. The appropriate threshold depends on the business cost of missed churners versus unnecessary retention interventions.

### 4. Recommendations

**Improve first-year retention**

Introduce structured onboarding, early satisfaction surveys, and proactive follow-ups during the first 12 months.

**Investigate customer support**

Review complaint resolution times, customer feedback, and support quality. Test improvements and measure their effect on churn.

**Review competitive positioning**

Benchmark internet speed, data allowances, and plan pricing against competitors.

**Target retention campaigns**

Use model predictions to prioritize customers at elevated risk. Evaluate campaign effectiveness using controlled experiments rather than assuming that prediction alone prevents churn.

**Improve churn-reason data**

Investigate unknown churn reasons and improve feedback collection when customers leave.

### 5. Limitations

- The findings describe associations in historical customer data and do not prove causation.
- Churn reasons are self-reported or recorded categories and may not capture every contributing factor.
- Model performance should be validated on unseen data and compared with business intervention costs.
- Survival estimates depend on how tenure and churn events are defined and should be interpreted alongside the model and descriptive analyses.

### 6. Conclusion

The strongest immediate priorities are first-year retention, customer support quality, and competitive plan positioning. Combining predictive modeling with measurable retention experiments can help the business direct its resources toward customers most likely to leave.

### 7. Project Artifacts

Analysis charts are available in `reports/figures/`, including:

- `churn_by_contract.png` or `survival_by_contract.png`
- `churn_by_tenure_group.png`
- `top_churn_reasons.png`
- `shap_feature_importance.png`
- `shap_summary.png`
- `kaplan_meier_survival.png`

Actual filenames should be checked against the files present in the project.


## Tuned Model Findings

### Model Performance

The tuned LightGBM classifier achieved a test ROC-AUC of **0.8497**, indicating good ability to distinguish between customers who churn and those who remain.

At the default classification threshold of 0.50:

- Accuracy: 74.13%
- Precision: 50.85%
- Recall: 79.68%
- F1-score: 62.08%

The model identified 298 of 374 actual churn cases in the test set, while flagging 288 non-churning customers as potential churners.

### Key Predictive Features

The tuned model's feature-importance analysis identified the following leading predictors:

1. Monthly Charges
2. Tenure Months
3. Total Charges
4. Dependents
5. Electronic-check payment method
6. Contract type

These findings suggest that customer spending, relationship duration, payment behaviour, and contract arrangements are important signals for identifying customers at risk of churn.

Feature importance measures predictive contribution, not causation. SHAP analysis should be used to examine whether specific feature values tend to increase or decrease individual churn predictions.

### Recommended Retention Actions

- **Early-tenure customers:** Introduce onboarding support and proactive engagement during the first year.
- **High-charge customers:** Review plan suitability and offer appropriate alternatives where value concerns are identified.
- **Electronic-check customers:** Investigate payment friction and encourage convenient, reliable payment options.
- **Contract strategy:** Test incentives for longer-term contracts while monitoring conversion and retention outcomes.
- **Targeted retention:** Prioritise customers with high predicted churn risk and measure campaign effectiveness against a control group.

### Limitations

The model identifies statistical patterns in historical customer data and does not establish causal relationships. Retention campaigns should be tested before broad deployment. Model performance and the selected decision threshold should be reviewed periodically.

