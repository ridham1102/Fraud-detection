# Fraud Transaction Detection with XGBoost

In this project, I developed a machine learning solution to detect fraudulent financial transactions using a synthetic dataset of over 6.3 million rows. This dataset simulated 30 days of activity and included a variety of transaction types—such as PAYMENT, CASH_OUT, TRANSFER, DEBIT, and CASH_IN—with features like transaction amounts, sender and receiver balances, and binary flags for fraud status. The primary objective was to build an accurate fraud detection model while maintaining data integrity, ensuring fairness, and addressing all the typical data science challenges such as missing values, outliers, class imbalance, and high cardinality categorical features.

### Getting Started

The process began with data exploration. I inspected the dataset size, structure, and data types to understand what I was dealing with. I discovered three object-type columns: `type`, `nameOrig`, and `nameDest`, which required preprocessing. The `type` column was categorical and had to be converted into numerical form using one-hot encoding. The other two—representing sender and recipient names—had thousands of unique values and were not directly useful in their raw form, so I engineered features from them instead.

A critical observation was the presence of merchant accounts, which had names starting with "M" and were missing proper balance details. These accounts dominated the dataset, making up over 5.6 million rows. Dropping them would reduce the dataset drastically, but imputing with mean or median values would distort reality. So I decided to keep them by introducing a new flag called `isMerchant`, replacing invalid balances with `NaN`, and letting models like XGBoost handle the missing values natively.

### Data Cleaning & Preprocessing

In this phase, I handled several issues:

- **Missing Values**: Old and new destination balances were missing for merchants. I marked these rows with a binary `isMerchant` flag and replaced their values with `NaN`.
- **Outliers**: Using IQR-based filtering, I identified and visualized large transactions that could influence model behavior. Most frauds involved high-value transactions, so instead of removing these, I retained them as they carried useful patterns.
- **Multicollinearity**: I used a correlation heatmap to explore relationships between features. Sender and recipient balances before and after transactions showed high correlation, so I engineered features like balance deltas and ratios to better capture dynamics.
- **Feature Engineering**: I created behavioral and contextual features such as `delta_balanceOrig`, `delta_balanceDest`, and `amount_to_balance_ratio` to represent the net flow of funds and anomalies in updates. I also used the `step` column to simulate time-based features like bursts of activity.

### Modeling

I used **XGBoostClassifier**, which is powerful for large tabular datasets and handles missing values efficiently. I started with default parameters and then optimized them with a basic grid search for better performance. Due to the severe class imbalance (fraud was extremely rare), I used stratified splitting to preserve the fraud distribution in train and test sets. I also attempted SMOTE to oversample the fraud class, but it didn’t work due to `NaN` values. Rather than dropping or imputing again, I stuck to tree-based models that work well with incomplete data.

I trained and evaluated two models:

1. **Model with full data including merchants (with NaNs):**
   - Accuracy: 0.9997
   - Precision: 93%
   - Recall: 83%
   - F1 Score: 0.88

2. **Model with merchants dropped (no NaNs):**
   - Accuracy: 0.9994
   - Precision: 84%
   - Recall: 66%
   - F1 Score: 0.74

This comparison clearly showed that keeping merchant data (even with missing values) gave much better fraud detection capability, especially in recall.

### Evaluation

I used a full suite of metrics:
- **Accuracy**: Mostly to ensure the model isn’t broken
- **Confusion Matrix**: To see true/false positives and negatives
- **Precision & Recall**: To measure how well the model detects fraud
- **F1 Score**: A balance between precision and recall, which is crucial in fraud detection

Using `classification_report`, I summarized all the metrics, and `feature_importances_` from XGBoost helped me understand which features mattered most. The most important ones were:
- `amount`
- `oldbalanceOrg`
- `delta_balanceOrig`
- `isMerchant`
- `amount_to_balance_ratio`

These made sense because frauds often involved emptying sender accounts in large chunks, frequently to merchant-like destinations.

### Real Challenges Faced

This was my first time working with such a massive dataset. Loading and processing over 6 million rows in Google Colab brought memory issues and slow performance. I had to reset RAM multiple times and optimize my workflow. Even heatmaps were slow to render, and model training required GPU acceleration. SMOTE failed due to `NaN`s, and I had to debug and rethink the whole oversampling plan. But these struggles were valuable—through them, I learned how to use missing values to my advantage, when not to drop data, and how to maintain balance between model simplicity and accuracy.

### Final Thoughts

This project gave me hands-on experience with real-world complexities:
- Working with large-scale, noisy datasets
- Deciding when to drop versus transform data
- Feature engineering based on logic and domain understanding
- Choosing the right model based on data type and structure
- Building pipelines that perform well without overfitting
- Thinking beyond accuracy to detect edge-case behaviors like fraud

I also learned the trade-offs between recall and precision, and how even a 1% drop in recall could mean thousands of missed frauds in the real world. That’s why I chose to keep the challenging merchant data instead of going for a cleaner but less informative dataset.

Overall, this end-to-end fraud detection system—from data loading to model evaluation—taught me not just the mechanics of machine learning, but the judgment that separates good data scientists from great ones. While the results are already strong, I still have doubts and aim to go deeper—perhaps using SHAP for explainability, or graph analysis for account network patterns—in future iterations.
