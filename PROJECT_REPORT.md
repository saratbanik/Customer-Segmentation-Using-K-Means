# Internship Task 2 Report

## Project title
**Customer Segmentation Using K-Means Clustering**

## 1. Objective
Apply unsupervised machine learning to group retail mall customers into interpretable segments using annual income and spending score. The goal is to explore how customer groups differ and demonstrate segment assignment for a new example customer.

## 2. Dataset
The supplied `Mall_Customers.csv` has 200 rows and five columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. In the supplied dataset, there are no missing values and no duplicate rows. Annual income is expressed in thousands of dollars; the spending score ranges from 1 to 100.

**Scope note:** The dataset does not include transaction-level purchase histories. Annual income and spending score are used as proxy features. Gender, age, and CustomerID are retained for descriptive profiling only and are not used by K-means to form the clusters.

## 3. Tools and technologies
Python, pandas, NumPy, scikit-learn, Matplotlib, Plotly, Streamlit, and joblib.

## 4. Methodology
1. Validate required columns and check missing values / customer IDs.
2. Select annual income and spending score.
3. Standardize both numeric features using `StandardScaler`.
4. Evaluate cluster counts from k=2 to k=10 using inertia and silhouette score.
5. Select the k with the highest silhouette score in this search range.
6. Fit K-means with `random_state=42` and `n_init=20`.
7. Assign interpretable segment names based on the cluster centers, save segment profiles, export the segmented data, and create diagnostic charts.

## 5. Results and evaluation
The dataset contains **200 customers**. The selected clustering features have **0 missing values** and there are **0 duplicate customer IDs**. A search over k=2 through k=10 selected **k=5**, which gave the highest silhouette score in that range.

- **Selected k:** 5
- **Silhouette score:** 0.5547
- **Inertia:** 65.5684
- **Random seed:** 42

Segment profile for this run:

| Segment | Customer count | Average annual income (k$) | Average spending score |
|---|---:|---:|---:|
| Mid-range, moderate-spending | 81 | 55.30 | 49.52 |
| High-income, high-spending | 39 | 86.54 | 82.13 |
| High-income, low-spending | 35 | 88.20 | 17.11 |
| Lower-income, low-spending | 23 | 26.30 | 20.91 |
| Lower-income, high-spending | 22 | 25.73 | 79.36 |

Silhouette score is a relative measure of cluster cohesion and separation, not a classification-accuracy percentage. Inertia is the within-cluster sum of squared distances and generally decreases as k increases. Results are also saved to `artifacts/metrics.json`, with the full k comparison stored in `artifacts/cluster_selection_scores.csv`.

## 6. Deliverables
- `train_model.py`: data validation, model selection, training, evaluation, and charts.
- `predict_segment.py`: command-line segment assignment for a sample customer.
- `app.py`: interactive Streamlit dashboard.
- `house_price_customer_segmentation.ipynb`: step-by-step notebook.
- `artifacts/`: saved model, scores, cluster profiles, segmented CSV, and plots.

## 7. Limitations
K-means groups customers based on distance in the selected feature space. It does not explain why a customer has a particular spending score, predict future purchases, or establish causation. The five-segment naming rules are descriptive labels for this dataset and may need revision for other data. The dataset is small and illustrative. More useful commercial segmentation would require verified transaction history, recency, frequency, monetary value, product categories, and validation with business stakeholders.

## 8. Conclusion
The project demonstrates an end-to-end unsupervised-learning workflow: data checks, feature scaling, cluster-count assessment, K-means modeling, cluster interpretation, visual exploration, and assignment of a new customer to a learned segment. The dataset's limitations are explicitly stated to avoid overstating what the model can infer.
