# Breast Cancer Wisconsin (Diagnostic) – Classification Model Comparison

## a. Problem statement

Build and compare five supervised classification models to classify breast tumour cases as benign (B) or malignant (M) using the Breast Cancer Wisconsin (Diagnostic) dataset. Malignant is treated as the positive class for precision, recall, F1, AUC and MCC.

## b. Dataset description

- Source: UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)
- 569 instances and 30 numerical predictor features
- Target: `diagnosis` (B = benign, M = malignant)
- Original patient ID excluded from model inputs
- Class counts: 357 benign (62.74%) and 212 malignant (37.26%)
- No missing values in the supplied dataset

## c. Github Repository Link
https://github.com/ron-daniel1/ML-BrestCancerImageprocessing.git

## d. Models used and evaluation

The dataset was split using an 80:20 stratified train-test split (`random_state=42`). StandardScaler was used in pipelines for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes were fitted without scaling. The models were evaluated on the same held-out test set of 114 records.

| ML Model               | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
| ---------------------- | -------: | -----: | --------: | -----: | -----: | -----: |
| Logistic Regression    |   0.9649 | 0.9960 |    0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree          |   0.9298 | 0.9246 |    0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN                    |   0.9561 | 0.9823 |    0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes   |   0.9386 | 0.9934 |    1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine |   0.9737 | 0.9947 |    1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Model performance observations

- **Logistic Regression:** Strong performance with 96.49% accuracy and the highest AUC (0.9960). Malignant recall was 0.9286.
- **Decision Tree:** Lowest overall performance in this experiment, with 92.98% accuracy and 0.9246 AUC. It is comparatively interpretable but may be less stable.
- **KNN:** Achieved 95.61% accuracy. Scaling is important for distance-based comparisons; malignant recall was 0.9048.
- **Gaussian Naive Bayes:** Achieved perfect malignant precision (1.0000) but the lowest malignant recall (0.8333), missing 7 of 42 malignant test cases.
- **Support Vector Machine:** Highest accuracy (97.37%), precision (1.0000), F1 (0.9630) and MCC (0.9442) in this split. It correctly identified 39 of 42 malignant cases.

### Overall winner

The Support Vector Machine is the strongest overall model on this held-out test split based on accuracy, F1 and MCC, while Logistic Regression achieved the highest AUC. SVM and Logistic Regression both achieved malignant recall of 0.9286. The choice of winner depends on the evaluation priority; for the aggregate metrics in this experiment, SVM is selected.

## Streamlit application

The application allows a user to upload a labelled test CSV, select one of the five saved models, and view accuracy, AUC, precision, recall, F1, MCC, a confusion matrix and a classification report. The app loads saved model files and does not retrain on uploaded test data.

## How to run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. Upload `test_data.csv` in the app.

## Deployment

Push the project folder contents to a GitHub repository. In Streamlit Community Cloud, create a new app, select the repository and branch, set the main file path to `app.py`, and deploy.

## Limitations

This is a benchmark classification exercise, not a clinically validated diagnostic system. Performance is based on one stratified hold-out split and should not be interpreted as evidence of real-world clinical effectiveness.
