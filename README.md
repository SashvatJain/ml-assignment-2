# Machine Learning Assignment 2 - Classification Models & Streamlit Deployment

**Course:** M.Tech (AIML) - Machine Learning  
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)  

---

### a. Problem Statement
The objective of this project is to build, evaluate, and deploy an end-to-end Machine Learning classification pipeline. Using clinical diagnostic measurements of cell nuclei, the binary classification task predicts whether a breast mass tumor is **Malignant (0)** or **Benign (1)**. Six ML models are trained, evaluated across various performance metrics, and deployed via an interactive Streamlit web dashboard to simulate a real-world ML deployment workflow.

### b. Dataset Description
* **Source:** UCI Machine Learning Repository (Breast Cancer Wisconsin Diagnostic Dataset)
* **Instances:** 569 instances (Minimum required: 500)
* **Features:** 30 continuous numerical features (Minimum required: 12) computed from digitized images of fine needle aspirate (FNA) of breast mass.
* **Target:** Binary classification (`0` = Malignant, `1` = Benign)
* **Preprocessing:** Train-Test Split (80/20), Feature Scaling using `StandardScaler` for distance and gradient-based models.

### c. Github Repository Link
* **GitHub Repository:** [Github](https://github.com/SashvatJain/ml-assignment-2)
* **Live Streamlit Web App:** [Streamlit](https://2025ac05819-sashvatjain.streamlit.app/)

### d. Models used:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **kNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest** | 0.9561 | 0.9937 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Gradient Boosting** | 0.9561 | 0.9907 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved outstanding classification capability with the highest Accuracy (0.9825) and MCC (0.9623). The linear decision boundary fits the scaled high-dimensional biomedical features highly efficiently. |
| **Decision Tree** | Shows the lowest baseline performance (Accuracy 0.9123, MCC 0.8174). It exhibits some variance and overfitting compared to the ensemble techniques on this dataset. |
| **kNN** | Strong performance (Accuracy 0.9561, F1 0.9655) after standardized feature scaling, leveraging spatial proximity effectively in the feature space. |
| **Naive Bayes** | Yields solid baseline results (Accuracy 0.9298, AUC 0.9868) despite strong feature independence assumptions among correlated nuclear measurements. |
| **Random Forest** | Excellent performance with low variance (Accuracy 0.9561, AUC 0.9937), handling multi-collinearity well across tree ensembles and matching kNN in accuracy. |
| **Gradient Boosting** | Exceptional generalization (Accuracy 0.9561, Recall 0.9861), balancing precision and recall symmetrically across test splits, matching Logistic Regression in Recall. |
| **Overall Winner for your dataset?** | **Logistic Regression** is the overall best performer for this dataset, achieving the top Accuracy (0.9825), AUC (0.9954), F1 Score (0.9861), and MCC (0.9623). |
