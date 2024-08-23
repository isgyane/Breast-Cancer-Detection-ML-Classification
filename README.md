# Breast Cancer Detection

### Project Overview
Breast cancer is a significant health concern, with early detection being crucial for improving patient outcomes and reducing treatment costs. This project demonstrates the development and evaluation of three machine learning models aimed at accurately predicting whether a breast tumor is benign or malignant. By leveraging these models, healthcare providers can potentially make more informed decisions, leading to earlier interventions and better patient prognosis.

### Business Value
The primary goal of this project is to provide a reliable, automated tool for breast cancer detection that can assist radiologists and oncologists in making quicker and more accurate diagnoses. Implementing this tool could lead to:
* **Reduced Diagnostic Time:** Automating the initial screening process, allowing healthcare professionals to focus on more complex cases.
* **Improved Accuracy:** Enhancing diagnostic precision, reducing the likelihood of false positives and negatives, and consequently improving patient trust and treatment outcomes.
* **Cost Savings:** By identifying malignant tumors earlier, the treatment can be less invasive and more cost-effective, reducing the financial burden on patients and healthcare systems

### Dataset
The dataset used is the Wisconsin Breast Cancer Dataset, which contains 569 samples and 32 features.
* Link to Dataset - [Wisconsin Breast Cancer Dataset]([url](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)) 
* The target variable - `diagnosis` - indicates whether the tumor is benign (B) or malignant (M)

#### **Key Data Preprocessing steps**
    * Statistical Summarization and handling of missing values
    * Categorical variable encoding and correlation analysis   
    * Feature scaling using normalization techniques

### Project Directory
* data: Contains the dataset file.
* notebooks: Jupyter Notebooks for data exploration, modeling, and evaluation.
* scripts: Reusable Python code for model training and evaluation.
* results: Exported images and results from model evaluations

### Dependencies
This project was built using Python and the following key libraries:
* `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Usage
1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook named `notebooks/breast_cancer_detection.ipynb`.

### Key Models Developed
* **Logistic Regression:** Achieved a cross-validation accuracy of 97.36%, demonstrating strong predictive performance.
* **Random Forest:** Offers interpretability and robustness, useful for clinical settings.
* **Support Vector Machine:** Effective in high-dimensional spaces, providing additional insights into tumor classification.

### Results and Validation
Using the Stratified cross-validation score, the SVC model outperformed the others with a score of 97.36%, making it the most reliable model for this dataset. This high accuracy suggests that the model is well-suited for **initial breast cancer screening** in a clinical setting.

### Future Work and Recommendations
* **Hyperparameter Tuning:** Experiment with different model hyperparameters to improve predictive accuracy.
* **Feature Engineering:** Remove or combine features with high self-correlation and low correlation with the target variable to reduce overfitting and improve model interpretability.
* **Model Deployment:** Integrate the model into a web-based application for real-time predictions, enabling broader access for healthcare providers.

### Contributing
Contributions are welcome! Please follow these steps to contribute:
* Fork the repository and create a new branch
* Implement your changes and commit them with a clear message
* Push to the branch and open a Pull Request for review.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Key Resources Used
* [Extracting metrics from a classification report]([url](https://stackoverflow.com/questions/48417867/access-to-numbers-in-classification-report-sklearn))
* [Importing a notebook into another notebbook]([url](https://stackoverflow.com/questions/20186344/importing-an-ipynb-file-from-another-ipynb-file))
* [Choosing the right estimator]([url](https://scikit-learn.org/stable/machine_learning_map.html#choosing-the-right-estimator))
