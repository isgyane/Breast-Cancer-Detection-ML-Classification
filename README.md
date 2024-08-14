# Breast Cancer Detection

### Project Overview
Breast cancer is a significant health concern, hence early detection is crucial for successful treatment.
This project aims to develop and evaluate 3 machine learning models to predict whether a tumor is benign or malignant based on features derived from a breast cancer dataset. The project focuses on using various supervised learning algorithms to achieve high accuracy in classification.

### Dataset
The dataset used is the Wisconsin Breast Cancer Dataset, which contains 506 samples and 32 features.
* Link to Dataset - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
* Target - The target variable is `diagnosis`, which indicates whether the tumor is benign (B) or malignant (M)
* Data Preprocessing steps
    * Statisical Summarization
    * Removed a Column with missing values
    * Handling of Categorical variables
    * Correlation analysis within features and between features and target variable
    * Feature scaling using Normalization technique

### Project Directory
* data - Contains the dataset file (`data/breast-cancer-wisconsin-data_work.csv`)
* notebooks - Contains Jupyter Notebooks for data exploration, modeling, and evaluation
* scripts - Stores python re-usable python code
* results - Contains exported images for results

### Dependencies
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

### Usage
1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook named `notebooks/breast_cancer_detection.ipynb`.

### Models
* Logistic Regression
* Random Forest
* Support Vector Machine

### Results
The best performing model is Logistic Regression with a cross validation accuracy of 97.36%.

### Future Work
* Remove features with high self-correlation.
* Remove features with low correlation to the target variable
* Analyse for feature importance
* Modularize the analysis - https://www.youtube.com/watch?v=53VCqbceq2U

### Contributing
Contributions are welcome! Please follow these steps to contribute:
    Fork the repository.
    Create a new branch (git checkout -b feature/YourFeature).
    Make your changes.
    Commit your changes (git commit -m 'Add new feature').
    Push to the branch (git push origin feature/YourFeature).
    Open a Pull Request.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Acknowledgements
Scikit-learn Documentation for the extensive machine learning tools.
UCI Machine Learning Repository for providing the Breast Cancer Wisconsin (Diagnostic) Data Set.

### Resources Used
* Extracting metrics from a classification report - "https://stackoverflow.com/questions/48417867/access-to-numbers-in-classification-report-sklearn"
* Importing a notebook into another notebbook - dhttps://stackoverflow.com/questions/20186344/importing-an-ipynb-file-from-another-ipynb-file
* Choosing the right estimator - https://scikit-learn.org/stable/machine_learning_map.html#choosing-the-right-estimator