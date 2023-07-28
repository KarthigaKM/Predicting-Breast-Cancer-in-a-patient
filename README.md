# [MAIN PROJECT : Predicting Breast Cancer in a patient](https://github.com/KarthigaKM/Predicting-Breast-Cancer-in-a-patient)
## Breast Cancer Prediction Web App
 * This project is a Streamlit-based web application for predicting breast cancer using machine learning algorithms. The app allows users to perform data preprocessing, exploratory data analysis (EDA), and model evaluation on breast cancer data.

## How to Use the App
 * Install the required libraries by running pip install -r requirements.txt.
 * Run the app locally by executing streamlit run app.py in your terminal.
 * The app will open in your web browser.
 * Use the sidebar menu to navigate through different sections of the app:
### Home: 
![](https://github.com/KarthigaKM/Predicting-Breast-Cancer-in-a-patient/blob/main/brea%20cancer%201.PNG?raw=true)
 * Provides an overview of breast cancer, its impact, and the importance of early detection and prevention.
### Exploratory Data Analysis (EDA): 
![](https://github.com/KarthigaKM/Predicting-Breast-Cancer-in-a-patient/blob/main/images/pairplot.png?raw=true) ![](https://github.com/KarthigaKM/Predicting-Breast-Cancer-in-a-patient/blob/main/images/Brea%20cancer%20correla.png?raw=true)
 * Visualizes the dataset, displaying pair plots and distribution of diagnoses (Malignant and Benign). Also, shows a heatmap of feature correlations.
### Data Preprocessing:
![](https://github.com/KarthigaKM/Predicting-Breast-Cancer-in-a-patient/blob/main/images/brea%20cancer%20piechart.png?raw=true)
 * Demonstrates data preprocessing steps, such as splitting the data, scaling features, and dealing with class imbalance.
### Validation Curve: 
![](https://github.com/KarthigaKM/Predicting-Breast-Cancer-in-a-patient/blob/main/images/brea%20cnacer%20val%20curve.png?raw=true)
 * Evaluates the model's performance using the validation curve for the Support Vector Classifier (SVC) with Radial Basis Function (RBF) kernel. Helps identify underfitting and overfitting.
### Learning Curve:
![](https://github.com/KarthigaKM/Predicting-Breast-Cancer-in-a-patient/blob/main/images/brae%20cnacer%20learn%20curve.png?raw=true)
 * Analyzes the learning curve of the model, showing the training and validation scores for different training set sizes.
### Other Metrics: 
 * Displays other accuracy metrics, such as accuracy score, confusion matrix, classification report, precision, recall, specificity, and area under the ROC curve (AUC ROC).
### Ensemble Model Metrics: 
 * Utilizes a combination of Random Forest and Gradient Boosting classifiers to create an ensemble model.
 * Evaluates the ensemble model's performance using accuracy, confusion matrix, and classification report.

## Dataset
 * The breast cancer data is loaded from a CSV file (cancer.csv), containing information on various features of breast tumor cells.
 * The target variable is the diagnosis, with 'M' representing Malignant and 'B' representing Benign.

## Model Building
 * The app uses Support Vector Classifier (SVC) with a Radial Basis Function (RBF) kernel and an ensemble model with Random Forest and Gradient Boosting classifiers.

## Data Preprocessing
 * Data preprocessing involves splitting the dataset into training and testing sets, scaling the features using StandardScaler, and handling class imbalance using RandomOverSampler.

## Exploratory Data Analysis (EDA)
 * The EDA section presents visualizations of feature correlations and distribution of diagnoses using pair plots and count plots, respectively.

## Validation and Learning Curves
 * The validation and learning curves are used to assess model performance and identify potential underfitting or overfitting.

## Other Metrics
 * Other accuracy metrics, including precision, recall, specificity, and area under the ROC curve (AUC ROC), are provided for model evaluation.

## Ensemble Model
 * The ensemble model combines the predictions of Random Forest and Gradient Boosting classifiers using a majority vote.

## Acknowledgments
 * The breast cancer dataset used in this project is sourced from GUVI.
 * The app utilizes the Streamlit library for interactive web development.
 * Throughout the code, you use various libraries such as sklearn for machine learning algorithms, pandas for data manipulation, seaborn and matplotlib for data 
   visualization, and streamlit for building the web application.

## Future Enhancements 
 * Multi-Class Classification: Extend the project to handle multi-class classification if relevant to the application.
 * Error Analysis: Perform in-depth error analysis to understand the types of errors the model makes and identify areas for improvement.
 * User Authentication: If sensitive data is involved, consider adding user authentication and access control to protect user privacy.

## About
 * This project is part of IITM Advanced Programming Professional in Master Data Science Course,GUVI, IIT Madras Research Park



