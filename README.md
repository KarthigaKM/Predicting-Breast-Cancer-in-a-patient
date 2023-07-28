# [MAIN PROJECT : Predicting Breast Cancer in a patient]()
## Breast Cancer Prediction Web App
 * This project is a Streamlit-based web application for predicting breast cancer using machine learning algorithms. The app allows users to perform data preprocessing, exploratory data analysis (EDA), and model evaluation on breast cancer data.

## How to Use the App
 * Install the required libraries by running pip install -r requirements.txt.
 * Run the app locally by executing streamlit run app.py in your terminal.
 * The app will open in your web browser.
 * Use the sidebar menu to navigate through different sections of the app:
### Home: 
 * Provides an overview of breast cancer, its impact, and the importance of early detection and prevention.
### Exploratory Data Analysis (EDA): 
 * Visualizes the dataset, displaying pair plots and distribution of diagnoses (Malignant and Benign). Also, shows a heatmap of feature correlations.
### Data Preprocessing: 
 * Demonstrates data preprocessing steps, such as splitting the data, scaling features, and dealing with class imbalance.
### Validation Curve: 
 * Evaluates the model's performance using the validation curve for the Support Vector Classifier (SVC) with Radial Basis Function (RBF) kernel. Helps identify underfitting and overfitting.
### Learning Curve: 
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
 * The breast cancer dataset used in this project is sourced from (mention the data source if applicable).
 * The app utilizes the Streamlit library for interactive web development.
 * The machine learning models are built using scikit-learn and imbalanced-learn libraries.

## About
 * This project is part of IITM Advanced Programming Professional in Master Data Science Course,GUVI, IIT Madras Research Park



