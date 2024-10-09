# Bank Marketing Prediction Project

## Project Overview
This project aims to predict whether clients will subscribe to a term deposit based on various attributes from a marketing campaign conducted by a Portuguese banking institution. The dataset used for this project is sourced from the UCI Machine Learning Repository.

## Dataset
The dataset contains client data such as age, job, marital status, education, balance, and details of the marketing campaign, including contact communication type and last contact duration. The target variable indicates whether the client subscribed to a term deposit (binary classification: "yes" or "no").

## Objectives
- Preprocess the data by handling categorical variables and scaling numerical features.
- Implement traditional machine learning models including:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- Build and evaluate an Artificial Neural Network (ANN) to compare performance against traditional models.

## Methodology
1. **Data Preprocessing**: 
   - The dataset is split into features and target variable. 
   - Categorical variables are transformed using one-hot encoding, and numerical features are standardized.

2. **Model Training and Evaluation**:
   - Traditional machine learning models are trained and evaluated on the preprocessed data.
   - Performance metrics such as accuracy, precision, recall, and F1-score are used to assess model performance.

3. **Deep Learning**:
   - An ANN is constructed with an input layer, two hidden layers, and an output layer using the Keras library.
   - The model is trained on the training dataset and evaluated on the test dataset, providing insights into its predictive capabilities.

## Results
- Each model's performance is documented, showcasing accuracy and detailed classification reports.
- The ANN's performance is evaluated with a confusion matrix and classification report, highlighting its effectiveness in predicting client subscriptions.

## Conclusion
This project demonstrates the practical application of both traditional machine learning and deep learning techniques to solve a binary classification problem in the banking sector. The comparison of models allows for insights into the strengths and weaknesses of each approach, showcasing the importance of selecting appropriate methods based on the problem at hand.

## Future Work
Potential future improvements include hyperparameter tuning for each model, exploring ensemble methods, and analyzing feature importance to enhance prediction accuracy.
