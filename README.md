# Bank-Loan-Prediction
This project focuses on building a deep learning model to predict whether a customer will take a personal loan based on their demographic and financial attributes.

## Project Overview

The goal of this project is to leverage a deep learning model to assist banks in identifying customers who are likely to accept a personal loan offer. The dataset includes various customer attributes such as age, income, education, credit card spending, and more. The project goes through data preprocessing, exploratory data analysis (EDA), model building, and optimization to deliver a reliable predictive model.

## Key Features
- **Data Preprocessing**: Cleaning and preparing the dataset, handling missing values, normalizing numerical features, and converting fractions in the data for accurate analysis.
- **Exploratory Data Analysis (EDA)**: Visualizing the distribution of numerical and categorical variables to gain insights into customer behavior.
- **Model Building**: Creating a baseline deep learning model using TensorFlow and optimizing it using techniques like Batch Normalization and Dropout to improve model performance and prevent overfitting.
- **Model Evaluation**: Achieving high accuracy in predicting personal loan acceptance, with the final model providing reliable results for financial decision-making.

## Dataset
The dataset contains customer details such as:
- **Age**
- **Experience**
- **Income**
- **Family Size**
- **Credit Card Average Spending (CCAvg)**
- **Education**
- **Mortgage**
- **Personal Loan** (Target variable: whether the customer accepted a loan or not)
- **Other attributes** like Securities Account, CD Account, Online Banking, and Credit Card ownership.

## Model Architecture
The deep learning model is built using a Sequential neural network with the following layers:
- Input layer based on the number of features.
- Two hidden layers with ReLU activation.
- An output layer using a sigmoid activation function for binary classification (loan acceptance).
- Optimized using the Adam optimizer, with a lower learning rate and regularization techniques like Batch Normalization and Dropout.

## Process
1. **Data Preprocessing**: The dataset was cleaned, missing values were handled, and numerical variables were normalized to ensure uniformity in the model.
2. **Exploratory Data Analysis**: Visualizations like histograms, box plots, and bar plots were used to understand the data and relationships between variables.
3. **Modeling**: A neural network was developed, and performance was optimized using regularization techniques, ensuring a balance between accuracy and model generalization.
4. **Model Evaluation**: The model was evaluated using validation data, and it achieved high accuracy, demonstrating its reliability in predicting loan acceptance.

## Results
The final model showed excellent predictive capability, with an accuracy rate of over 98%, making it highly effective for predicting whether a customer would accept a personal loan.

## Conclusion
This project successfully developed a robust deep learning model that can help financial institutions predict personal loan acceptance. The model was fine-tuned through various optimizations, and the final result was a high-performing, accurate prediction tool.

## Technologies Used
- **Python**: For data preprocessing, model building, and analysis.
- **TensorFlow/Keras**: For deep learning model development.
- **Pandas, NumPy**: For data manipulation and analysis.
- **Matplotlib, Seaborn**: For data visualization.
