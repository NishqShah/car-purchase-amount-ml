# Car Purchase Amount Prediction

This machine learning project aims to predict how much a customer is likely to spend on purchasing a car based on their demographics and personal information. Multiple regression models were trained and evaluated to identify the best-performing model for accurate predictions.

## Dataset

The dataset contains customer information such as:
- Customer Name
- Customer e-mail
- Country
- Gender
- Age
- Annual Salary
- Credit Card Debt
- Net Worth
- Car Purchase Amount (Target Variable)

## Key Tasks Performed

- Data Loading and Cleaning
- Exploratory Data Analysis (EDA)
- Feature Selection and Target Separation
- Data Scaling with `MinMaxScaler`
- Model Training using 7 Regression Algorithms
- Model Training using Artificial Neural Network
- Model Comparison using R² score
- Final Prediction and Visualization
- Model Saving using `joblib`

## Exploratory Data Analysis

EDA was performed to understand the data distribution and relationships between features. Some visualizations included:
- **Data** (Pairplot)
- **Net Worth vs Car Purchase Amount** (Scatter plot)
- **Correlation Heatmap**

These helped identify useful patterns and confirmed the relevance of input features.

## Regression Models Used

Eight regression models were trained and evaluated:
1. **Linear Regression**
2. **Support Vector Regressor**
3. **Decision Tree Regressor**
4. **KNeighbours Regressor**
5. **Random Forest Regressor**
6. **Gradient Boosting Regressor**
7. **XGBoost Regressor**
8. **Artificial Neural Network** *(Best)*

## 🧠 Artificial Neural Network (ANN)

The ANN was implemented using TensorFlow/Keras. It significantly outperformed traditional ML models.

### Architecture:
- **Input Layer**: 5 neurons (for 5 features)
- **Hidden Layer 1**: 25 neurons, ReLU
- **Hidden Layer 2**: 25 neurons, ReLU
- **Output Layer**: 1 neuron (linear output)

### Hyperparameters:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Epochs**: 100
- **Batch Size**: 50

### ANN Performance:
- **R² Score**: ~0.992 on test data

## Model Comparison

The models were compared using R² score:

| Model                     | R² Score |
|---------------------------|----------|
| Linear Regression         | 0.98     |
| Support Vector Regressor  | 0.95     |
| Decision Tree Regressor   | 0.87     |
| KNeighbours Regressor     | 0.87     |
| Random Forest Regressor   | 0.95     |
| Gradient Boosting         | 0.97     |
| XGBoost Regressor         | 0.95     |
| Artificial Neural Net     | 0.99     | *(Best)*

 The **Artificial Neural Net** gave the highest R² score and was selected as the final model and saved for deployment using `joblib`.

## Final Visualization

To visualize how well the model predicted values:

- **Models R2_Score** using bar plot.

## Model Saving

The best-performing model (`ANN`) and the preprocessing pipeline were exported using `joblib`. 
These can be reused later for making predictions on unseen data.:
- `car_model.pkl`

These files can be reused for prediction on new unseen data.

## Project Highlights

- End-to-end regression modeling pipeline
- Clean preprocessing using `StandardScaler`
- Visual performance comparison of 8 models
- ANN outperformed traditional ML models
- Easy to extend or deploy


## Files in This Repository

- `car_purchase_amount.ipynb` — Complete notebook including EDA, model training, and saving
- `car_model.pkl` — Trained ANN model
- `Car_Purchasing_Data` - Dataset
