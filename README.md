# ml_smoking
## Smoking Data Analysis and Classification

### Project Overview
This project focuses on an exploratory and predictive analysis of a dataset related to smoking habits and their correlation with various health indicators. It aims to identify patterns and predict whether an individual is a smoker based on specific physical and health-related features using machine learning models.

The dataset, which contains several health parameters for individuals, is preprocessed and analyzed to gain insights before building classification models. The project involves data cleaning, data visualization, feature scaling, and machine learning to predict smoking status.

### Data Exploration (EDA)
The project starts with an Exploratory Data Analysis (EDA) to understand the structure of the data and the relationships between features. Here are the key steps:

- **Data Cleaning**: Unnecessary columns (`ID` and `oral`) were removed, and categorical variables were encoded (`Gender` and `tartar` converted to numeric values).
- **Handling Missing Data**: Checked for and addressed any missing values.
- **Outlier Removal**: Anomalies were identified and filtered out based on domain knowledge (e.g., filtering extremely high values for ALT and LDL levels).
- **Duplicate Removal**: Duplicate entries were removed, reducing the dataset's size significantly.
- **Data Visualization**: Various visualizations were created, such as box plots, histograms, scatter plots, and violin plots to explore the distributions and relationships between features like age, gender, BMI, and smoking status.

### Feature Engineering and Preprocessing
Before building the machine learning models, several preprocessing steps were conducted:

- **Feature Scaling**: Applied `StandardScaler` to normalize the dataset's numerical features.
- **Feature Selection**: Certain non-contributive features, such as vision and hearing test results, were dropped to streamline the dataset.
- **New Feature Creation**: BMI was calculated using height and weight data, and an age group category was added to enhance the analysis.

### Data Visualization
The cleaned and preprocessed data was visualized to uncover potential patterns:

- **Correlation Heatmap**: A heatmap was generated to identify correlations between variables.
- **Box Plots**: Provided a visual summary of the distribution and spread of different health features.
- **Violin and KDE Plots**: Illustrated the distribution of metrics like BMI and blood pressure, segmented by smoking status.
- **Pair Plot**: Showed the relationships between numerical variables and how they vary based on smoking status.

### Classification Algorithms
The primary goal of the project was to build a classification model that predicts smoking status. Two different Decision Tree models were employed:

1. **Decision Tree Classifier**:
   - Created using `sklearn`'s `DecisionTreeClassifier`.
   - Optimized parameters like `min_samples_leaf` and `min_samples_split` to improve accuracy.
   - Achieved accuracy on both training and test datasets, with performance evaluated using a confusion matrix.

2. **Pruned Decision Tree Classifier**:
   - A simpler Decision Tree model was built with a maximum depth of 3 to visualize decision-making paths.
   - The pruned tree model was evaluated for training and testing accuracy, highlighting the trade-offs between model complexity and generalization.

3. **K-Nearest Neighbors (KNN) Classifier**:
   - Trained a KNeighborsClassifier to predict smoking status.
   - Evaluated using a classification report and confusion matrix.

4. **Naive Bayes Classifier**:
   - Applied the Bernoulli Naive Bayes model (BernoulliNB).
   - Measured performance with accuracy scores and classification reports.

### Ensemble Learning
To enhance classification performance, the project incorporated ensemble learning techniques:

#### Stacking Classifier:
- Combined `RandomForestClassifier` and `GradientBoostingClassifier` as base classifiers.
- Used a `LogisticRegression` model as the meta-classifier (with an alternative option for SVM).
- Evaluated individual classifier accuracies and overall ensemble performance.
- Confusion matrix visualizations were created for deeper insights.

#### Bagging Classifier:
- Implemented a `BaggingClassifier` with a base `DecisionTreeClassifier`.
- Tuned hyperparameters for optimal performance, including the number of estimators and max samples.
- Visualized results using confusion matrices.

### Hyperparameter Tuning and Advanced Techniques

#### XGBoost Hyperparameter Tuning:
- Utilized `GridSearchCV` to optimize parameters like `max_depth`, `n_estimators`, and `learning_rate`.
- Achieved improved accuracy with the best parameters.
- Applied cross-validation to evaluate the generalization of the tuned model.

#### Handling Imbalanced Data:
- Used SMOTE (Synthetic Minority Oversampling Technique) to balance the class distribution in the training data.
- Improved model performance by addressing class imbalance.

### Neural Network Implementation
A feedforward neural network was implemented to classify smoking status:

#### Architecture:
- Input layer matched the number of features in the dataset.
- Two hidden layers with ReLU activation.
- Output layer with sigmoid activation for binary classification.

#### Training and Evaluation:
- Trained using the `Adam` optimizer and binary cross-entropy loss.
- Evaluated performance with metrics such as accuracy and confusion matrices.
- Compared neural network results with traditional machine learning models to highlight differences in predictive power.

### Conclusion
This project demonstrates the application of advanced machine learning and deep learning techniques to classify smoking habits based on health metrics. Ensemble models and neural networks were implemented to explore their predictive potential, each offering unique advantages in performance and interpretability.

### Dependencies
To run this project, the following Python packages are required:

```python
pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  
xgboost  
tensorflow  
imblearn  
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or further information, please contact me at [tiz.berlanda@gmail.com].
