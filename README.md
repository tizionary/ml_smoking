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

### Results and Insights
- **Accuracy Metrics**: Detailed performance metrics were obtained for both the full and pruned Decision Tree models.
- **Feature Importance**: Visualizations of decision paths and feature importance were used to interpret the model's behavior and the factors most strongly associated with smoking.
- **Confusion Matrix**: A confusion matrix heatmap was generated to illustrate the model's performance in predicting smokers and non-smokers.

### Conclusion
The project successfully demonstrates how data cleaning, preprocessing, and machine learning can be applied to classify individuals based on their smoking habits using health metrics. The models built provide a reasonable prediction accuracy, with visualizations helping to interpret results.

### Dependencies
To run this project, the following Python packages are required:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or further information, please contact me at [tiz.berlanda@gmail.com].
