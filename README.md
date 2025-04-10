# Forest Fire Prediction

## Overview
This project aims to predict the Fire Weather Index (FWI) based on meteorological and environmental conditions. The model is trained using the **Algerian Forest Fires Dataset**, which contains various weather attributes and fire-related indices.

## Dataset Description
The dataset consists of **243 records** with **15 features**. It includes meteorological data along with fire weather index components.

### **Features**
| Column Name   | Description |
|--------------|------------|
| **day**       | Day of the month when the data was recorded |
| **month**     | Month when the data was recorded |
| **year**      | Year when the data was recorded |
| **Temperature** | Temperature in degrees Celsius |
| **RH**        | Relative Humidity (%) |
| **Ws**        | Wind speed (km/h) |
| **Rain**      | Amount of rainfall (mm) |
| **FFMC**      | Fine Fuel Moisture Code (fire danger index component) |
| **DMC**       | Duff Moisture Code (fire danger index component) |
| **DC**        | Drought Code (fire danger index component) |
| **ISI**       | Initial Spread Index (fire danger index component) |
| **BUI**       | Buildup Index (fire danger index component) |
| **FWI**       | Fire Weather Index - the final output indicating fire risk |
| **Classes**   | Fire occurrence classification (fire or not fire) |
| **Region**    | Region identifier (0 or 1, indicating different areas) |

## Project Structure
```
Forestfire-main/
│── models/                # Trained ML models and scalers
│── notebooks/             # Jupyter Notebooks for data analysis
│── templates/             # HTML templates for web app
│── application.py         # Main Flask application
│── requirements.txt       # Required dependencies
│── README.md              # Project documentation
```

## Dataset
- Source: `Algerian_forest_fires_dataset.csv`
- Features: Various weather and environmental factors.
- Target Variable: `FWI` (Fire Weather Index)
- Categorical Encoding: `Classes` column was converted to binary (fire = 1, no fire = 0).

## Preprocessing Steps
1. **Dropped Unnecessary Columns**: `day`, `month`, `year`
2. **Feature Scaling**: Standardized input variables.
3. **Train-Test Split**: 75% training, 25% testing.
4. **Checked for Multicollinearity**: Heatmap analysis.

## Algorithms Used
The notebook implements the following models:
1. **Linear Regression**
   - Used to establish a baseline prediction model.
   - Evaluated using **Mean Absolute Error (MAE)** and **R² Score**.
2. **Lasso Regression**
   - A regularized version of Linear Regression to reduce overfitting.
   - Model selection via **LassoCV** (cross-validation tuning).

## Model Evaluation
The models were evaluated using:
- **Mean Absolute Error (MAE)**: Measures prediction accuracy.
- **R² Score**: Determines how well the model explains variance in the target variable.

## Results
- The trained models provided reasonable predictive performance, with **Linear Regression** and **Lasso Regression** producing different error metrics.
- Further improvements could involve feature engineering, additional regularization techniques, or ensemble models.

## How to Run the Model
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Model Training.ipynb"
   ```

## Future Work
- Explore advanced models like **Random Forest** or **Gradient Boosting**.
- Perform hyperparameter tuning for improved accuracy.
- Integrate real-time fire risk assessment using weather APIs.

## License
This project is open-source and can be used for research and educational purposes.

