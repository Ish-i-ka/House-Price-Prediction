import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer # Import SimpleImputer
import joblib

# --- Functions for loading and cleaning ---

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers(df):
    outliers_to_drop = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
    df = df.drop(outliers_to_drop)
    return df

# --- NEW AND IMPROVED PREPROCESSOR ---
# This function now handles everything: imputation, scaling, and encoding.

def create_preprocessor():
    """
    Creates a full preprocessing pipeline that handles missing values,
    scales numerical features, and one-hot encodes categorical features.
    """
    # Define transformers for numerical features
    # Pipeline for numerical data: 1. Impute with median, 2. Scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define transformers for categorical features
    # Pipeline for categorical data: 1. Impute with 'None' (most frequent), 2. One-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- Define which columns get which treatment ---
    # These lists are based on our EDA.
    
    numerical_features = [
        'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
    ]
    
    categorical_features = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
        'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]

    # Create the master preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any columns not specified (like 'Id')
    )
    
    return preprocessor

# --- MAIN FUNCTION ---

def main():
    """Main function to run the training pipeline with hyperparameter tuning."""
    print("Starting the training pipeline...")
    
    # Load and clean data
    df = load_data('notebook/train.csv')
    df = remove_outliers(df)
    y = np.log1p(df['SalePrice'])
    X = df.drop(columns=['SalePrice']) # We no longer drop 'Id' here, the preprocessor will.

    # We NO LONGER need the separate handle_missing_values(X) call.
    
    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor()), # The new preprocessor handles everything
        ('regressor', Ridge())
    ])
    
    # Define the parameter grid
    param_grid = {
        'regressor__alpha': [5, 8, 10, 12, 15, 20, 25] 
    }
    
    print("Setting up GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model_pipeline, 
        param_grid=param_grid, 
        cv=5, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1,
        verbose=1
    )
    
    print("Running grid search...")
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    print("\n--- Grid Search Complete ---")
    print(f"Best Alpha: {grid_search.best_params_['regressor__alpha']}")
    print(f"Best Cross-Validated RMSE (log scale): {-grid_search.best_score_:.4f}")
    
    print("\nSaving the BEST pipeline to 'house_price_model.pkl'...")
    joblib.dump(best_model, 'house_price_model.pkl')
    
    print("Pipeline training complete and optimized model saved!")

if __name__ == "__main__":
    main()