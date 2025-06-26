import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

# --- 1. DATA LOADING AND INITIAL CLEANUP ---

def load_data(filepath):
    """Loads the training data."""
    return pd.read_csv(filepath)

def remove_outliers(df):
    """Removes specific outliers identified during EDA."""
    outliers_to_drop = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
    df = df.drop(outliers_to_drop)
    return df

# --- 2. PREPROCESSING AND FEATURE ENGINEERING ---

def create_preprocessor(df):
    """Creates a preprocessing pipeline for the data."""
    
    # Identify numerical and categorical features FROM THE DATAFRAME IT RECEIVES
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # *** FIX IS HERE: We no longer need to remove columns manually ***
    # The dataframe passed to this function (X) will already have Id and SalePrice removed.
    
    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def handle_missing_values(df):
    """Implements the missing value strategy from the notebook."""
    # Features where NA means "None"
    for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
                'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']:
        df[col] = df[col].fillna('None')
        
    # Features where NA means 0
    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
                'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']:
        df[col] = df[col].fillna(0)
    
    # Impute LotFrontage with neighborhood median
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # For the few remaining, use the mode
    for col in ['MSZoning', 'Utilities', 'Functional', 'Electrical', 'KitchenQual', 'SaleType', 'Exterior1st', 'Exterior2nd']:
         df[col] = df[col].fillna(df[col].mode()[0])
         
    return df

# --- 3. MODEL TRAINING AND SAVING ---

def main():
    """Main function to run the training pipeline."""
    print("Starting the training pipeline...")
    
    # Load data from the 'notebook' subdirectory
    df = load_data('notebook/train.csv')
    
    # Initial Cleanup
    df = remove_outliers(df)
    
    # Separate features and target
    y = np.log1p(df['SalePrice'])
    
    # *** FIX IS HERE: Drop both SalePrice and the non-predictive Id column ***
    X = df.drop(columns=['SalePrice', 'Id'])
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Create the full pipeline: Preprocessor + Model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor(X)),
        ('regressor', Ridge(alpha=10))
    ])
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_log = model_pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_log))
    print(f"Validation RMSE (on log-transformed SalePrice): {rmse:.4f}")
    
    # Save the final trained pipeline
    print("Saving the pipeline to 'house_price_model.pkl'...")
    joblib.dump(model_pipeline, 'house_price_model.pkl')
    
    print("Pipeline training complete and model saved!")


if __name__ == "__main__":
    main()