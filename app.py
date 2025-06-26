import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. LOAD THE SAVED MODEL AND DATA ---

@st.cache_resource
def load_model_and_data():
    """Loads the model, original training data, and mappings."""
    pipeline = joblib.load('house_price_model.pkl')
    train_df_raw = pd.read_csv('notebook/train.csv')
    return pipeline, train_df_raw

model_pipeline, train_df_raw = load_model_and_data()

# --- Dictionaries for Human-Readable Text ---

# Mapping for Exterior Quality
EXTER_QUAL_MAP = {
    'Ex': 'Excellent',
    'Gd': 'Good',
    'TA': 'Average/Typical',
    'Fa': 'Fair',
    'Po': 'Poor'
}
EXTER_QUAL_MAP_REVERSE = {v: k for k, v in EXTER_QUAL_MAP.items()}

# Mapping for Neighborhood Tiers
NEIGHBORHOOD_TIER_MAP = {
    'Affluent': ['NridgHt', 'NoRidge', 'StoneBr'],
    'Mid-Range': ['CollgCr', 'Veenker', 'Crawfor', 'Somerst', 'NWAmes', 'Gilbert', 'Blmngtn', 'Timber'],
    'Affordable': ['Sawyer', 'NAmes', 'Mitchel', 'BrkSide', 'OldTown', 'Edwards', 'SawyerW', 'IDOTRR', 'BrDale', 'SWISU', 'MeadowV', 'NPkVill', 'Blueste']
}


# --- 2. DEFINE THE UI OF THE APP ---

st.set_page_config(layout="wide")
st.title("🏡 Advanced House Price Prediction App")
st.write(
    "This app predicts house prices in Ames, Iowa. "
    "Use the sidebar to input house features and see how they affect the price."
)

st.sidebar.header("House Features")

# --- 3. CREATE INPUT WIDGETS ---

st.sidebar.subheader("Key Attributes")
overall_qual = st.sidebar.slider('Overall Quality', 1, 10, 5)
year_built = st.sidebar.slider('Year Built', 1870, 2024, 2005)
exter_qual_friendly = st.sidebar.selectbox(
    'Exterior Quality',
    options=list(EXTER_QUAL_MAP.values())
)
exter_qual_code = EXTER_QUAL_MAP_REVERSE[exter_qual_friendly]

st.sidebar.subheader("Size and Space")
gr_liv_area = st.sidebar.slider('Above Ground Living Area (sq ft)', 500, 5000, 1500)
total_bsmt_sf = st.sidebar.slider('Total Basement Area (sq ft)', 0, 6000, 1000)
garage_cars = st.sidebar.slider('Garage Capacity (Cars)', 0, 4, 2)
full_bath = st.sidebar.slider('Full Bathrooms', 0, 4, 2)

st.sidebar.subheader("Location")
neighborhood_tier = st.sidebar.selectbox(
    'Select Neighborhood Tier',
    options=list(NEIGHBORHOOD_TIER_MAP.keys())
)
neighborhood_code = NEIGHBORHOOD_TIER_MAP[neighborhood_tier][0]
st.sidebar.info(f"Using '{neighborhood_code}' as a representative neighborhood for the '{neighborhood_tier}' tier.")


# --- 4. PREPARE THE INPUT FOR THE MODEL ---

input_data = {
    'OverallQual': overall_qual, 'GrLivArea': gr_liv_area, 'GarageCars': garage_cars,
    'TotalBsmtSF': total_bsmt_sf, 'YearBuilt': year_built, 'FullBath': full_bath,
    'Neighborhood': neighborhood_code, 'ExterQual': exter_qual_code,

    'MSSubClass': 60, 'MSZoning': 'RL', 'LotFrontage': 65.0, 'LotArea': 8450,
    'Street': 'Pave', 'Alley': 'None', 'LotShape': 'Reg', 'LandContour': 'Lvl',
    'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl',
    'Condition1': 'Norm', 'Condition2': 'Norm', 'BldgType': '1Fam',
    'HouseStyle': '2Story', 'OverallCond': 5, 'YearRemodAdd': 2005,
    'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd', 'MasVnrType': 'None', 'MasVnrArea': 0.0,
    'ExterCond': 'TA', 'Foundation': 'PConc', 'BsmtQual': 'Gd',
    'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'GLQ', 'BsmtFinSF1': 706,
    'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': 150, 'Heating': 'GasA',
    'HeatingQC': 'Ex', 'CentralAir': 'Y', 'Electrical': 'SBrkr', '1stFlrSF': 1000,
    '2ndFlrSF': 0, 'LowQualFinSF': 0, 'BsmtFullBath': 1, 'BsmtHalfBath': 0,
    'HalfBath': 1, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'KitchenQual': 'Gd',
    'TotRmsAbvGrd': 8, 'Functional': 'Typ', 'Fireplaces': 0, 'FireplaceQu': 'None',
    'GarageType': 'Attchd', 'GarageYrBlt': 2005.0, 'GarageFinish': 'RFn',
    'GarageArea': 550, 'GarageQual': 'TA', 'GarageCond': 'TA', 'PavedDrive': 'Y',
    'WoodDeckSF': 0, 'OpenPorchSF': 61, 'EnclosedPorch': 0, '3SsnPorch': 0,
    'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': 'None', 'Fence': 'None',
    'MiscFeature': 'None', 'MiscVal': 0, 'MoSold': 2, 'YrSold': 2008,
    'SaleType': 'WD', 'SaleCondition': 'Normal'
}

input_df = pd.DataFrame([input_data])


# --- 5. MAKE PREDICTION AND DISPLAY RESULTS ---
main_col, viz_col = st.columns([1, 1])

with main_col:
    st.subheader("Prediction")
    prediction_container = st.container()
    with prediction_container:
        if st.button('Predict Price', type="primary"):
            prediction_log = model_pipeline.predict(input_df)
            prediction_dollars = np.expm1(prediction_log[0])
            st.markdown(f"### Predicted Price: `${prediction_dollars:,.2f}`")

    st.subheader("What's Driving the Price?")
    with st.expander("See Prediction Explanation"):
        # This part remains the same
        try:
            regressor = model_pipeline.named_steps['regressor']
            preprocessor = model_pipeline.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            coefficients = regressor.coef_
            transformed_input = preprocessor.transform(input_df)
            contributions = transformed_input.toarray()[0] * coefficients
            
            contribution_df = pd.DataFrame({
                'Feature': feature_names,
                'Contribution': contributions
            }).sort_values(by='Contribution', key=abs, ascending=False).head(15)
            
            contribution_df['Feature'] = contribution_df['Feature'].str.replace('cat__', '').str.replace('num__', '')
            
            positive_contrib = contribution_df[contribution_df['Contribution'] > 0]
            negative_contrib = contribution_df[contribution_df['Contribution'] < 0]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(positive_contrib['Feature'], positive_contrib['Contribution'], color='green', label='Increases Price')
            ax.barh(negative_contrib['Feature'], negative_contrib['Contribution'], color='red', label='Decreases Price')
            ax.set_xlabel("Impact on Predicted Price (log scale)")
            ax.set_title("Top 15 Features Influencing This Prediction")
            ax.invert_yaxis()
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not generate explanation: {e}")

# --- Interactive Visualization ---
with viz_col:
    st.subheader("How Your House Compares")
    
    # --- FIX IS HERE ---
    
    # 1. Create an explicit dictionary to map dropdown choices to the actual variables
    feature_variable_map = {
        'GrLivArea': gr_liv_area,
        'TotalBsmtSF': total_bsmt_sf,
        'YearBuilt': year_built,
        'OverallQual': overall_qual
    }

    # 2. Let the user select from the dictionary keys
    feature_to_plot = st.selectbox(
        "Choose a feature to visualize",
        options=list(feature_variable_map.keys())
    )
    
    # 3. Get the user's current input value using the dictionary (this is the fix)
    user_value = feature_variable_map[feature_to_plot]

    # The rest of the plotting code is the same
    fig, ax = plt.subplots()
    sns.histplot(train_df_raw[feature_to_plot], kde=True, ax=ax, color='skyblue', label='Distribution of All Houses')
    ax.axvline(user_value, color='red', linestyle='--', linewidth=2, label='Your Input')
    ax.set_title(f'Distribution of {feature_to_plot}')
    ax.set_xlabel(f'{feature_to_plot}')
    ax.legend()
    st.pyplot(fig)