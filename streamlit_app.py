import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os # Import os for file path checks

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder # Keep for now, but will advise against for high-cardinality nominal features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics # Import specific metrics module

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Life Expectancy 🫀",
    layout="centered",
    page_icon="🫀",
)

st.sidebar.title("Life Expectancy Dashboard 🫀")
page = st.sidebar.selectbox("Select Page", ["Introduction 📘", "Visualization 📊", "Prediction"])

st.image('life.jpg') # Ensure 'life.jpg' is in your app's directory
st.write("   ")
st.write("   ")
st.write("   ")

# --- Centralized Data Loading and Initial Preprocessing (with caching) ---
@st.cache_data
def load_and_preprocess_data(file_path="Life Expectancy Data.csv"):
    if not os.path.exists(file_path):
        st.error(f"Error: Data file '{file_path}' not found.")
        st.stop() # Stop the app if data is not found
    
    df_raw = pd.read_csv(file_path)
    # Apply consistent column cleaning once
    df_raw.columns = df_raw.columns.str.strip()
    df_raw.columns = df_raw.columns.str.replace(' ', '_')
    return df_raw

# Initialize session state for the DataFrame if not already present
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = load_and_preprocess_data()

# Always work with the DataFrame from session state for current operations
# Make a copy if modifications are temporary for the current run, otherwise modify session_state directly.
df = st.session_state.df_processed.copy()
def clean_missing(df: pd.DataFrame, numeric_strategy="median"):
    """
    Return a copy with missing values handled.
    numeric_strategy = "median" | "mean" | "knn"
    """
    df_clean = df.copy()

    # Ensure column names are clean before proceeding
    df_clean.columns = df_clean.columns.str.strip()
    df_clean.columns = df_clean.columns.str.replace(' ', '_')

    num_cols = df_clean.select_dtypes(include="number").columns
    cat_cols = df_clean.select_dtypes(exclude="number").columns

    if numeric_strategy == "knn":
        # KNNImputer needs to be fitted on all numerical columns
        # It's better to isolate the columns before imputation if only specific ones need KNN
        # For simplicity, assuming all numerical columns are suitable for KNN if chosen.
        knn = KNNImputer(n_neighbors=5)
        # Apply KNN imputation only to columns that contain NaNs and are numeric
        cols_to_impute_knn = df_clean[num_cols].columns[df_clean[num_cols].isnull().any()]
        if not cols_to_impute_knn.empty:
            df_clean[cols_to_impute_knn] = knn.fit_transform(df_clean[cols_to_impute_knn])
    else:
        imp = SimpleImputer(strategy=numeric_strategy)
        cols_to_impute_simple = df_clean[num_cols].columns[df_clean[num_cols].isnull().any()]
        if not cols_to_impute_simple.empty:
            df_clean[cols_to_impute_simple] = imp.fit_transform(df_clean[cols_to_impute_simple])

    if len(cat_cols) > 0: # Check if there are any categorical columns
        cat_imp = SimpleImputer(strategy="most_frequent")
        # Apply categorical imputation only to columns that contain NaNs and are categorical
        cols_to_impute_cat = df_clean[cat_cols].columns[df_clean[cat_cols].isnull().any()]
        if not cols_to_impute_cat.empty:
            df_clean[cols_to_impute_cat] = cat_imp.fit_transform(df_clean[cols_to_impute_cat])

    return df_clean
if page == "Introduction 📘":
    st.subheader("01 Introduction 📘")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows)) # 'df' here is a copy of st.session_state.df_processed

    st.markdown("##### 📝 DataFrame Info")
    with st.expander("Show data info output ⇣"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.code(info_str)

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ Your raw data have missing values")

    st.markdown("##### 🧹 Handle Missing Values")
    methods = {
        "Drop rows with any NA": "drop",
        "Fill numeric with median": "median",
        "Fill numeric with mean": "mean",
        "KNN imputation (k=5)": "knn",
    }
    choice = st.selectbox("Choose a strategy", list(methods.keys()), key="na_strategy")

    if st.button("Apply strategy"):
        if methods[choice] == "drop":
            st.session_state.df_processed = st.session_state.df_processed.dropna().reset_index(drop=True)
        else:
            st.session_state.df_processed = clean_missing(st.session_state.df_processed, numeric_strategy=methods[choice])

        st.success("Missing-value handling applied 🎉")
        st.write("Remaining NA counts:")
        st.write(st.session_state.df_processed.isnull().sum())

        csv = st.session_state.df_processed.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download cleaned CSV",
            csv,
            file_name="life_expectancy_clean.csv",
            mime="text/csv",
        )

    st.markdown("##### 📈 Summary Statistics")
    # Show describe table by default, possibly in an expander
    with st.expander("Show Summary Statistics ⇣"):
        st.dataframe(df.describe())
elif page == "Visualization 📊":
    st.subheader("02 Data Visualization 📊 - Insights & Storytelling")

    # Use the processed DataFrame from session state
    df_viz = st.session_state.df_processed.copy()

    # Filter out non-numeric columns from options if plotting numeric charts
    numeric_cols = df_viz.select_dtypes(include=np.number).columns.tolist()
    all_cols = df_viz.columns.tolist()

    col_x = st.selectbox("Select X-axis variable", all_cols, index=0)
    col_y = st.selectbox("Select Y-axis variable", numeric_cols, index=1 if "Life_expectancy" in numeric_cols else 0)

    tab1, tab2, tab3 = st.tabs(["Bar Chart 📊", "Line Chart 📈", "Correlation Heatmap 🔥"])

    with tab1:
        st.subheader(f"Bar Chart: {col_y} by {col_x}")
        
        # Original Bar Chart Code
        df_bar = df_viz[[col_x, col_y]].copy()
        df_bar[col_y] = pd.to_numeric(df_bar[col_y], errors='coerce')
        df_bar = df_bar.dropna(subset=[col_x, col_y])

        if col_x in numeric_cols:
            st.warning("For numerical X-axis in bar charts, consider aggregating data (e.g., mean per category). Displaying as is might not be optimal.")
            df_bar_sorted = df_bar.sort_values(by=col_x).set_index(col_x)
            st.bar_chart(df_bar_sorted[col_y], use_container_width=True)
        else:
            df_grouped = df_bar.groupby(col_x)[col_y].mean().reset_index()
            df_grouped_sorted = df_grouped.sort_values(by=col_x).set_index(col_x)
            st.bar_chart(df_grouped_sorted[col_y], use_container_width=True)

        # --- ADD INSIGHTS FOR BAR CHART HERE ---
        st.markdown("---") # Separator
        st.subheader("💡 Bar Chart Insights")
        st.write(f"This bar chart illustrates the average **{col_y.replace('_', ' ')}** across different categories of **{col_x.replace('_', ' ')}**.")
        st.markdown("""
        **Key Observations:**
        * Here, we clearly see a significant disparity: the average Life Expectancy in 'Developed' nations is approximately 80.12 years, while in 'Developing' nations, it's notably lower, at about 66.85 years. This represents a substantial difference of approximately 13.27 years, highlighting the profound global health inequalities linked to a country's development status. This suggests that economic development, robust healthcare infrastructure, and broader access to resources play a critical role in extending lifespan."
         """)

    with tab2:
        st.subheader(f"Line Chart: {col_y} over {col_x}")

        # Original Line Chart Code
        df_line = df_viz[[col_x, col_y]].copy()
        df_line[col_y] = pd.to_numeric(df_line[col_y], errors='coerce')
        df_line = df_line.dropna(subset=[col_x, col_y])

        if col_x in numeric_cols or df_line[col_x].nunique() <= 20:
            df_line_sorted = df_line.sort_values(by=col_x).set_index(col_x)
            st.line_chart(df_line_sorted[col_y], use_container_width=True)
        else:
            st.warning("Line chart might not be suitable for the selected X-axis variable. Consider another chart type or aggregate the data.")
            
        # --- ADD INSIGHTS FOR LINE CHART HERE ---
        st.markdown("---") # Separator
        st.subheader("💡 Line Chart Insights")
        st.write(f"This line chart illustrates the trend of **{col_y.replace('_', ' ')}** over **{col_x.replace('_', ' ')}**.")
        st.markdown("""
        **Key Observations:**
        * We observe a strong and consistent increasing trend in average Life Expectancy across the years, from approximately 66.97 years in 2000 to 70.93 years in 2015. This represents a significant increase of about 3.96 years over this 16-year period. This positive trend underscores significant global advancements in public health, medical care, and socio-economic conditions that collectively contribute to longer and healthier lives worldwide.""")

    with tab3:
        st.subheader("Correlation Matrix")
        df_numeric = df_viz.select_dtypes(include=np.number)

        if not df_numeric.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(18, 14))
            sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
            plt.title("Correlation Matrix of Numerical Variables")
            st.pyplot(fig_corr)
        else:
            st.warning("No numerical columns available for correlation matrix.")

        # --- ADD INSIGHTS FOR CORRELATION HEATMAP HERE ---
        st.markdown("---") # Separator
        st.subheader("💡 Correlation Heatmap Insights")
        st.write("This heatmap displays the correlation coefficients between all numerical variables in the dataset. Values range from -1 (strong negative correlation) to 1 (strong positive correlation), with 0 indicating no linear correlation.")
        st.markdown(f"""
        **Key Observations Regarding '{'Life_expectancy'.replace('_', ' ')}' (Your Target Variable):**
        * We identified strong positive correlations with Life Expectancy for features like Schooling (r = 0.72) and Income composition of resources (r = 0.70). This strongly suggests that higher educational attainment and better access to resources are critically linked to longer, healthier lives.

        * Conversely, Adult Mortality (r = -0.69) and HIV/AIDS (r = -0.55) show profound negative correlations. This is intuitive, as higher prevalence of these issues directly diminishes average life expectancy within a population."

        * We also observe high correlations between some independent variables, for example, GDP and Percentage Expenditure (r = 0.89). This makes sense as wealthier nations often invest more in health, but it's an important point to consider for feature independence in our predictive model. """)
       
elif page == "Prediction":
    st.subheader("Prediction with Linear Regression")
    st.title("Life Expectancy Prediction")

    # Use the processed DataFrame from session state
    df_pred = st.session_state.df_processed.copy()

    # --- Preprocessing for Prediction Page ---
    # Ensure all numerical columns are filled (in case user skipped 'Apply Strategy' in Intro)
    for col in df_pred.select_dtypes(include=np.number).columns:
        if df_pred[col].isnull().any():
            median_val = df_pred[col].median()
            df_pred[col].fillna(median_val, inplace=True)

    # Handle 'Status' column: One-hot encode it if it's still an object type.
    # If the `clean_missing` function or prior steps have already handled it, this might be redundant.
    # Assuming `Status` (or 'Status_Developing') is ready.
    if 'Status' in df_pred.columns and df_pred['Status'].dtype == 'object':
        df_pred = pd.get_dummies(df_pred, columns=['Status'], drop_first=True, dtype=int)
        
    # Drop 'Country' column as it has high cardinality and is not suitable for simple Linear Regression
    if 'Country' in df_pred.columns:
        df_pred = df_pred.drop(columns=['Country'])

    # Verify no missing values remain before model training
    if df_pred.isnull().sum().sum() > 0:
        st.warning("Warning: Missing values still exist after preprocessing for prediction. This might affect model quality.")
        st.write(df_pred.isnull().sum())
        # Consider a more robust error handling or default imputation here if this happens often.

    # Define target column name (consistent with cleaned data)
    target_col_name = "Life_expectancy"
    if target_col_name not in df_pred.columns:
        st.error(f"Error: Target column '{target_col_name}' not found in the processed data. Please ensure column names are consistent.")
        st.stop()

    # Sidebar: select features, target, and metrics
    all_cols_for_prediction = df_pred.columns.tolist()
    
    # Remove the target column from default features if it exists
    default_features_for_selection = [c for c in all_cols_for_prediction if c != target_col_name]
    
    # Filter out potential problematic columns like 'Country' if it somehow still exists
    # Although it should be dropped above, this adds a layer of safety.
    default_features_for_selection = [
        f for f in default_features_for_selection if f not in ['Country']
    ]

    features_selection = st.sidebar.multiselect(
        "Select Features (X)", all_cols_for_prediction, default=default_features_for_selection
    )
    
    # Default target variable selection
    default_target_idx = all_cols_for_prediction.index(target_col_name) if target_col_name in all_cols_for_prediction else 0
    target_selection = st.sidebar.selectbox(
        "Select Target Variable (Y)", all_cols_for_prediction, index=default_target_idx
    )
    
    metrics_options = ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"]
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Display", metrics_options, default=["Mean Absolute Error (MAE)"]
    )

    # Construct X and y
    # Ensure selected features exist and handle case where no features are selected
    valid_features = [f for f in features_selection if f in df_pred.columns]
    if not valid_features:
        st.warning("Please select at least one feature for the model.")
        st.stop() # Stop execution if no features are selected

    X = df_pred[valid_features]
    y = df_pred[target_selection]

    st.subheader("Features (first 5 rows)")
    st.dataframe(X.head())
    st.subheader("Target (first 5 rows)")
    st.dataframe(y.head())

    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate and display metrics
    st.subheader("Model Evaluation Metrics")
    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, y_pred)
        st.write(f"- **Mean Squared Error (MSE)**: `{mse:.3f}`")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, y_pred)
        st.write(f"- **Mean Absolute Error (MAE)**: `{mae:.3f}`")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, y_pred)
        st.write(f"- **R² Score**: `{r2:.3f}`")

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size for better readability
    ax.scatter(y_test, y_pred, alpha=0.6) # Use alpha for transparency
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--r", linewidth=2, label="Perfect Prediction Line"
    )
    ax.set_xlabel("Actual Life Expectancy")
    ax.set_ylabel("Predicted Life Expectancy")
    ax.set_title("Actual vs Predicted Life Expectancy")
    ax.legend()
    st.pyplot(fig)