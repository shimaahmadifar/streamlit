import pandas as pd
import glob
import os
import streamlit as st





# Function to process DateTime (for Grafico delle Catture dataset)
def separate_data(df):
    """
    Processes DataFrame by converting 'DateTime' to proper format, 
    and separating it into 'Date' and 'Time' columns.
    """
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time
    df = df.drop(columns=['DateTime'])
    return df

# Function to correct temperature and humidity values (for Dati Meteo Storici dataset)
def correct_temperature_and_humidity(df):
    st.write("🔍 **Detected columns:**", df.columns.tolist())  # Debugging: Show available columns
    
    # Standardize column names (remove spaces)
    df.columns = df.columns.str.strip()

    # Expected column names
    required_columns = ['Media Temperatura', 'Temperatura Intervallo', 'Temperatura Intervallo.1', 'Media Umidità']
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing columns in dataset: {missing_cols}")
        return df  # Return unmodified DataFrame

    # Convert columns to string and apply correction
    for col in required_columns:
        df[col] = df[col].astype(str)

        # Handle cases where values already have a decimal or are invalid
        df[col] = df[col].apply(lambda x: x if '.' in x else x[:2] + '.' + x[2:] if len(x) > 2 else x)

        # Convert back to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

    st.success("✅ Temperature & humidity correction applied successfully!")
    return df

# Define datasets
datasets = {
    "Grafico delle Catture": {
        "Cicalino 1": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/grafico-delle-catture%20(Cicalino%201).csv",
        "Cicalino 2": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/grafico-delle-catture%20(Cicalino%202).csv",
        "Imola 1": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/grafico-delle-catture%20(Imola%201).csv",
        "Imola 2": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/grafico-delle-catture%20(Imola%202).csv",
        "Imola 3": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/grafico-delle-catture%20(Imola%203).csv",
    },
    "Dati Meteo Storici": {
        "Cicalino 1": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/dati-meteo-storici%20(Cicalino%201).csv",
        "Cicalino 2": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/dati-meteo-storici%20(Cicalino%202).csv",
        "Imola 1": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/dati-meteo-storici%20(Imola%201).csv",
        "Imola 2": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/dati-meteo-storici%20(Imola%202).csv",
        "Imola 3": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/dati-meteo-storici%20(Imola%203).csv",
    }
}


# Location mapping
locations = ["Cicalino 1", "Cicalino 2", "Imola 1", "Imola 2", "Imola 3"]
location_mapping = {loc: loc.lower().replace(" ", "") for loc in locations}

# Streamlit App
st.title("📊 Interactive CSV Processor")

# Select dataset category
dataset_category = st.radio("Select dataset category:", list(datasets.keys()))

# Upload multiple CSV files
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Select file to display
    file_options = [file.name for file in uploaded_files]
    selected_file = st.selectbox("Select a file to process:", file_options)

    for file in uploaded_files:
        if file.name == selected_file:
            df = pd.read_csv(file)

            # Apply specific processing based on dataset type
            if dataset_category == "Grafico delle Catture":
                df = separate_data(df)

            elif dataset_category == "Dati Meteo Storici":
                st.write("📌 **Before correction:**")
                st.dataframe(df.head(5))  # Show before correction

                # Checkbox to apply correction
                apply_correction = st.checkbox("Apply Temperature & Humidity Correction")

                if apply_correction:
                    df = correct_temperature_and_humidity(df)
                    st.write("📌 **After correction:**")
                    st.dataframe(df.head(5))  # Show after correction

            # Select location for the file
            selected_location = st.selectbox(f"Select location for {selected_file}", locations)
            df["Location"] = location_mapping[selected_location]

            # Display processed data
            st.subheader(f"Processed Data - {selected_file}")
            num_rows = st.slider("Select number of rows to display:", 5, 50, 10)
            st.dataframe(df.head(num_rows))

            # Search functionality
            search_query = st.text_input(f"Search in DataFrame for {selected_file}:", key=f"search_{selected_file}")
            if search_query:
                df_filtered = df[df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]
                st.dataframe(df_filtered)

            # Download processed data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed CSV", csv, f"processed_{selected_file}", "text/csv")


dft1 = pd.read_csv("https://raw.githubusercontent.com/seyedehsara/streamlit/main/dft1.csv")
dft2 = pd.read_csv("https://raw.githubusercontent.com/seyedehsara/streamlit/main/dft2.csv")
dft3 = pd.read_csv("https://raw.githubusercontent.com/seyedehsara/streamlit/main/dft3.csv")
dft4 = pd.read_csv("https://raw.githubusercontent.com/seyedehsara/streamlit/main/dft4.csv")
dft5 = pd.read_csv("https://raw.githubusercontent.com/seyedehsara/streamlit/main/dft5.csv")
df = pd.read_csv("https://raw.githubusercontent.com/seyedehsara/streamlit/main/df.csv")
df_merged = pd.read_csv("https://raw.githubusercontent.com/seyedehsara/streamlit/main/df_saved.csv")


st.title("Interactive Data Viewer")

# Dropdown to select dataset
dataset_options = {
    "cicalino 1": dft1,
    "cicalino 2": dft2,
    "Imola 1": dft3,
    "Imola 2": dft4,
    "Imola 3": dft5,
    "graffico":df,
    "merged all together":df_merged,
}

selected_dataset = st.selectbox("Select a dataset:", list(dataset_options.keys()))

# Show the selected dataset
st.write(f"Displaying: **{selected_dataset}**")
st.dataframe(dataset_options[selected_dataset])  # Interactive table
if st.checkbox("Show Summary Statistics"):
    st.write(dataset_options[selected_dataset].describe())

#  Show column names
if st.checkbox("Show Column Names"):
    st.write(list(dataset_options[selected_dataset].columns))




# Strip spaces from column names (fixes hidden space issues)
df_merged.columns = df_merged.columns.str.strip()

# Convert 'New Captures (per Event)' to binary classification target
df_merged['New Captures (per Event)'] = df_merged['New Captures (per Event)'].apply(lambda x: 0 if x == 0 else 1)

# Ensure 'Max Temperature' and 'Min Temperature' exist
if 'Max Temperature' in df_merged.columns and 'Min Temperature' in df_merged.columns:
    df_merged['temp_change'] = df_merged['Max Temperature'] - df_merged['Min Temperature']
else:
    print("Error: 'Max Temperature' or 'Min Temperature' column not found.")

# Display first few rows to verify
print(df_merged.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of Target Variable
sns.countplot(x='New Captures (per Event)', data=df_merged)
plt.title('Distribution of New Captures (per Event)')
plt.show()


st.title("📊 Interactive Data Analysis Dashboard")

# Show dataset preview
st.subheader("📋 Dataset Preview")
st.dataframe(df_merged.head(100))

# Checkbox to show statistics
if st.checkbox("📊 Show Summary Statistics", key="summary_stats"):
    st.write(df_merged.describe())

# Checkbox to show column names
if st.checkbox("📝 Show Column Names", key="column_names"):
    st.write(df_merged.columns.tolist())

# Checkbox for first chart: Distribution of New Captures
if st.checkbox("📈 Show Distribution of New Captures", key="new_captures_chart"):
    st.subheader("📊 Distribution of New Captures (per Event)")
    fig, ax = plt.subplots()
    sns.countplot(x='New Captures (per Event)', data=df_merged, ax=ax)
    ax.set_title('Distribution of New Captures (per Event)')
    st.pyplot(fig)









# Show dataset preview
st.subheader("📋 Dataset Preview")
st.dataframe(df_merged.head())

st.header("Distribution of average temperature")
st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/download%20(2).png", use_container_width=True)

st.header("Temperature trends")
st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/photo_2025-02-18_21-16-37.jpg", use_container_width=True)

st.header("Insect capture over time")
st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/photo_2025-02-18_21-16-25.jpg", use_container_width=True)

st.header("Record duration")
st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/photo_2025-02-18_21-16-31.jpg", use_container_width=True)



# Section: Correlation Heatmap
st.header("🔥 Correlation Analysis")

if st.checkbox("📊 Show Correlation Heatmap", key="correlation_heatmap"):
    st.subheader("🔥 Feature Correlation Heatmap")

    # Drop 'Location' and 'Date' if they exist
    if 'Location' in df_merged.columns and 'Date' in df_merged.columns:
        df_numeric = df_merged.drop(columns=['Location', 'Date'])
    else:
        df_numeric = df_merged  # If they don't exist, use original

    # User selects columns for heatmap
    selected_columns = st.multiselect(
        "📝 Select features to include in the heatmap:",
        df_numeric.columns.tolist(),
        default=df_numeric.columns.tolist()
    )
    
    if selected_columns:
        df_selected = df_numeric[selected_columns]
    else:
        st.warning("⚠️ Please select at least one feature!")
        df_selected = df_numeric

    # Correlation Matrix
    correlation_matrix = df_selected.corr()

    # Display feature with the highest correlation (excluding diagonal)
    correlation_matrix_no_diag = correlation_matrix.mask(
        correlation_matrix == 1.0  # Mask diagonal values (which are always 1)
    )
    max_corr_value = correlation_matrix_no_diag.max().max()  # Max value in the matrix
    max_corr_pair = correlation_matrix_no_diag.stack().idxmax()  # Pair of features with max correlation

    # Show the most correlated feature pair
    st.subheader("📊 Most Correlated Feature Pair")
    st.write(f"The pair of features with the highest correlation is **{max_corr_pair[0]}** and **{max_corr_pair[1]}** with a correlation value of **{max_corr_value:.2f}**.")

    # Heatmap Type Selection
    heatmap_type = st.radio(
        "🔍 Select Heatmap Type",
        ["Full Heatmap", "High Correlation Only"],
        index=0
    )

    # Filter for high correlations if selected
    if heatmap_type == "High Correlation Only":
        threshold = st.slider("📊 Set correlation threshold:", 0.0, 1.0, 0.6, 0.05)
        mask = (correlation_matrix.abs() >= threshold) | (correlation_matrix.abs() == 1.0)
        correlation_matrix = correlation_matrix.where(mask)

    # Color map selection
    colormap = st.selectbox(
        "🎨 Choose a color theme:",
        ["coolwarm", "viridis", "magma", "cividis", "Blues", "Greens"],
        index=0
    )

    # Show correlation values
    show_values = st.checkbox("🔢 Show correlation values", value=True, key="show_corr_values")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        correlation_matrix,
        annot=show_values,
        cmap=colormap,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)





from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

df_clean = df_merged[['Avg Temperature', 'Avg Humidity', 'Min Temperature', 'Max Temperature', 'Number of Insects', 'Cleaning_Flag', 'temp_change', 'New Captures (per Event)']]



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Assuming df_merged is loaded correctly and pre-processed

# Sidebar for user inputs
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose Model", ("Logistic Regression", "Random Forest", "XGBoost"))

# Train-test split (same for all models)
X = df_merged[['Avg Temperature', 'Avg Humidity', 'Min Temperature', 'Max Temperature', 'Number of Insects', 'Cleaning_Flag', 'temp_change']]
y = df_merged['New Captures (per Event)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to plot ROC Curve
def plot_roc_curve(fpr, tpr, model_name, roc_auc):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {model_name}')
    ax.legend(loc='lower right')
    st.pyplot(fig)

# Models
if model_option == "Logistic Regression":
    logreg = LogisticRegression(solver='liblinear', random_state=42)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, "Logistic Regression", roc_auc_score(y_test, y_pred))

elif model_option == "Random Forest":
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, "Random Forest", roc_auc_score(y_test, y_pred))

elif model_option == "XGBoost":
    xgb_model = xgb.XGBClassifier(scale_pos_weight=1, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, "XGBoost", roc_auc_score(y_test, y_pred))



# Additional Evaluation Metrics
if st.sidebar.checkbox("Show Evaluation Metrics"):
    st.subheader(f"Classification Report for {model_option}")
    st.text(classification_report(y_test, y_pred))

    st.subheader(f"Confusion Matrix for {model_option}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5, cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f"Confusion Matrix for {model_option}")
    st.pyplot(fig)



st.title("🔥 Correlation Analysis")

# Create tabs
tabs = st.tabs([
    "Pearson (Cicalino)", "Spearman (Cicalino)", "Kendall (Cicalino)",
    "Pearson (Imola)", "Spearman (Imola)", "Kendall (Imola)"
])

# Image paths
image_paths = [
    "https://raw.githubusercontent.com/seyedehsara/streamlit/main/heatmap%20c1.png",    # Linear heatmap
    "https://raw.githubusercontent.com/seyedehsara/streamlit/main/spearman%20c1.png",    # Spearman
    "https://raw.githubusercontent.com/seyedehsara/streamlit/main/kendall%20c1.png",     # Kendall
    "https://raw.githubusercontent.com/seyedehsara/streamlit/main/heatmap%20I1.png",     # Imola 1
    "https://raw.githubusercontent.com/seyedehsara/streamlit/main/spearman%20i1.png",    # Imola 2
    "https://raw.githubusercontent.com/seyedehsara/streamlit/main/kendall%20i1.png",     # Imola 3
]


# Display images in respective tabs
for tab, image_path in zip(tabs, image_paths):
    with tab:
        st.image(image_path, use_container_width=True)


st.header("Number of insect caught over time")

st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/number%20of%20insect%20caught%20over%20time.png", use_container_width=True)





st.title("📊 Insect Count")

# Radio button selection
option = st.radio(
    "Choose the dataset to display:",
    ["DFC1 Over Time", "DFC2 Over Time", "Total Count"]
)

# Image mapping
image_paths = {
    "DFC1 Over Time": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/insect%20count%20dfc1.png",  # DFC1 Over Time
    "DFC2 Over Time": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/insect%20count%20dfc2.png",  # DFC2 Over Time
    "Total Count": "https://raw.githubusercontent.com/seyedehsara/streamlit/main/total%20count.png",  # Total Count
}


# Display selected image
st.image(image_paths[option], caption=option, use_container_width=True)

st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/p%2Cq%2Cd.png", use_container_width=True)


st.title("Insect count from all")
st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/insect%20count%20dfc1%20over%20time.png", use_container_width=True)


st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/p%2Cq%2Cd%2C2.png", use_container_width=True)


st.title("prediction ARIMA")
st.image("https://raw.githubusercontent.com/seyedehsara/streamlit/main/arima.png", use_container_width=True)



