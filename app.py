# Streamlit Iris App
# This app demonstrates the Iris dataset using Streamlit.
# You can view data, filter species, and visualize features interactively.
# Streamlit is used to quickly build web apps without HTML/CSS knowledge.

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# -----------------------------
# Load Iris dataset
# -----------------------------
iris_data = load_iris()
df = pd.DataFrame(
    iris_data.data, columns=iris_data.feature_names
)
df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# -----------------------------
# App Title
# -----------------------------
st.title("🌸 Iris Dataset Streamlit App")
st.write("Interactive visualization and exploration of the Iris dataset.")

# -----------------------------
# Show Dataset
# -----------------------------
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)

# -----------------------------
# Filter by species
# -----------------------------
species_list = df['species'].unique().tolist()
selected_species = st.multiselect("Select species to view", species_list, default=species_list)
filtered_df = df[df['species'].isin(selected_species)]

st.write(f"Showing data for: {', '.join(selected_species)}")
st.dataframe(filtered_df)

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("📊 Feature Plots")

# Scatter plot of two selected features
feature_x = st.selectbox("Select X-axis feature", df.columns[:-1], index=0)
feature_y = st.selectbox("Select Y-axis feature", df.columns[:-1], index=1)

fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x=feature_x, y=feature_y, hue='species', ax=ax)
ax.set_title(f"{feature_x} vs {feature_y}")
st.pyplot(fig)

# Histogram of selected feature
st.subheader("📈 Feature Histogram")
feature_hist = st.selectbox("Select feature for histogram", df.columns[:-1], index=0)
bins = st.slider("Number of bins", 5, 50, 15)

fig2, ax2 = plt.subplots()
ax2.hist(filtered_df[feature_hist], bins=bins, color='skyblue', edgecolor='black')
ax2.set_title(f"Histogram of {feature_hist}")
ax2.set_xlabel(feature_hist)
ax2.set_ylabel("Count")
st.pyplot(fig2)

