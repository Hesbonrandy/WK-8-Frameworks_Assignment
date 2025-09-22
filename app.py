import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# === Load and Prepare Data ===
@st.cache_data
def load_data():
    df = pd.read_csv('metadata.csv')
    df = df.dropna(subset=['title', 'abstract'], how='all').copy()
    df['journal'].fillna('Unknown', inplace=True)
    df.dropna(subset=['publish_time'], inplace=True)
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df.dropna(subset=['publish_time'], inplace=True)
    df['year'] = df['publish_time'].dt.year
    return df

df = load_data()

# === App Title ===
st.title("CORD-19 Research Explorer")
st.write("Explore trends in COVID-19 research papers from the CORD-19 dataset.")

# === Year Filter ===
min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_years = st.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(2020, 2021)
)

# Filter data
df_filtered = df[(df['year'] >= selected_years[0]) & (df['year'] <= selected_years[1])]

# === Display Stats ===
st.subheader(f"Showing {len(df_filtered)} papers from {selected_years[0]} to {selected_years[1]}")

# === Visualization 1: Publications Over Time (Filtered) ===
st.subheader("Publications Over Time")
year_counts = df_filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(year_counts.index, year_counts.values, color='lightcoral')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# === Visualization 2: Top Journals ===
st.subheader("Top Journals")
top_journals = df_filtered['journal'].value_counts().head(10)
fig, ax = plt.subplots()
ax.barh(top_journals.index, top_journals.values, color='teal')
ax.set_xlabel("Number of Papers")
st.pyplot(fig)

# === Visualization 3: Word Cloud ===
st.subheader("Most Common Words in Titles")
all_titles = ' '.join(df_filtered['title'].fillna('').str.lower())
words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles)
word_freq = Counter(words)
top_words = dict(word_freq.most_common(50))  # Top 50 for better cloud

if top_words:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.write("No words to display.")

# === Show Sample Data ===
st.subheader("Sample Papers")
st.dataframe(df_filtered[['title', 'journal', 'year', 'abstract']].head(10))

# === Footer ===
st.write("---")
st.caption("Built with Streamlit for educational purposes. Dataset: CORD-19.")