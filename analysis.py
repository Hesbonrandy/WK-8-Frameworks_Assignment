# ==================================================
# PART 1: DATA LOADING AND BASIC EXPLORATION
# ==================================================

# Import libraries
import pandas as pd

# Load the data
print("Loading data...")
df = pd.read_csv('metadata.csv')

# Display first 5 rows
print("\n=== First 5 rows ===")
print(df.head())

# Check shape (rows, columns)
print(f"\n=== Dataset Shape ===\n{df.shape}")

# Check data types
print("\n=== Data Types ===")
print(df.dtypes)

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Basic stats for numerical columns
print("\n=== Basic Statistics ===")
print(df.describe())

# ==================================================
# PART 2: DATA CLEANING AND PREPARATION
# ==================================================

# Check percentage of missing values
print("\n=== Missing Values Percentage ===")
print((df.isnull().sum() / len(df)) * 100)

# Drop rows where both title and abstract are missing
df_clean = df.dropna(subset=['title', 'abstract'], how='all').copy()

# Fill missing 'journal' with 'Unknown'
df_clean['journal'].fillna('Unknown', inplace=True)

# Drop rows with missing publish_time
df_clean.dropna(subset=['publish_time'], inplace=True)

# Convert publish_time to datetime
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
df_clean.dropna(subset=['publish_time'], inplace=True)

# Extract year
df_clean['year'] = df_clean['publish_time'].dt.year

# Add abstract word count
df_clean['abstract_word_count'] = df_clean['abstract'].fillna('').str.split().str.len()

print(f"\n=== After cleaning: {df_clean.shape} rows remaining ===")
print("\n=== Sample after cleaning ===")
print(df_clean[['title', 'year', 'journal', 'abstract_word_count']].head())

# ==================================================
# PART 3: DATA ANALYSIS AND VISUALIZATION
# ==================================================

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# Set style
sns.set_style("whitegrid")

# 1. Papers by Year
year_counts = df_clean['year'].value_counts().sort_index()
print("\n=== Papers by Year ===")
print(year_counts)

# 2. Top Journals
top_journals = df_clean['journal'].value_counts().head(10)
print("\n=== Top 10 Journals ===")
print(top_journals)

# 3. Most Frequent Words in Titles
all_titles = ' '.join(df_clean['title'].fillna('').str.lower())
words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles)
word_freq = Counter(words)
top_words = word_freq.most_common(20)

print("\n=== Top 20 Words in Titles ===")
for word, count in top_words:
    print(f"{word}: {count}")

# Plot 1: Publications Over Time
plt.figure(figsize=(10, 5))
plt.bar(year_counts.index, year_counts.values, color='skyblue')
plt.title('Number of Publications by Year')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.xticks(year_counts.index)
plt.tight_layout()
plt.savefig('pubs_by_year.png')
plt.show()

# Plot 2: Top Journals
plt.figure(figsize=(10, 6))
sns.barplot(x=top_journals.values, y=top_journals.index, palette='viridis')
plt.title('Top 10 Journals Publishing COVID-19 Research')
plt.xlabel('Number of Papers')
plt.tight_layout()
plt.savefig('top_journals.png')
plt.show()

# Plot 3: Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(top_words))
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Paper Titles')
plt.savefig('wordcloud.png')
plt.show()

# Plot 4: Distribution by Source
source_counts = df_clean['source_x'].value_counts()
plt.figure(figsize=(8, 5))
plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Papers by Source')
plt.tight_layout()
plt.savefig('source_distribution.png')
plt.show()