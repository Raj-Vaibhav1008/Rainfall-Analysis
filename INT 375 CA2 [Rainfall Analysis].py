import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

file_path = r"/Users/rajvaibhav/Desktop/Sem-4 Docs:Notes/Rainfall Data.csv"
df = pd.read_csv(file_path)

print("Initial Shape:", df.shape)
df.dropna(inplace=True)
df.columns = df.columns.str.strip().str.upper().str.replace("-", "_")
print("After Cleaning Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

print("\nSummary Statistics:\n", df.describe())

monthly_avg = df[['JUN', 'JUL', 'AUG', 'SEP']].mean()

plt.figure()
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, palette="Blues")
plt.title("Average Rainfall (1901–2021) for Each Monsoon Month")
plt.ylabel("Rainfall (mm)")
plt.xlabel("Month")
plt.tight_layout()
plt.show()

rain_by_year = df.groupby('YEAR')['JUN_SEP'].mean().reset_index()

plt.figure()
sns.lineplot(x='YEAR', y='JUN_SEP', data=rain_by_year, color='green')
plt.title("Average Monsoon Rainfall Over Years (All Subdivisions)")
plt.ylabel("Rainfall (mm)")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

top_subdivisions = df.groupby('SUBDIVISION')['JUN_SEP'].mean().sort_values(ascending=False).head(5)

plt.figure()
sns.barplot(x=top_subdivisions.values, y=top_subdivisions.index, palette="viridis")
plt.title("Top 5 Wettest Subdivisions (Avg. Jun-Sep Rainfall)")
plt.xlabel("Average Rainfall (mm)")
plt.ylabel("Subdivision")
plt.tight_layout()
plt.show()

corr_matrix = df[['JUN', 'JUL', 'AUG', 'SEP']].corr()

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Monthly Rainfall")
plt.tight_layout()
plt.show()

plt.figure()
sns.boxplot(data=df[['JUN', 'JUL', 'AUG', 'SEP']], palette="Set2")
plt.title("Rainfall Distribution & Outliers (1901–2021)")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.show()

df['RAIN_CATEGORY'] = np.where(df['JUN_SEP'] > 2000, 'Heavy', 
                               np.where(df['JUN_SEP'] < 1000, 'Low', 'Normal'))

print("\nRainfall Category Distribution:\n", df['RAIN_CATEGORY'].value_counts())
