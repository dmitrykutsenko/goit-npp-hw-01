import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Завантажуємо дані
iris = load_iris()

# Створюємо DataFrame з ознак
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Додаємо колонку з мітками класів
df['target'] = iris.target

# Переглядаємо перші рядки
# print(df.head())

# Отримуємо базові статистичні характеристики
# stats = df.describe()
# print(stats)

df['class'] = df['target'].apply(lambda x: iris.target_names[x])

# Побудова графіка розподілу за класами
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='class', palette='viridis')

plt.title("Розподіл спостережень за класами Iris")
plt.xlabel("Клас квітки")
plt.ylabel("Кількість спостережень")
plt.show()

# Спробуємо ще Pairplot
df_pairplot = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_pairplot['class'] = iris.target_names[iris.target]
sns.pairplot(df_pairplot, hue='class', palette='viridis')
plt.show()

# Спробуємо ще Boxplot
plt.figure(figsize=(12, 6))
df_melted = df.melt(id_vars='class', var_name='feature', value_name='value')

sns.boxplot(data=df_melted, x='feature', y='value', hue='class', palette='viridis')
plt.xticks(rotation=45)
plt.title("Boxplot розподілу ознак за класами")
plt.show()

# Спробуємо ще Histplot
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=df, x=feature, hue='class', kde=True, palette='viridis')
    plt.title(f"Розподіл: {feature}")

plt.tight_layout()
plt.show()

# Спробуємо ще Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='sepal length (cm)',
    y='petal length (cm)',
    hue='class',
    palette='viridis',
    s=80
)
plt.title("Scatterplot: sepal length vs petal length")
plt.show()
