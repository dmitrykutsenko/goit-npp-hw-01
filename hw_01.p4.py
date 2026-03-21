import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# 1. Завантажуємо дані
iris = load_iris()

# 2. Створюємо DataFrame з ознак
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Додаємо колонку з мітками класів
df['target'] = iris.target

# Вибираємо лише числові ознаки
X = df[iris.feature_names]

# Стандартизація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Перетворюємо назад у DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)

print(df_scaled.head())
