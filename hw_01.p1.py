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
print(df.head())
