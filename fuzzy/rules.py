import pandas as pd

# Створення таблиці правил
data = [
    ['low', 'low', 'unstable', 'low'],
    ['low', 'low', 'average', 'low'],
    ['low', 'low', 'stable', 'low'],
    ['low', 'medium', 'unstable', 'low'],
    ['low', 'medium', 'average', 'medium'],
    ['low', 'medium', 'stable', 'medium'],
    ['low', 'high', 'unstable', 'low'],
    ['low', 'high', 'average', 'medium'],
    ['low', 'high', 'stable', 'high'],
    ['medium', 'low', 'unstable', 'low'],
    ['medium', 'low', 'average', 'low'],
    ['medium', 'low', 'stable', 'medium'],
    ['medium', 'medium', 'unstable', 'low'],
    ['medium', 'medium', 'average', 'medium'],
    ['medium', 'medium', 'stable', 'high'],
    ['medium', 'high', 'unstable', 'medium'],
    ['medium', 'high', 'average', 'high'],
    ['medium', 'high', 'stable', 'high'],
    ['high', 'low', 'unstable', 'low'],
    ['high', 'low', 'average', 'medium'],
    ['high', 'low', 'stable', 'high'],
    ['high', 'medium', 'unstable', 'medium'],
    ['high', 'medium', 'average', 'high'],
    ['high', 'medium', 'stable', 'high'],
    ['high', 'high', 'unstable', 'medium'],
    ['high', 'high', 'average', 'high'],
    ['high', 'high', 'stable', 'high'],
]

# Створення DataFrame
df = pd.DataFrame(data, columns=['Price', 'Volume', 'Market Stability', 'Recommended Share'])


# Виведення таблиці
print(df)