import pandas as pd
import numpy as np

# Generate dummy data for QuantFlow_TrainingData.csv
n_rows = 1000
data = {
    '<DATE>': ['2023.01.01'] * n_rows,
    '<TIME>': ['00:00:00'] * n_rows,
    '<OPEN>': np.random.uniform(1800, 1900, n_rows),
    '<HIGH>': np.random.uniform(1900, 1910, n_rows),
    '<LOW>': np.random.uniform(1790, 1800, n_rows),
    '<CLOSE>': np.random.uniform(1800, 1900, n_rows),
    '<TICKVOL>': np.random.randint(10, 100, n_rows),
    '<VOL>': np.random.randint(10, 100, n_rows),
    '<SPREAD>': [10] * n_rows
}

df = pd.DataFrame(data)
df.to_csv('QuantFlow_TrainingData.csv', sep=' ', index=False)
print("Dummy QuantFlow_TrainingData.csv created.")
