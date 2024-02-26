import pandas as pd
import numpy as np

# Creating a DataFrame with 4 different values
data = {'Value1': [10, 20, 30, 40],
        'Value2': [5.5, 8.2, 12.6, 15.3],
        'Value3': ['A', 'B', 'C', 'D'],
        'Value4': np.random.rand(4)}

df = pd.DataFrame(data)

# Displaying the DataFrame
print(df)
