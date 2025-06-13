import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(50, 10, 100),   # Main data
    [10, 100]                        # Two outliers
])
df = pd.DataFrame({'Value': data})

# Compute statistics
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
print("IQR: ", IQR)
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR
median = np.median(data)
print("Lower: ", lower_whisker)
print("Upper: ", upper_whisker)

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Value', data=df, whis=1.5, orient='h', color='lightblue')

# Annotations
plt.axvline(Q1, color='orange', linestyle='--', label='Q1 (25th percentile)')
plt.axvline(Q3, color='orange', linestyle='--', label='Q3 (75th percentile)')
plt.axvline(median, color='green', linestyle='-', label='Median')

plt.axvline(lower_whisker, color='red', linestyle='--', label='Lower Whisker (Q1 - 1.5*IQR)')
plt.axvline(upper_whisker, color='red', linestyle='--', label='Upper Whisker (Q3 + 1.5*IQR)')

plt.title("Boxplot with 1.5×IQR Whiskers")
plt.legend()
plt.tight_layout()
plt.show()
