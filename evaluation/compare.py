# tu bedzie porownanie wynikow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Opcje dla wąsów:
whiskers = [5, 95] # np.inf, [5, 95], 1.5 / min-max, 5 and 95 prontyl, 1.5 IQR

# List of CSV file paths
csv_files = [
    'fbl/feedback_linearization_results.csv',
    'rl/DDPG/DDPG_results.csv',
    'rl/PPO/PPO_results.csv',
    'rl/SAC/SAC_results.csv',
    'rl/TD3/TD3_results.csv'
]

labels = ['FBL', 'DDPG', 'PPO', 'SAC', 'TD3']

# Columns to extract
column_to_plot = 'energy_cost'
column_to_plot2 = 'reward'

combined_data = pd.DataFrame()
combined_data2 = pd.DataFrame()

for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    label = labels[i]
    if column_to_plot in df.columns:
        temp_df = df[[column_to_plot]].copy()
        temp_df['Source'] = label
        combined_data = pd.concat([combined_data, temp_df], ignore_index=True)
    if column_to_plot2 in df.columns:
        temp_df2 = df[[column_to_plot2]].copy()
        temp_df2['Source'] = label
        combined_data2 = pd.concat([combined_data2, temp_df2], ignore_index=True)

# Plotting both boxplots
fig, axes = plt.subplots(2, 1)  # 2 rows, 1 column

sns.boxplot(x='Source', y=column_to_plot, data=combined_data, ax=axes[0], whis=whiskers)
axes[0].set_title(f'Boxplot of "{column_to_plot}"')
axes[0].set_xlabel('CSV Files')
axes[0].set_ylabel(column_to_plot)
axes[0].grid(True)

sns.boxplot(x='Source', y=column_to_plot2, data=combined_data2, ax=axes[1], whis=whiskers)
axes[1].set_title(f'Boxplot of "{column_to_plot2}"')
axes[1].set_xlabel('CSV Files')
axes[1].set_ylabel(column_to_plot2)
axes[1].grid(True)

plt.tight_layout()
plt.show()
