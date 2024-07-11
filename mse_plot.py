import pandas as pd
import matplotlib.pyplot as plt

# Data for the first csv
data1 = {
    'S': [10, 20, 30, 40, 50],
    'MSE_without_filter': [770.46, 344.27, 166.36, 79.12, 36.72],
    'MSE_with_filter': [655.05, 338.94, 213.40, 151.28, 126.79]
}

# Data for the second csv
data2 = {
    'S': [10, 30, 50, 100, 150],
    'MSE_without_filter': [864.76, 538.39, 428.65, 245.84, 144.30],
    'MSE_with_filter': [786.70, 520.13, 441.49, 318.85, 258.61]
}

# Convert to pandas DataFrame
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Plotting
plt.figure(figsize=(14, 6))

# Plot for the first csv
plt.subplot(1, 2, 1)
plt.plot(df1['S'], df1['MSE_without_filter'], label='Without Filter', marker='o')
plt.plot(df1['S'], df1['MSE_with_filter'], label='With Filter', marker='o')
plt.title('MSE vs. S for Boat Image')
plt.xlabel('S')
plt.ylabel('MSE')
plt.legend()

# Plot for the second csv
plt.subplot(1, 2, 2)
plt.plot(df2['S'], df2['MSE_without_filter'], label='Without Filter', marker='o')
plt.plot(df2['S'], df2['MSE_with_filter'], label='With Filter', marker='o')
plt.title('MSE vs. S for Nature Image')
plt.xlabel('S')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()
