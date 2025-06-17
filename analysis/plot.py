import pandas as pd
import numpy as np

# Load your CSV file
# Assuming your CSV has columns like 'timestamp', 'odom_y', 'filtered_y'
# Adjust column names based on your actual CSV export
df = pd.read_csv('analysis/odom_y_to_filteredstate_y.csv')

# --- Calculate the Difference ---
# Ensure you have columns for both the odom and filtered y-positions.
# Let's assume your columns are named 'odom_y' and 'filtered_y'
# If your CSV has separate columns for pose and position, you'll need to
# combine or select them appropriately before this step.
# For example, if you have columns like:
# timestamp, odom_pose_x, odom_pose_y, odom_pose_z, ...
# you'd select odom_pose_y and filtered_state_pose_pose_position_y

# Example assuming columns are directly named 'odom_y' and 'filtered_y'
# If your column names are different, replace them accordingly.
# You might need to find the exact column name from your CSV export.

# Let's try to infer column names if they are not obvious.
# You might have something like:
# '/odom/pose/pose/position/y' and '/filtered_state/pose/pose/position/y'
# after flattening the topic names in export.
# Let's assume for demonstration:
odom_col = '/odom/pose/pose/position/y'
filtered_col = '/filtered_state/pose/pose/position/y'

# Check if these columns exist
if odom_col in df.columns and filtered_col in df.columns:
    df['difference'] = df[odom_col] - df[filtered_col]

    # --- Calculate the Cumulative Sum (Integral Approximation) ---
    # np.cumsum calculates the cumulative sum of the 'difference' column
    df['cumulative_difference'] = np.cumsum(df['difference'])

    # Display the results
    print("DataFrame with Difference and Cumulative Difference:")
    print(df[['difference', 'cumulative_difference']].head())

    # You can now save this to a new CSV if needed
    df.to_csv('data_with_difference.csv', index=False)

else:
    print("Error: Could not find expected columns in the CSV.")
    print("Available columns:", df.columns)
    print("Please check your CSV export and adjust column names accordingly.")


# --- How to plot this using Python (e.g., with Matplotlib) ---
import matplotlib.pyplot as plt

if 'difference' in df.columns and 'cumulative_difference' in df.columns:
    plt.figure(figsize=(12, 6))

    # Plot the difference
    plt.subplot(2, 1, 1) # 2 rows, 1 column, first plot
    plt.plot(df.index, df['difference'], label='Difference (odom - filtered)')
    plt.title('Difference between Odom and Filtered State (Y-position)')
    plt.ylabel('Difference')
    plt.grid(True)
    plt.legend()

    # Plot the cumulative difference (integral approximation)
    plt.subplot(2, 1, 2) # 2 rows, 1 column, second plot
    plt.plot(df.index, df['cumulative_difference'], label='Cumulative Difference', color='orange')
    plt.title('Cumulative Sum of Difference (Integral Approximation)')
    plt.xlabel('Sample Index (or Time)')
    plt.ylabel('Cumulative Difference')
    plt.grid(True)
    plt.legend()

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()