import pandas as pd
import os

file_path = r'c:\Users\tmoussa\Desktop\Irisa Job\PULSE project\data\trajectories\walking_measured.csv'

# Load the data
df = pd.read_csv(file_path)

# First point position
first_point = df.iloc[0]
x1, y1, z1 = first_point['x'], first_point['y'], first_point['z']

# Determine time step (assuming constant)
if len(df) > 1:
    dt = df.iloc[1]['timestamp'] - df.iloc[0]['timestamp']
else:
    dt = 0.02 # fallback

num_fixing_points = 100

# Create the fixing points
fixing_data = []
for i in range(num_fixing_points):
    fixing_data.append([i * dt, x1, y1, z1])

# Shift original timestamps
df['timestamp'] = df['timestamp'] + (num_fixing_points * dt)

# Combine
fixing_df = pd.DataFrame(fixing_data, columns=['timestamp', 'x', 'y', 'z'])
new_df = pd.concat([fixing_df, df], ignore_index=True)

# Save back
new_df.to_csv(file_path, index=False, float_format='%.6f')

print(f"Added {num_fixing_points} fixing points at the start. Total points now: {len(new_df)}")
