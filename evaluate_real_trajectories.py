import sys
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath('.'))

from src.core.uwb.imu import IMUSimulator, IMUData
from src.core.motion.motion_classifier import IndependentMotionClassifier
from src.core.uwb.uwb_devices import Position

class TagMock:
    def __init__(self):
        self.imu_data = IMUData(max_samples=20000)
        self.velocity = type('Vel', (), {'x':0.0, 'y':0.0, 'z':0.0})()
        self.angular_velocity = 0.0

def process_trajectory(csv_path):
    print(f"\n--- Processing {os.path.basename(csv_path)} ---")
    df = pd.read_csv(csv_path)
    
    if len(df) < 2:
        print("Dataset too small.")
        return
        
    df = df.sort_values('timestamp').reset_index(drop=True)
    t = df['timestamp'].values
    x = df['x'].values
    y = df['y'].values
    
    # Calculate kinematics
    dt_arr = np.diff(t)
    # Avoid zero division
    dt_arr[dt_arr == 0] = 1e-6
    
    vx = np.diff(x) / dt_arr
    vy = np.diff(y) / dt_arr
    
    vx = np.insert(vx, 0, vx[0])
    vy = np.insert(vy, 0, vy[0])
    
    ax = np.diff(vx) / dt_arr
    ay = np.diff(vy) / dt_arr
    ax = np.insert(ax, 0, ax[0])
    ay = np.insert(ay, 0, ay[0])
    
    yaw = np.arctan2(vy, vx)
    # Unwrap yaw
    yaw = np.unwrap(yaw)
    
    w = np.diff(yaw) / dt_arr
    w = np.insert(w, 0, w[0])
    
    methods = [
        "Method 1: Gyro Kinematic",
        "Method 2: PCA",
        "Method 3: Centripetal",
        "Method 4: Variance/ZCR"
    ]
    
    sim = IMUSimulator(sample_rate=100, acc_noise_std=0.05, gyro_noise_std=0.01)
    
    results = {m: {'y_true': [], 'y_pred': []} for m in methods}
    
    classifiers = {m: IndependentMotionClassifier() for m in methods}
    tags = {m: TagMock() for m in methods}
    
    for i in range(len(t)):
        if i == 0: continue
        dt = t[i] - t[i-1]
        if dt <= 0: continue
        
        pos = Position(x[i], y[i], 0.0)
        vel_world = np.array([vx[i], vy[i], 0.0])
        accel_world = np.array([ax[i], ay[i], 0.0])
        angular_vel_world = np.array([0, 0, w[i]])
        
        acc_meas, gyro_meas = sim.generate_imu_data(
            pos, yaw[i], dt, velocity=vel_world, acceleration=accel_world, angular_velocity=angular_vel_world
        )
        
        for m in methods:
            tag = tags[m]
            tag.velocity.x = vx[i]
            tag.velocity.y = vy[i]
            tag.angular_velocity = w[i]
            
            tag.imu_data.add_measurement(t[i], acc_meas[0], acc_meas[1], acc_meas[2], gyro_meas[0], gyro_meas[1], gyro_meas[2])
            
            res = classifiers[m].update(tag, selected_method=m)
            if res is not None:
                gt, pred = res
                results[m]['y_true'].append(gt)
                results[m]['y_pred'].append(pred)

    for m in methods:
        y_true = results[m]['y_true']
        y_pred = results[m]['y_pred']
        
        if len(y_true) == 0:
            print(f"{m}: Not enough data for prediction")
            continue
            
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        print(f"{m}: Acc={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

if __name__ == '__main__':
    traj_dir = os.path.join('data', 'trajectories')
    for file in os.listdir(traj_dir):
        if file.endswith('.csv'):
            process_trajectory(os.path.join(traj_dir, file))
