import nbformat
import shutil

nb_path = 'motion_classification_analysis.ipynb'
backup_path = 'motion_classification_analysis_backup.ipynb'

# Restore notebook from backup to avoid appending multiple times
shutil.copyfile(backup_path, nb_path)

# Read notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

markdown_cell = nbformat.v4.new_markdown_cell(source="""## 4. 2D Trajectory Evaluation on Real Data
In this section, we process real trajectory CSVs (`log_traj.csv` and `test_calss.csv`) and visualize the 2D path in the X-Y plane. The color of the points corresponds to the identified motion state at that point in time (Linear = Blue, Circular = Green, Random = Red), allowing us to compare the predicted classifications against the estimated ground truth geometrically.""")

code_cell = nbformat.v4.new_code_cell(source="""def evaluate_real_trajectory(csv_path):
    print(f"\\n--- Evaluating Real Trajectory: {csv_path} ---")
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return
        
    df = df.sort_values('timestamp').reset_index(drop=True)
    t = df['timestamp'].values
    x = df['x'].values
    y = df['y'].values
    
    # Calculate kinematics
    dt_arr = np.diff(t)
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
    
    # Store state at each point
    gt_states = []
    pred_states = {m: [] for m in methods}
    valid_indices = []
    
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
        
        gt = "Unknown"
        preds = {m: "Unknown" for m in methods}
        has_pred = False
        
        for m in methods:
            tag = tags[m]
            tag.velocity.x = vx[i]
            tag.velocity.y = vy[i]
            tag.angular_velocity = w[i]
            
            tag.imu_data.add_measurement(t[i], acc_meas[0], acc_meas[1], acc_meas[2], gyro_meas[0], gyro_meas[1], gyro_meas[2])
            
            res = classifiers[m].update(tag, selected_method=m)
            if res is not None:
                gt, pred = res
                preds[m] = pred
                has_pred = True
                
        if has_pred:
            valid_indices.append(i)
            gt_states.append(gt)
            for m in methods:
                pred_states[m].append(preds[m])

    if len(valid_indices) == 0:
        print("Not enough data to run sliding windows.")
        return

    # Plot 2D trajectories colored by state
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    state_to_color = {"Linear Motion": "blue", "Circular/Curvilinear Motion": "green", "Random Walk": "red", "Unknown": "gray"}
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in state_to_color.items()]
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'2D Trajectory with Motion State Classification\\nDataset: {csv_path}', y=1.05, fontsize=16)
    
    plot_x = x[valid_indices]
    plot_y = y[valid_indices]
    
    # 1. Plot Ground Truth
    colors_gt = [state_to_color.get(s, "gray") for s in gt_states]
    axes[0].scatter(plot_x, plot_y, c=colors_gt, s=15)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].axis('equal')
    
    # 2. Plot Methods
    for i, m in enumerate(methods):
        colors_pred = [state_to_color.get(s, "gray") for s in pred_states[m]]
        axes[i+1].scatter(plot_x, plot_y, c=colors_pred, s=15)
        
        # Calculate accuracy for title
        y_t = [s for s in gt_states]
        y_p = [s for s in pred_states[m]]
        acc = accuracy_score(y_t, y_p)
        
        axes[i+1].set_title(f"{m}\\nAcc: {acc:.2f}")
        axes[i+1].set_xlabel("X (m)")
        axes[i+1].axis('equal')
        
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.show()

# Evaluate on specific trajectories
evaluate_real_trajectory('data/trajectories/log_traj.csv')
evaluate_real_trajectory('data/trajectories/test_calss.csv')""")

nb.cells.extend([markdown_cell, code_cell])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated with 2D plotting successfully.")
