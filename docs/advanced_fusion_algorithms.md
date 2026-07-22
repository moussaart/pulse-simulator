# Advanced Sensor Fusion Algorithms

This document presents a rigorous mathematical formulation of three localization algorithms implemented within the PULSE simulator: (1) IMU Speed Dead Reckoning, (2) NLOS-Aware Adaptive UWB-IMU Fusion Extended Kalman Filter (NA-AEKF), and (3) Duty-Cycled UWB-IMU Fusion NA-AEKF. For each algorithm, the complete state model, all matrices with their explicit element-wise definitions, and a formal pseudocode summary are provided.

---

## 1. IMU Speed Dead Reckoning

### 1.1. Overview

The IMU Speed Dead Reckoning algorithm estimates the tag's 2D position using exclusively inertial sensor data. It integrates gyroscope measurements to track heading and projects a known or assumed movement speed along that heading to derive velocity. A Zero-Velocity Update (ZUPT) mechanism detects stationary phases from accelerometer and gyroscope statistics, correcting gyroscope bias and preventing unbounded positional drift.

### 1.2. System Inputs

| Symbol | Description | Dimension |
|---|---|---|
| $\mathbf{a}_k = [a_x, a_y, a_z]^T$ | Raw accelerometer reading at time $k$ | $\mathbb{R}^3$ |
| $\boldsymbol{\omega}_k = [g_x, g_y, g_z]^T$ | Raw gyroscope reading at time $k$ | $\mathbb{R}^3$ |
| $S$ | Movement speed (known or assumed, default $1.0$ m/s) | $\mathbb{R}^+$ |
| $\Delta t$ | Sampling interval | $\mathbb{R}^+$ |

### 1.3. State Vector and Auxiliary Variables

**State vector** (4-dimensional):
$$
\mathbf{x}_k = \begin{bmatrix} x_k \\ y_k \\ v_{x,k} \\ v_{y,k} \end{bmatrix} \in \mathbb{R}^4
$$
where $(x_k, y_k)$ denotes the 2D position and $(v_{x,k}, v_{y,k})$ the 2D velocity.

**Auxiliary scalar variables** (maintained outside the Kalman state):
- $\theta_k \in \mathbb{R}$: Integrated yaw heading angle (radians).
- $b_{g,k} \in \mathbb{R}$: Estimated z-axis gyroscope bias (rad/s).

**Covariance and noise matrices** (initialised at $k=0$):
$$
\mathbf{P}_0 = 0.1 \cdot \mathbf{I}_4 = \begin{bmatrix} 0.1 & 0 & 0 & 0 \\ 0 & 0.1 & 0 & 0 \\ 0 & 0 & 0.1 & 0 \\ 0 & 0 & 0 & 0.1 \end{bmatrix}
$$
$$
\mathbf{Q} = 10^{-3} \cdot \mathbf{I}_4 = \begin{bmatrix} 0.001 & 0 & 0 & 0 \\ 0 & 0.001 & 0 & 0 \\ 0 & 0 & 0.001 & 0 \\ 0 & 0 & 0 & 0.001 \end{bmatrix}, \quad
\mathbf{R} = 10^{-3} \cdot \mathbf{I}_2 = \begin{bmatrix} 0.001 & 0 \\ 0 & 0.001 \end{bmatrix}
$$

### 1.4. ZUPT Detection

A sliding window of $W$ accelerometer norm samples is maintained:
$$
\mathcal{B}_k = \{ \|\mathbf{a}_{k-W+1}\|, \; \|\mathbf{a}_{k-W+2}\|, \; \dots, \; \|\mathbf{a}_k\| \}, \quad \text{where } \|\mathbf{a}_j\| = \sqrt{a_{x,j}^2 + a_{y,j}^2 + a_{z,j}^2}
$$

The sample variance of this buffer is:
$$
\sigma^2_{\mathcal{B}} = \frac{1}{W-1} \sum_{j=1}^{W} \left( \|\mathbf{a}_j\| - \bar{\mathcal{B}} \right)^2, \quad \bar{\mathcal{B}} = \frac{1}{W} \sum_{j=1}^{W} \|\mathbf{a}_j\|
$$

The tag is declared **stationary** if:
$$
\text{ZUPT} = \begin{cases}
\text{True}, & \text{if } \sigma^2_{\mathcal{B}} < \tau_{\text{zupt}} \;\;\text{AND}\;\; \|\boldsymbol{\omega}_k\| < \tau_{\text{gyro}} \\
\text{False}, & \text{otherwise}
\end{cases}
$$
with default thresholds $\tau_{\text{zupt}} = 0.08 \;\text{m}^2/\text{s}^4$ and $\tau_{\text{gyro}} = 0.05 \;\text{rad/s}$.

### 1.5. Heading Integration and Velocity Computation

**Case 1 — Stationary ($\text{ZUPT} = \text{True}$):**

The gyroscope bias is updated via Exponential Moving Average (EMA) with smoothing coefficient $\alpha_b = 0.05$:
$$
b_{g,k} = (1 - \alpha_b) \cdot b_{g,k-1} + \alpha_b \cdot g_{z,k}
$$

The heading $\theta_k$ remains unchanged and the velocity is zeroed:
$$
v_{x,k} = 0, \quad v_{y,k} = 0
$$

**Case 2 — Moving ($\text{ZUPT} = \text{False}$):**

The heading is integrated using the bias-corrected z-axis gyroscope rate:
$$
\theta_k = \theta_{k-1} + (g_{z,k} - b_{g,k-1}) \cdot \Delta t
$$

The velocity is projected from the movement speed along the current heading:
$$
v_{x,k} = S \cdot \cos(\theta_k), \quad v_{y,k} = S \cdot \sin(\theta_k)
$$

### 1.6. Position Propagation

The 2D position is propagated via first-order Euler integration:
$$
x_k = x_{k-1} + v_{x,k} \cdot \Delta t
$$
$$
y_k = y_{k-1} + v_{y,k} \cdot \Delta t
$$

### 1.7. System Outputs

| Symbol | Description |
|---|---|
| $(x_k, y_k)$ | Estimated 2D position |
| $\mathbf{x}_k$ | Full state vector $[x, y, v_x, v_y]^T$ |
| $\theta_k$ | Current heading angle |
| $b_{g,k}$ | Current gyroscope bias estimate |
| ZUPT flag | Boolean indicating stationarity |

### 1.8. Algorithm — IMU Speed Dead Reckoning

> **Algorithm 1: IMU Speed Dead Reckoning**
>
> ---
>
> **Input:** $\mathbf{a}_k, \boldsymbol{\omega}_k, S, \Delta t$
>
> **Output:** $(x_k, y_k), \mathbf{x}_k, \theta_k, b_{g,k}$
>
> ---
>
> 1. **Initialisation** ($k = 0$):
>    - $\mathbf{x}_0 \leftarrow [x_{\text{init}}, y_{\text{init}}, 0, 0]^T$
>    - $\theta_0 \leftarrow \theta_{\text{init}}$
>    - $b_{g,0} \leftarrow 0$
>    - $\mathcal{B} \leftarrow \emptyset$
>
> 2. **For each time step** $k = 1, 2, \dots$:
>
>    a. Compute $\|\mathbf{a}_k\| = \sqrt{a_{x,k}^2 + a_{y,k}^2 + a_{z,k}^2}$
>
>    b. Append $\|\mathbf{a}_k\|$ to $\mathcal{B}$; if $|\mathcal{B}| > W$, remove oldest entry
>
>    c. Compute $\sigma^2_{\mathcal{B}}$ (sample variance of $\mathcal{B}$)
>
>    d. Compute $\|\boldsymbol{\omega}_k\| = \sqrt{g_{x,k}^2 + g_{y,k}^2 + g_{z,k}^2}$
>
>    e. **ZUPT decision:** $\text{ZUPT} \leftarrow (\sigma^2_{\mathcal{B}} < \tau_{\text{zupt}}) \;\wedge\; (\|\boldsymbol{\omega}_k\| < \tau_{\text{gyro}})$
>
>    f. **If** ZUPT = True:
>       - $b_{g,k} \leftarrow (1 - \alpha_b) \cdot b_{g,k-1} + \alpha_b \cdot g_{z,k}$
>       - $v_{x,k} \leftarrow 0, \quad v_{y,k} \leftarrow 0$
>       - $\theta_k \leftarrow \theta_{k-1}$
>
>    g. **Else:**
>       - $\theta_k \leftarrow \theta_{k-1} + (g_{z,k} - b_{g,k-1}) \cdot \Delta t$
>       - $v_{x,k} \leftarrow S \cdot \cos(\theta_k)$
>       - $v_{y,k} \leftarrow S \cdot \sin(\theta_k)$
>
>    h. **Position update:**
>       - $x_k \leftarrow x_{k-1} + v_{x,k} \cdot \Delta t$
>       - $y_k \leftarrow y_{k-1} + v_{y,k} \cdot \Delta t$
>
>    i. **Return** $(x_k, y_k),\; \mathbf{x}_k = [x_k, y_k, v_{x,k}, v_{y,k}]^T,\; \theta_k,\; b_{g,k}$

---

## 2. NLOS-Aware Adaptive UWB-IMU Fusion EKF (NA-AEKF)

### 2.1. Overview

The NLOS-Aware Adaptive UWB-IMU Fusion Extended Kalman Filter (NA-AEKF) is a tightly coupled sensor fusion architecture combining UWB range observations with IMU acceleration readings. It extends a standard Adaptive Extended Kalman Filter (AEKF) with per-anchor NLOS gating: measurements flagged as Non-Line-Of-Sight are penalized by inflating their noise variance in the $\mathbf{R}$ matrix, thereby reducing their influence on the state correction without discarding them entirely. Both $\mathbf{R}$ and $\mathbf{Q}$ are adapted online from innovation statistics.

The filter operates in three modes: **UWB-only** (constant-acceleration prediction + UWB distance update), **IMU-only** (heading+speed dead reckoning + ZUPT), and **Hybrid** (IMU-enhanced prediction + combined UWB+IMU correction).

### 2.2. System Inputs

| Symbol | Description | Dimension |
|---|---|---|
| $d_i$ | UWB range measurement to anchor $i$ | $\mathbb{R}^+$ |
| $(x_i, y_i)$ | Known 2D coordinates of anchor $i$ | $\mathbb{R}^2$ |
| $\mathbf{a}_k = [a_x, a_y, a_z]^T$ | Raw IMU accelerometer reading | $\mathbb{R}^3$ |
| $\boldsymbol{\omega}_k = [g_x, g_y, g_z]^T$ | Raw IMU gyroscope reading | $\mathbb{R}^3$ |
| $\text{is\_nlos}[i] \in \{0, 1\}$ | NLOS flag per anchor ($0$=LOS, $1$=NLOS) | $\{0,1\}^n$ |
| $\Delta t$ | Sampling interval | $\mathbb{R}^+$ |

### 2.3. State Vector

The full internal state is an 8-dimensional vector partitioned into the EKF sub-state and auxiliary IMU variables:
$$
\mathbf{x}_{\text{full}} = \begin{bmatrix} \underbrace{x \\ y \\ v_x \\ v_y \\ a_x \\ a_y}_{\mathbf{x}_{\text{EKF}} \in \mathbb{R}^6} \\ \theta \\ b_g \end{bmatrix} \in \mathbb{R}^8
$$

The EKF operations (prediction, update, covariance) are applied exclusively on the 6-dimensional sub-state $\mathbf{x}_{\text{EKF}}$. The heading $\theta$ and gyroscope bias $b_g$ are maintained separately and updated via the same IMU integration logic described in Section 1.

### 2.4. Initialisation

$$
\mathbf{x}_{\text{EKF},0} = \begin{bmatrix} x_0 \\ y_0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{P}_0 = \begin{bmatrix}
5 & 0 & 0 & 0 & 0 & 0 \\
0 & 5 & 0 & 0 & 0 & 0 \\
0 & 0 & 10 & 0 & 0 & 0 \\
0 & 0 & 0 & 10 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

The large initial variances on velocity ($10$) reflect the absence of prior velocity knowledge; position variances ($5$) accommodate moderate initial uncertainty; acceleration variances ($1$) are lower since the tag is assumed initially static.

### 2.5. Prediction Phase (Dynamic Model)

#### 2.5.1. State Transition Matrix $\mathbf{F}$

A constant-acceleration kinematic model is employed. For a sampling interval $\Delta t$:
$$
\mathbf{F} = \begin{bmatrix}
1 & 0 & \Delta t & 0 & \frac{\Delta t^2}{2} & 0 \\
0 & 1 & 0 & \Delta t & 0 & \frac{\Delta t^2}{2} \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix} \in \mathbb{R}^{6 \times 6}
$$

This encodes:
- Position is extrapolated from velocity and acceleration: $x_{k+1} = x_k + v_x \Delta t + \frac{1}{2} a_x \Delta t^2$.
- Velocity is extrapolated from acceleration: $v_{x,k+1} = v_{x,k} + a_x \Delta t$.
- Acceleration is assumed constant: $a_{x,k+1} = a_{x,k}$.

#### 2.5.2. Noise Input Matrix $\mathbf{G}$

The process noise is driven by a piecewise white-noise jerk model. The noise input matrix maps jerk perturbations $[j_x, j_y]^T$ into state-space effects:
$$
\mathbf{G} = \begin{bmatrix}
\frac{\Delta t^3}{6} & 0 \\
0 & \frac{\Delta t^3}{6} \\
\frac{\Delta t^2}{2} & 0 \\
0 & \frac{\Delta t^2}{2} \\
\Delta t & 0 \\
0 & \Delta t
\end{bmatrix} \in \mathbb{R}^{6 \times 2}
$$

#### 2.5.3. Process Noise Covariance $\mathbf{Q}$

The initial process noise covariance is derived from the jerk model:
$$
\mathbf{Q} = \mathbf{G} \cdot \mathbf{Q}_{\text{jerk}} \cdot \mathbf{G}^T, \quad \mathbf{Q}_{\text{jerk}} = \sigma_j^2 \cdot \mathbf{I}_2
$$
where $\sigma_j^2 = 1.0$ (jerk variance). Expanding this product:
$$
\mathbf{Q} = \sigma_j^2 \begin{bmatrix}
\frac{\Delta t^6}{36} & 0 & \frac{\Delta t^5}{12} & 0 & \frac{\Delta t^4}{6} & 0 \\
0 & \frac{\Delta t^6}{36} & 0 & \frac{\Delta t^5}{12} & 0 & \frac{\Delta t^4}{6} \\
\frac{\Delta t^5}{12} & 0 & \frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2} & 0 \\
0 & \frac{\Delta t^5}{12} & 0 & \frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2} \\
\frac{\Delta t^4}{6} & 0 & \frac{\Delta t^3}{2} & 0 & \Delta t^2 & 0 \\
0 & \frac{\Delta t^4}{6} & 0 & \frac{\Delta t^3}{2} & 0 & \Delta t^2
\end{bmatrix}
$$

This $\mathbf{Q}$ is subsequently adapted online (see Section 2.7).

#### 2.5.4. State Prediction

The predicted state $\hat{\mathbf{x}}_{k|k-1}$ depends on the operating mode:

**If Stationary:**
$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \cdot \mathbf{x}_{k-1|k-1}, \quad \text{then set } \hat{v}_x = \hat{v}_y = \hat{a}_x = \hat{a}_y = 0
$$

**If IMU Active (Moving):**
The gyroscope is integrated to update the heading:
$$
\theta_k = \theta_{k-1} + (g_{z,k} - b_{g,k-1}) \cdot \Delta t
$$
A pseudo-velocity from IMU heading and the actual speed $S_k$ is computed:
$$
v_{x,\text{imu}} = S_k \cdot \cos(\theta_k), \quad v_{y,\text{imu}} = S_k \cdot \sin(\theta_k)
$$
The predicted velocity is then blended:
$$
\hat{v}_{x} = (1 - \omega) \cdot [\mathbf{F} \cdot \mathbf{x}_{k-1}]_{v_x} + \omega \cdot v_{x,\text{imu}}
$$
$$
\hat{v}_{y} = (1 - \omega) \cdot [\mathbf{F} \cdot \mathbf{x}_{k-1}]_{v_y} + \omega \cdot v_{y,\text{imu}}
$$
where $\omega = 0.7$ is the IMU blending weight.

**If UWB-Only (no IMU):**
$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \cdot \mathbf{x}_{k-1|k-1}
$$

#### 2.5.5. Covariance Prediction

$$
\mathbf{P}_{k|k-1} = \mathbf{F} \cdot \mathbf{P}_{k-1|k-1} \cdot \mathbf{F}^T + \mathbf{Q}
$$

### 2.6. Measurement Model

#### 2.6.1. Measurement Vector

The measurement vector combines $n$ UWB range observations and 2 IMU accelerometer readings:
$$
\mathbf{z}_k = \begin{bmatrix} d_1 \\ d_2 \\ \vdots \\ d_n \\ a_{x,\text{imu}} \\ a_{y,\text{imu}} \end{bmatrix} \in \mathbb{R}^{n+2}
$$

#### 2.6.2. Predicted Measurement Vector

$$
\hat{\mathbf{z}}_k = \mathbf{h}(\hat{\mathbf{x}}_{k|k-1}) = \begin{bmatrix}
\sqrt{(\hat{x} - x_1)^2 + (\hat{y} - y_1)^2} \\
\sqrt{(\hat{x} - x_2)^2 + (\hat{y} - y_2)^2} \\
\vdots \\
\sqrt{(\hat{x} - x_n)^2 + (\hat{y} - y_n)^2} \\
\hat{a}_x \\
\hat{a}_y
\end{bmatrix}
$$

#### 2.6.3. Jacobian Matrix $\mathbf{H}$

The Jacobian is constructed by stacking rows for UWB ranges and IMU accelerations:

For UWB anchor $i$ at position $(x_i, y_i)$ with predicted distance $\hat{d}_i = \sqrt{(\hat{x} - x_i)^2 + (\hat{y} - y_i)^2}$:
$$
\mathbf{H}_{\text{uwb},i} = \begin{bmatrix} \frac{\hat{x} - x_i}{\hat{d}_i} & \frac{\hat{y} - y_i}{\hat{d}_i} & 0 & 0 & 0 & 0 \end{bmatrix}
$$

For the IMU accelerometer ($a_x$ maps to state index 4, $a_y$ to state index 5):
$$
\mathbf{H}_{\text{imu}} = \begin{bmatrix} 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}
$$

The full Jacobian matrix is therefore:
$$
\mathbf{H} = \begin{bmatrix}
\frac{\hat{x}-x_1}{\hat{d}_1} & \frac{\hat{y}-y_1}{\hat{d}_1} & 0 & 0 & 0 & 0 \\
\frac{\hat{x}-x_2}{\hat{d}_2} & \frac{\hat{y}-y_2}{\hat{d}_2} & 0 & 0 & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\frac{\hat{x}-x_n}{\hat{d}_n} & \frac{\hat{y}-y_n}{\hat{d}_n} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix} \in \mathbb{R}^{(n+2) \times 6}
$$

#### 2.6.4. Initial Measurement Noise Covariance $\mathbf{R}$

The initial $\mathbf{R}$ is diagonal, with per-measurement variances:
$$
\mathbf{R}_0 = \begin{bmatrix}
\sigma_{\text{uwb}}^2 & & & & & \\
& \sigma_{\text{uwb}}^2 & & & & \\
& & \ddots & & & \\
& & & \sigma_{\text{uwb}}^2 & & \\
& & & & \sigma_{\text{imu}}^2 & \\
& & & & & \sigma_{\text{imu}}^2
\end{bmatrix} \in \mathbb{R}^{(n+2) \times (n+2)}
$$
with $\sigma_{\text{uwb}} = 0.19$ m and $\sigma_{\text{imu}} = 1.0$ m/s².

### 2.7. NLOS-Aware Adaptive Update Phase

#### 2.7.1. Innovation Vector

$$
\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})
$$

#### 2.7.2. Adaptive $\mathbf{R}$ Update with NLOS Gating

**Step 1 — Empirical innovation covariance:**
$$
\mathbf{C}_{\text{innov}} = \mathbf{y}_k \cdot \mathbf{y}_k^T \in \mathbb{R}^{(n+2) \times (n+2)}
$$

**Step 2 — Per-measurement new variance estimate:**
$$
r_{i,\text{new}} = \left| [\mathbf{C}_{\text{innov}}]_{ii} - [\mathbf{H} \cdot \mathbf{P}_{k|k-1} \cdot \mathbf{H}^T]_{ii} \right|, \quad i = 1, \dots, n+2
$$

**Step 3 — NLOS inflation (UWB rows only):**
$$
r_{i,\text{new}} \leftarrow
\begin{cases}
r_{i,\text{new}}, & \text{if } i \leq n \text{ and } \text{is\_nlos}[i] = 0 \;\; (\text{LOS}) \\
\lambda_{\text{NLOS}} \cdot r_{i,\text{new}}, & \text{if } i \leq n \text{ and } \text{is\_nlos}[i] = 1 \;\; (\text{NLOS}) \\
r_{i,\text{new}}, & \text{if } i > n \;\; (\text{IMU rows — never inflated})
\end{cases}
$$
where $\lambda_{\text{NLOS}} = 5.0$.

**Step 4 — Exponential smoothing:**
$$
\mathbf{R}_k = \alpha \cdot \mathbf{R}_{k-1} + (1 - \alpha) \cdot \text{diag}(r_{1,\text{new}}, \dots, r_{n+2,\text{new}})
$$
with smoothing factor $\alpha = 0.3$.

#### 2.7.3. Adaptive $\mathbf{Q}$ Update

The process noise covariance is scaled by a factor derived from the normalised innovation norm:
$$
\gamma_k = \max\left(1, \; \frac{\|\mathbf{y}_k\|}{n + 2}\right)
$$
$$
\mathbf{Q}_{\text{new}} = \gamma_k \cdot \mathbf{I}_6
$$
$$
\mathbf{Q}_k = \beta \cdot \mathbf{Q}_{k-1} + (1 - \beta) \cdot \mathbf{Q}_{\text{new}}
$$
with smoothing factor $\beta = 0.5$.

#### 2.7.4. EKF Correction

**Innovation covariance:**
$$
\mathbf{S}_k = \mathbf{H} \cdot \mathbf{P}_{k|k-1} \cdot \mathbf{H}^T + \mathbf{R}_k
$$

**Kalman gain:**
$$
\mathbf{K}_k = \mathbf{P}_{k|k-1} \cdot \mathbf{H}^T \cdot \mathbf{S}_k^{-1}
$$

**A posteriori state estimate:**
$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \cdot \mathbf{y}_k
$$

**A posteriori covariance:**
$$
\mathbf{P}_{k|k} = (\mathbf{I}_6 - \mathbf{K}_k \cdot \mathbf{H}) \cdot \mathbf{P}_{k|k-1}
$$

### 2.8. ZUPT Measurement Update

When the tag is detected as stationary, a supplementary pseudo-measurement forces velocities and accelerations to zero:

$$
\mathbf{z}_{\text{ZUPT}} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{H}_{\text{ZUPT}} = \begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}, \quad
\mathbf{R}_{\text{ZUPT}} = \begin{bmatrix}
10^{-4} & 0 & 0 & 0 \\
0 & 10^{-4} & 0 & 0 \\
0 & 0 & 10^{-4} & 0 \\
0 & 0 & 0 & 10^{-4}
\end{bmatrix}
$$

The innovation, gain, and corrections follow the standard EKF update:
$$
\mathbf{y}_{\text{ZUPT}} = \mathbf{z}_{\text{ZUPT}} - \mathbf{H}_{\text{ZUPT}} \cdot \hat{\mathbf{x}}_{k|k} = -\begin{bmatrix} \hat{v}_x \\ \hat{v}_y \\ \hat{a}_x \\ \hat{a}_y \end{bmatrix}
$$
$$
\mathbf{S}_{\text{ZUPT}} = \mathbf{H}_{\text{ZUPT}} \cdot \mathbf{P}_{k|k} \cdot \mathbf{H}_{\text{ZUPT}}^T + \mathbf{R}_{\text{ZUPT}}
$$
$$
\mathbf{K}_{\text{ZUPT}} = \mathbf{P}_{k|k} \cdot \mathbf{H}_{\text{ZUPT}}^T \cdot \mathbf{S}_{\text{ZUPT}}^{-1}
$$
$$
\hat{\mathbf{x}}_{k|k} \leftarrow \hat{\mathbf{x}}_{k|k} + \mathbf{K}_{\text{ZUPT}} \cdot \mathbf{y}_{\text{ZUPT}}
$$
$$
\mathbf{P}_{k|k} \leftarrow (\mathbf{I}_6 - \mathbf{K}_{\text{ZUPT}} \cdot \mathbf{H}_{\text{ZUPT}}) \cdot \mathbf{P}_{k|k}
$$

The extremely small $\mathbf{R}_{\text{ZUPT}}$ values ($10^{-4}$) enforce a near-certain constraint, effectively clamping velocities and accelerations to zero.

### 2.9. Covariance Repair

After each update cycle, the covariance matrix is repaired to guarantee symmetry and positive semi-definiteness:
$$
\mathbf{P} \leftarrow \frac{\mathbf{P} + \mathbf{P}^T}{2}
$$
$$
\text{If } \lambda_{\min}(\mathbf{P}) < 10^{-9}: \quad \mathbf{P} \leftarrow \mathbf{P} + (10^{-9} - \lambda_{\min}) \cdot \mathbf{I}_6
$$

### 2.10. Algorithm — NLOS-Aware UWB-IMU Fusion AEKF (NA-AEKF)

> **Algorithm 2: NLOS-Aware Adaptive UWB-IMU Fusion EKF**
>
> ---
>
> **Input:** $d_{1..n},\; (x_i, y_i)_{i=1}^{n},\; \mathbf{a}_k,\; \boldsymbol{\omega}_k,\; \text{is\_nlos}[1..n],\; \Delta t$
>
> **Output:** $(\hat{x}_k, \hat{y}_k),\; \mathbf{x}_{\text{full},k},\; \mathbf{P}_{k|k},\; \mathbf{Q}_k,\; \mathbf{R}_k$
>
> **Parameters:** $\alpha = 0.3,\; \beta = 0.5,\; \omega = 0.7,\; \lambda_{\text{NLOS}} = 5.0,\; \alpha_b = 0.05$
>
> ---
>
> 1. **Initialisation** ($k = 0$):
>    - $\mathbf{x}_{\text{EKF}} \leftarrow [x_0, y_0, 0, 0, 0, 0]^T$
>    - $\mathbf{P} \leftarrow \text{diag}(5, 5, 10, 10, 1, 1)$
>    - $\mathbf{Q} \leftarrow \mathbf{G} \cdot \sigma_j^2 \mathbf{I}_2 \cdot \mathbf{G}^T$
>    - $\theta \leftarrow \theta_0, \quad b_g \leftarrow 0$
>
> 2. **For each time step** $k = 1, 2, \dots$:
>
>    a. **Stationarity check**: Evaluate ZUPT condition (Section 1.4 logic)
>
>    b. **Prediction**:
>       - Build $\mathbf{F}$ from $\Delta t$ (Section 2.5.1)
>       - Compute $\hat{\mathbf{x}}_{k|k-1}$ using stationary / IMU / UWB-only mode (Section 2.5.4)
>       - Compute $\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$
>
>    c. **Build measurement vector** $\mathbf{z}_k$ and **Jacobian** $\mathbf{H}$ (Section 2.6)
>
>    d. **Innovation**: $\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$
>
>    e. **Adaptive $\mathbf{R}$**:
>       - $\mathbf{C}_{\text{innov}} = \mathbf{y}_k \mathbf{y}_k^T$
>       - $r_{i,\text{new}} = |[\mathbf{C}_{\text{innov}}]_{ii} - [\mathbf{H P}_{k|k-1} \mathbf{H}^T]_{ii}|$
>       - **For** $i = 1, \dots, n$: **if** $\text{is\_nlos}[i] = 1$: $r_{i,\text{new}} \leftarrow \lambda_{\text{NLOS}} \cdot r_{i,\text{new}}$
>       - $\mathbf{R}_k = \alpha \cdot \mathbf{R}_{k-1} + (1-\alpha) \cdot \text{diag}(r_{1,\text{new}}, \dots)$
>
>    f. **Adaptive $\mathbf{Q}$**:
>       - $\gamma = \max(1, \|\mathbf{y}_k\| / (n+2))$
>       - $\mathbf{Q}_k = \beta \cdot \mathbf{Q}_{k-1} + (1-\beta) \cdot \gamma \mathbf{I}_6$
>
>    g. **EKF correction**:
>       - $\mathbf{S} = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}_k$
>       - $\mathbf{K} = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}^{-1}$
>       - $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K} \mathbf{y}_k$
>       - $\mathbf{P}_{k|k} = (\mathbf{I}_6 - \mathbf{K} \mathbf{H}) \mathbf{P}_{k|k-1}$
>
>    h. **If stationary**: Apply ZUPT update (Section 2.8)
>
>    i. **Covariance repair**: Enforce symmetry and PSD (Section 2.9)
>
>    j. **Pack** $\mathbf{x}_{\text{full}} = [\hat{\mathbf{x}}_{k|k}^T, \theta_k, b_{g,k}]^T$
>
>    k. **Return** $(\hat{x}_k, \hat{y}_k),\; \mathbf{x}_{\text{full},k},\; \mathbf{P}_{k|k},\; \mathbf{Q}_k,\; \mathbf{R}_k$

---

## 3. Duty-Cycled UWB-IMU Fusion NA-AEKF

### 3.1. Overview

The Duty-Cycled UWB-IMU Fusion NA-AEKF extends the continuous NA-AEKF (Algorithm 2) by introducing a temporal duty-cycling mechanism for the UWB radio. The UWB transceiver alternates between active (hybrid UWB+IMU) and inactive (IMU-only) windows within a repeating cycle. This scheduling reduces UWB energy consumption while maintaining localization accuracy through continuous IMU-based prediction during UWB-off periods.

All EKF mathematics — state transition, Jacobian computation, NLOS gating, adaptive $\mathbf{R}$/$\mathbf{Q}$, ZUPT, and covariance repair — remain identical to Algorithm 2. The sole architectural difference lies in the duty-cycle gate and the persistent per-block $\mathbf{R}$ tracking.

### 3.2. Duty-Cycle Parameters

| Symbol | Description | Default |
|---|---|---|
| $T_{\text{cycle}}$ | Total period of one duty cycle | $4.0$ s |
| $T_{\text{active}}$ | Duration of the UWB-active (hybrid) window | $2.0$ s |
| $T_{\text{imu}}$ | Duration of the IMU-only window ($T_{\text{cycle}} - T_{\text{active}}$) | $2.0$ s |

These parameters can be dynamically overridden at runtime (e.g., by a Reinforcement Learning agent) via the `set_duty_cycle(cycle_length, active_window)` method.

### 3.3. Duty-Cycle Decision Logic

An internal elapsed-time counter $t_e$ is initialised to $0$ and incremented by $\Delta t$ at every tick. The current phase within the cycle determines UWB availability:
$$
\phi_k = t_e \mod T_{\text{cycle}}
$$
$$
\text{uwb\_enabled}_k =
\begin{cases}
\text{False}, & \text{if } \phi_k < T_{\text{imu}} \quad (\text{IMU-only window}) \\
\text{True}, & \text{if } \phi_k \geq T_{\text{imu}} \quad (\text{Hybrid window})
\end{cases}
$$

### 3.4. Persistent Per-Block $\mathbf{R}$ Tracking

In the standard NA-AEKF, $\mathbf{R}$ is a single combined matrix rebuilt each tick. In the duty-cycled variant, the diagonal of $\mathbf{R}$ is segregated into two persistent storage blocks:

$$
\mathbf{R}_{\text{uwb}} = \text{diag}(r_1, r_2, \dots, r_n) \in \mathbb{R}^{n \times n}
$$
$$
\mathbf{R}_{\text{imu}} = \text{diag}(r_{a_x}, r_{a_y}) \in \mathbb{R}^{2 \times 2}
$$

At each tick, the combined $\mathbf{R}$ is assembled from these blocks depending on which sensors are active:

**If UWB enabled (Hybrid):**
$$
\mathbf{R}_k = \text{diag}(\mathbf{R}_{\text{uwb}}, \mathbf{R}_{\text{imu}}) \in \mathbb{R}^{(n+2) \times (n+2)}
$$

**If UWB disabled (IMU-only):**
$$
\mathbf{R}_k = \mathbf{R}_{\text{imu}} \in \mathbb{R}^{2 \times 2}
$$

After the adaptive $\mathbf{R}$ smoothing step, the updated diagonal values are written back to the corresponding persistent block. When UWB is disabled, $\mathbf{R}_{\text{uwb}}$ is not modified and retains its last learned values. Upon reactivation, the smoothing resumes from the preserved state.

### 3.5. Measurement Vector Assembly

The measurement vector and Jacobian are dynamically sized based on the duty cycle:

**If UWB enabled:**
$$
\mathbf{z}_k = \begin{bmatrix} d_1 \\ \vdots \\ d_n \\ a_{x,\text{imu}} \\ a_{y,\text{imu}} \end{bmatrix}, \quad
\mathbf{H}_k = \begin{bmatrix}
\mathbf{H}_{\text{uwb}} \\
\mathbf{H}_{\text{imu}}
\end{bmatrix} \in \mathbb{R}^{(n+2) \times 6}
$$

**If UWB disabled:**
$$
\mathbf{z}_k = \begin{bmatrix} a_{x,\text{imu}} \\ a_{y,\text{imu}} \end{bmatrix}, \quad
\mathbf{H}_k = \mathbf{H}_{\text{imu}} = \begin{bmatrix} 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 6}
$$

All subsequent EKF operations (innovation, adaptive $\mathbf{R}$, adaptive $\mathbf{Q}$, Kalman gain, state/covariance correction, ZUPT, covariance repair) proceed identically to Algorithm 2.

### 3.6. Algorithm — Duty-Cycled UWB-IMU NA-AEKF

> **Algorithm 3: Duty-Cycled NLOS-Aware Adaptive UWB-IMU Fusion EKF**
>
> ---
>
> **Input:** $d_{1..n},\; (x_i, y_i)_{i=1}^{n},\; \mathbf{a}_k,\; \boldsymbol{\omega}_k,\; \text{is\_nlos}[1..n],\; \Delta t$
>
> **Output:** $(\hat{x}_k, \hat{y}_k),\; \mathbf{x}_{\text{full},k},\; \mathbf{P}_{k|k},\; \mathbf{Q}_k,\; \mathbf{R}_k,\; \text{uwb\_enabled}_k$
>
> **Parameters:** Same as Algorithm 2, plus $T_{\text{cycle}},\; T_{\text{active}}$
>
> ---
>
> 1. **Initialisation** ($k = 0$):
>    - Same as Algorithm 2, Step 1
>    - $t_e \leftarrow 0$
>    - $\mathbf{R}_{\text{uwb}} \leftarrow \sigma_{\text{uwb}}^2 \cdot \mathbf{1}_n$
>    - $\mathbf{R}_{\text{imu}} \leftarrow \sigma_{\text{imu}}^2 \cdot \mathbf{1}_2$
>
> 2. **For each time step** $k = 1, 2, \dots$:
>
>    a. **Stationarity check**: Same as Algorithm 2
>
>    b. **Prediction**: Same as Algorithm 2 (build $\mathbf{F}$, predict $\hat{\mathbf{x}}_{k|k-1}$, $\mathbf{P}_{k|k-1}$)
>
>    c. **Duty-cycle gate**:
>       - $t_e \leftarrow t_e + \Delta t$
>       - $\phi_k \leftarrow t_e \mod T_{\text{cycle}}$
>       - $T_{\text{imu}} \leftarrow T_{\text{cycle}} - T_{\text{active}}$
>       - **If** $\phi_k \geq T_{\text{imu}}$: $\text{uwb\_enabled} \leftarrow \text{True}$
>       - **Else:** $\text{uwb\_enabled} \leftarrow \text{False}$
>
>    d. **Build measurement vector** $\mathbf{z}_k$ and **Jacobian** $\mathbf{H}$:
>       - Include UWB rows only if $\text{uwb\_enabled} = \text{True}$
>       - Include IMU rows if IMU data is available
>
>    e. **Assemble** $\mathbf{R}$ from persistent blocks $\mathbf{R}_{\text{uwb}},\; \mathbf{R}_{\text{imu}}$
>
>    f. **NLOS-aware adaptive update**: Same as Algorithm 2, Steps (d)–(g)
>
>    g. **Persist** updated $\mathbf{R}$ diagonal back to $\mathbf{R}_{\text{uwb}}$ and $\mathbf{R}_{\text{imu}}$
>
>    h. **If stationary**: Apply ZUPT update (Section 2.8)
>
>    i. **Covariance repair**: Enforce symmetry and PSD (Section 2.9)
>
>    j. **Pack** $\mathbf{x}_{\text{full}} = [\hat{\mathbf{x}}_{k|k}^T, \theta_k, b_{g,k}]^T$
>
>    k. **Return** $(\hat{x}_k, \hat{y}_k),\; \mathbf{x}_{\text{full},k},\; \mathbf{P}_{k|k},\; \mathbf{Q}_k,\; \mathbf{R}_k,\; \text{uwb\_enabled}_k$

---

## 4. Summary of Key Default Parameters

| Parameter | Symbol | Value | Used In |
|---|---|---|---|
| UWB measurement noise | $\sigma_{\text{uwb}}$ | $0.19$ m | Alg. 2, 3 |
| IMU measurement noise | $\sigma_{\text{imu}}$ | $1.0$ m/s² | Alg. 2, 3 |
| Jerk variance | $\sigma_j^2$ | $1.0$ | Alg. 2, 3 |
| R smoothing factor | $\alpha$ | $0.3$ | Alg. 2, 3 |
| Q smoothing factor | $\beta$ | $0.5$ | Alg. 2, 3 |
| NLOS inflation factor | $\lambda_{\text{NLOS}}$ | $5.0$ | Alg. 2, 3 |
| IMU velocity blend weight | $\omega$ | $0.7$ | Alg. 2, 3 |
| Gyro bias EMA factor | $\alpha_b$ | $0.05$ | Alg. 1, 2, 3 |
| ZUPT accel threshold | $\tau_{\text{zupt}}$ | $0.08$ m²/s⁴ | Alg. 1 |
| ZUPT accel threshold (fusion) | $\tau_{\text{zupt}}$ | $0.08$ m/s² | Alg. 2, 3 |
| Gyro stillness threshold | $\tau_{\text{gyro}}$ | $0.05$ – $0.1$ rad/s | Alg. 1, 2, 3 |
| Sliding window size | $W$ | $5$ | Alg. 1 |
| ZUPT noise variance | $\mathbf{R}_{\text{ZUPT}}$ | $10^{-4} \cdot \mathbf{I}_4$ | Alg. 2, 3 |
| Duty cycle period | $T_{\text{cycle}}$ | $4.0$ s | Alg. 3 |
| Active window | $T_{\text{active}}$ | $2.0$ s | Alg. 3 |
