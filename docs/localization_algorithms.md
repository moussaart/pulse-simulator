# Trilateration

In the context of localizing a tag in a three-dimensional space $(x, y, z)$, trilateration involves utilizing distances measured from the tag to multiple anchors with known coordinates to compute its spatial location. Each anchor yields an algebraic equation representing a sphere.

---

### 1. Equations of the Spheres

For each anchor $i$ whose position is given by $p_i = (x_i, y_i, z_i)$ and for which the measured distance to the tag is $r_i$, the equation of the corresponding sphere is formulating as follows:
$$
(x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2 = r_i^2.
$$
To deduce a unique spatial position in 3D, a minimum of four anchors (providing four distinct equations) is strictly requisite.


### 2. Linearization of the System via Subtraction

To derive a linear algebraic system, the equation of the primary anchor is subtracted from the equations of the subsequent anchors. For instance, considering anchor 2 relative to anchor 1:

$$
\begin{aligned}
&(x - x_2)^2 + (y - y_2)^2 + (z - z_2)^2 \\
&\quad - \left[(x - x_1)^2 + (y - y_1)^2 + (z - z_1)^2\right] = r_2^2 - r_1^2.
\end{aligned}
$$

By expanding the terms and eliminating the common quadratic variables, this yields:

$$
-2x(x_2 - x_1) - 2y(y_2 - y_1) - 2z(z_2 - z_1) + \big[(x_2^2 - x_1^2) + (y_2^2 - y_1^2) + (z_2^2 - z_1^2)\big] = r_2^2 - r_1^2.
$$

Multiplying the expression by $-1$, we obtain:

$$
2x(x_2 - x_1) + 2y(y_2 - y_1) + 2z(z_2 - z_1) = r_1^2 - r_2^2 + (x_2^2 - x_1^2) + (y_2^2 - y_1^2) + (z_2^2 - z_1^2).
$$

An identical analytical procedure is executed for anchors 3 and 4.


### 3. Matrix Formulation of the Linear System

The three derived equations (formulated for anchors 2, 3, and 4 relative to the first anchor) can be encapsulated in the following matrix formulation:
$$
A \begin{pmatrix} x \\ y \\ z \end{pmatrix} = b,
$$
where:
$$
A = 2 \begin{pmatrix}
x_2 - x_1 & y_2 - y_1 & z_2 - z_1 \\
x_3 - x_1 & y_3 - y_1 & z_3 - z_1 \\
x_4 - x_1 & y_4 - y_1 & z_4 - z_1 
\end{pmatrix},
$$
and
$$
b = \begin{pmatrix}
r_1^2 - r_2^2 + (x_2^2 - x_1^2) + (y_2^2 - y_1^2) + (z_2^2 - z_1^2) \\
r_1^2 - r_3^2 + (x_3^2 - x_1^2) + (y_3^2 - y_1^2) + (z_3^2 - z_1^2) \\
r_1^2 - r_4^2 + (x_4^2 - x_1^2) + (y_4^2 - y_1^2) + (z_4^2 - z_1^2)
\end{pmatrix}.
$$

Solving this linear system provides the continuous spatial coordinates $(x, y, z)$ of the tag:

$$
\begin{pmatrix} x \\ y \\ z \end{pmatrix} = A^{-1}b,
$$
In practical implementations, this system is frequently resolved numerically (for example, employing a least-squares minimization approach in scenarios involving supernumerary anchors or significant measurement noise).

### 4. Practical Considerations

- **Number of Anchors:** In a 3D environment, a minimum of four anchors is mandatory to secure a unique positional solution. Should redundant anchors be accessible, the solution can be optimally refined utilizing regression techniques (e.g., least squares) to systematically curtail estimation error.
- **Singular Matrix Occurrences:** If the geometric arrangement of the anchors causes the matrix $A$ to become singular or ill-conditioned (for instance, if they are collinear or predominantly coplanar), the resulting solution becomes irremediably imprecise or entirely indeterminate.

---

# Extended Kalman Filter (EKF) 

The Extended Kalman Filter (EKF) functions as a recursive algorithm tailored for state estimation in non-linear dynamical systems. In the context of our application, the primary objective is to continuously estimate the spatial position and velocity of a tag translating within a 2D plane by synergizing:

- A **dynamic model** (to extrapolate the prospective future state), and  
- A non-linear **measurement model** (to calibrate this prediction leveraging empirical sensor observations).

### 1. State Vector and Fundamental Concepts

**State Vector ($\mathbf{x}$):**  
This vector mathematically embodies the variables necessitating estimation. For a tag navigating a 2D plane, we establish:

$$
\mathbf{x} = \begin{pmatrix} x \\ y \\ v_x \\ v_y \end{pmatrix},
$$

- $x,\, y$: Cartesian position coordinates,  
- $v_x,\, v_y$: Corresponding directional velocity components.

**State Covariance Matrix ($\mathbf{P}$):**  
The matrix $\mathbf{P}$ parametrically quantifies the prevailing uncertainty (or corresponding error) intrinsically tied to the state estimation.

**Noise Components:**  
- **Process Noise ($\mathbf{w}_k$):** The intrinsic uncertainty characterizing the system's dynamic evolution.  
- **Measurement Noise ($v_i$):** The empirical uncertainty embedded within the sensor-derived measurements.  
These perturbative noise terms are conventionally modeled as uncorrelated Gaussian random variables.

### 2. Dynamic Model (Prediction Phase)

The dynamic model formally delineates the temporal evolution of the target's state under motion constraints.

#### 2.1. State Equation

$$
\mathbf{x}_{k+1} = \mathbf{F}\,\mathbf{x}_k + \mathbf{w}_k,
$$

- $\mathbf{x}_k$: Abstract state vector at a discrete time epoch $k$.  
- $\mathbf{w}_k$: Process noise, encapsulating the cumulative uncertainty in the assumed motion dynamics.

#### 2.2. State Transition Matrix ($\mathbf{F}$)

Assuming a simplified kinematic trajectory featuring a constant velocity uniformly sampled over an interval $dt$, the transition matrix is defined as:

$$
\mathbf{F} = \begin{pmatrix}
1 & 0 & dt & 0 \\
0 & 1 & 0 & dt \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{pmatrix}.
$$

This matrix functionally establishes that an updated positional geometry is governed linearly by the anterior position augmented by current kinematic velocity vectors.

#### 2.3. Process Noise and its Associated Covariance (\(\mathbf{Q}\))

The covariance matrix characterizing the process noise is functionally prescribed by:

$$
\mathbf{Q} = q\,\mathbf{I}_4,
$$

where $q$ constitutes a scalar coefficient (e.g., $q=0.1$) and $\mathbf{I}_4$ delineates a four-dimensional identity matrix. This tensor mathematically formalizes the prognostic uncertainty of the predictive phase.

### 2.4. Prediction Execution

The mathematical prediction of the future abstract state and its concomitant covariance matrix is executed via:

$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F}\,\hat{\mathbf{x}}_{k-1|k-1},
$$
$$
\mathbf{P}_{k|k-1} = \mathbf{F}\,\mathbf{P}_{k-1|k-1}\,\mathbf{F}^T + \mathbf{Q}.
$$

- $\hat{\mathbf{x}}_{k|k-1}$: The calculated predictive estimation of the state prior to measurement intake at interval $k$.  
- $\mathbf{P}_{k|k-1}$: The predictive dispersion covariance corresponding natively with this state approximation.


### 3. Measurement Model and Analytical Linearization (Update Phase)

The sensor measurements are exclusively derived from anchors characterized by established geolocations. Every independent observation functions as a distance derivation between the tag and a specific anchor.

#### 3.1. Measurement Equation

For a reference anchor $i$ spatially situated at $p_i = (x_i, y_i)$, the empirically logged distance is robustly parameterized by:

$$
r_i = h_i(\mathbf{x}) + v_i = \sqrt{(x - x_i)^2 + (y - y_i)^2} + v_i,
$$

where $v_i$ is explicitly defined as the Gaussian-distributed noise of the measurement device.

### 3.2. Measurement Noise Covariance ($\mathbf{R}$)

The matrix $\mathbf{R}$ operates as the parametric covariance tensor associated with the sensor measurement noise. Taking an ensemble of $m$ individual measurements, one can conceptually structure:

$$
\mathbf{R} = \sigma_r^2\,\mathbf{I}_m,
$$

where $\sigma_r^2$ functions as the statistical variance of the sensor's characteristic measurement noise.

#### 3.3. Linearization Methodology: The Jacobian Matrix ($\mathbf{H}$)

Acknowledging that the functional mapping of measurement $h_i(\mathbf{x})$ is profoundly non-linear, a first-order Taylor expansion is executed around the current state approximation $\hat{\mathbf{x}}$ to formulate the Jacobian matrix. Analytically, for each anchor $i$:

$$
\frac{\partial h_i}{\partial x} = \frac{x - x_i}{d_i}, \quad \frac{\partial h_i}{\partial y} = \frac{y - y_i}{d_i},
$$

alongside

$$
d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2}.
$$

Derivatives formulated with respect to velocity variables inherently vanish (given that physical distance mapping operates solely upon positional attributes):

$$
\frac{\partial h_i}{\partial v_x} = 0, \quad \frac{\partial h_i}{\partial v_y} = 0.
$$

In a network of $m$ simultaneous measurements, the Jacobian matrix systematically deploys in the following format:

$$
\mathbf{H} = \begin{pmatrix}
\frac{x - x_1}{d_1} & \frac{y - y_1}{d_1} & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots \\
\frac{x - x_m}{d_m} & \frac{y - y_m}{d_m} & 0 & 0 \\
\end{pmatrix}.
$$

The mathematical expectation vector of the measurements (the anticipated observation array) is analytically denoted by:

$$
\mathbf{h}(\hat{\mathbf{x}}) = \begin{pmatrix}
\sqrt{(x - x_1)^2 + (y - y_1)^2} \\
\vdots \\
\sqrt{(x - x_m)^2 + (y - y_m)^2} \\
\end{pmatrix}.
$$


### 4. Update of the State Estimate (Correction Step)

#### 4.1. Calculation of the Kalman Gain

The proportional Kalman gain $\mathbf{K}$ quantifies optimal weighting criteria calibrating the incoming empirical measurements against the predictive model forecast, analytically evaluated via:

$$
\mathbf{S} = \mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T + \mathbf{R},
$$
$$
\mathbf{K} = \mathbf{P}_{k|k-1}\,\mathbf{H}^T\,\mathbf{S}^{-1}.
$$

- $\mathbf{S}$ constitutes the covariance of the observation innovation (alternatively labeled as the pre-fit residual covariance tensor).

#### 4.2. Refining the State Vector and Covariance Attributes

The necessary post-priori state correction operates via:

$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}\,\left(\mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})\right),
$$
$$
\mathbf{P}_{k|k} = \left(\mathbf{I} - \mathbf{K}\,\mathbf{H}\right)\,\mathbf{P}_{k|k-1}.
$$

- $\mathbf{z}$ denotes the unadulterated empirical measurement vector array.  
- $\mathbf{I}$ executes as the identity matrix tensor.  
- The scalar algebraic difference represented by $\mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$ is formally characterized as the **innovation**.

### 5. Architectural Advantages of the EKF relative to Conventional Trilateration

- **Integration of Dynamical Temporal Frameworks:**  
  The inherent incorporation of temporal evolution within the EKF's predictive model dynamically links adjacent computational states. This grants precise predictive capabilities between consecutive observation epochs, an indispensable parameter when tracking kinetic tag propagation. Conversely, standard trilateration fundamentally functions merely to identify isolated stationary point data utterly bereft of historical extrapolation.

- **Mitigation Mechanisms for Stochastic Perturbations:**  
  The mathematical assimilation of parametric uncertainties via formalized tensors $\mathbf{Q}$ and $\mathbf{R}$ enables the EKF construct to robustly filter statistical aberrations, granting significantly smoothed localization outputs amid chaotic measurement variability constraints.

- **Comprehensive Estimation Functionalities:**  
  Coupled organically with pure Cartesian positions, the EKF system reliably aggregates tag velocity parameters. This supplementary kinematic intelligence fundamentally refines trajectory analytics facilitating vastly superior continuous behavioral predictions concerning target kinetics.

- **Iterative Correction Architectures:**  
  As real-world observation streams accumulate chronologically over discrete sub-intervals, the EKF calibrates its structural confidence mathematically, cultivating a progressive enhancement of accuracy trajectories—contrasting sharply with rudimentary trilateration protocols functionally restricted solely by instantaneous data thresholds without recursive processing attributes.

---

### Conclusion

This formalized mathematical schema pertaining to the 2D Extended Kalman Filter constructs a systematically bifurcated protocol framework: an initial state geometric prediction orchestrated via a discrete kinetical model, logically succeeded by analytic parametric correction governed dynamically per linearized non-linear observation capabilities via Jacobian derivation frameworks. Every intrinsic variable of the modeling architecture is formulated strictly to warrant unmitigated transparency relative to the ongoing numerical filtering operation. Ultimately, this schema executes as thoroughly deterministic, operationally robust, and intrinsically more computationally adept relative to unassisted point-bound linear trilateration algorithms for the prolonged and unyielding spatial localization tracking of dynamically operating target platforms.

---



## Adaptive Extended Kalman Filter (AEKF) 

### 1. State Vector and Fundamental Concepts

The definitions outlining the state vector, the covariance matrix, and the fundamental noise typologies remain strictly identical to those formulated in the preceding section detailing the standard Extended Kalman Filter (EKF).

### 2. Dynamic Model (Prediction Phase)

The dynamic motion paradigm and its correlative state equation are congruent with the standard EKF architecture, retaining the identical state transition matrix $\mathbf{F}$.

### 3. Non-Linear Measurement Model

The analytical measurement architecture and its subsequent analytical linearization leveraging the Jacobian matrix $\mathbf{H}$ proceed unimpeded and unaltered from the classical EKF structure.

### 4. Algorithmic Processing Steps

The systematic propagation of prediction protocols and measurement corrections functionally replicate the standard EKF schema delineated previously. This encompasses the derivation equations dictating the Kalman gain, alongside the discrete parametric updates applied to both state geometries and error covariance arrays.

The paramount functional divergence defining the AEKF resides explicitly in the incorporation of a mathematically adaptive mechanism as elaborated below.


### 5. Adaptive Mechanism Governing $\mathbf{R}$ and $\mathbf{Q}$

The AEKF algorithm iteratively and dynamically modulates the parametric arrays $\mathbf{R}$ and $\mathbf{Q}$ as a continuous function of real-time observed statistical innovations.

#### 5.1. Adaptation Dynamics of the Measurement Noise Covariance $\mathbf{R}$

1. **Analytical Calculation of the Innovation Vector:**
   $$
   \mathbf{y} = \mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1}).
   $$
   
2. **Estimated Empirical Innovation Covariance:**
   $$
   \mathbf{C}_{\text{innov}} = \mathbf{y}\,\mathbf{y}^T.
   $$
   
3. **Transient Estimation formulation for $\mathbf{R}$:**  
   By analytically extracting the projected baseline predictive uncertainty:
   $$
   \mathbf{R}_{\text{new}} = \mathbf{C}_{\text{innov}} - \mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T.
   $$
   To structurally guarantee a non-negative definite covariance structure, exclusively the absolute principal diagonals are retained algebraically:
   $$
   \mathbf{R}_{\text{new}} = \operatorname{diag}\Bigl(\bigl|\operatorname{diag}(\mathbf{R}_{\text{new}})\bigr|\Bigr).
   $$
   
4. **Exponentially Smoothed Update for $\mathbf{R}$:**
   $$
   \mathbf{R} \leftarrow \alpha\,\mathbf{R} + (1 - \alpha)\,\mathbf{R}_{\text{new}},
   $$
   utilizing a scaling factor, such as $\alpha = 0.5$, to facilitate a robust, mathematically smoothed adaptation.

#### 5.2. Adaptation Dynamics of the Process Noise Covariance $\mathbf{Q}$

1. **Derivation of an Innovation-Dependent Scaling Coefficient:**  
   First quantifying the Euclidean norm of the innovation construct:
   $$
   \|\mathbf{y}\| = \sqrt{\mathbf{y}^T \mathbf{y}},
   $$
   a dynamically adjusting scaling coefficient is established via:
   $$
   \gamma = \max\left(1, \frac{\|\mathbf{y}\|}{m}\right),
   $$
   where $m$ characterizes the arithmetic total of aggregated measurements.
   
2. **Translational Update Parameterization for $\mathbf{Q}$:**  
   The variance magnitude specific to the underlying process noise is proportionally calibrated:
   $$q_{\text{new}} = \gamma,$$
   consequently yielding:
   $$
   \mathbf{Q}_{\text{new}} = q_{\text{new}} \times  \mathbf{I}
   $$
   
3. **Exponentially Smoothed Formulation for $\mathbf{Q}$:**
   $$
   \mathbf{Q} \leftarrow \beta\,\mathbf{Q} + (1 - \beta)\,\mathbf{Q}_{\text{new}},
   $$
   orchestrating a parameter value like $\beta = 0.5$ to secure a mathematically progressive and oscillation-free transitional flow.

### 6. Conclusion

The Adaptive Extended Kalman Filter (AEKF) operates systematically through two dominant functional architectures:

1. **Prediction Phase:**  
   Utilization of deterministic linear kinematic structures to logically extrapolate imminent future states, executing a parallel parameterization of the error covariance dimensions:
   $$
   \hat{\mathbf{x}}_{k|k-1} = \mathbf{F}\,\hat{\mathbf{x}}_{k-1|k-1} \\
   \quad \mathbf{P}_{k|k-1} = \mathbf{F}\,\mathbf{P}_{k-1|k-1}\,\mathbf{F}^T + \mathbf{Q}.
   $$

2. **Update Phase (Adaptive Correction):**  
   - Analytically compute the innovation discrepancy: $\mathbf{y} = \mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$.  
   - Derive the proportional Kalman gain $\mathbf{K}$ executing correlative algebraic corrections to both state geometries and error covariances.  
   - **Adaptively calibrate** the definitive structural arrays $\mathbf{R}$ and $\mathbf{Q}$, proportionally indexed logically against the derived deterministic magnitude of the systemic innovation and its corresponding deviation parallel to fundamentally projected stochastic uncertainties.

This integrated adaptive functionality fundamentally imbues the architecture with the analytical prowess to parametrically optimize confidence ratios contrasting empirical measurements against deterministic model predictions in rigorous real-time. This dynamic equilibrium decisively augments structural robustness and overall locational accuracy amid severely fluctuating or procedurally degraded physical measuring environments.

---
## NLOS-Aware Adaptive Extended Kalman Filter (NA-AEKF)

### Adaptation Protocol of the Measurement Noise Matrix $\mathbf{R}$ Employing the **is\_NLOS** Vector

For each discrete sensory distance measurement $z_i$ harvested from a geographic anchor, the binary logical descriptor **is\_NLOS[i]** decisively explicitly dictates whether the derived measurement implies degradation via Non-Line-Of-Sight (NLOS) interference (designated logically as 1) or otherwise remains pristine (signified structurally as 0). This discrete characterization grants the mathematical capability to functionally customize the associated statistical variance intrinsic to each measurement within the $\mathbf{R}$ matrix array via the succeeding formulation:

1. **Definition of Fundamental Variance Parameters:**  
   For any pristine measurement captured under pristine Line-Of-Sight (LOS) conditions, an established reference variance governs the function:
   $$
   r_{\text{LOS}}
   $$
   For any degraded measurement subjected to pervasive NLOS conditions, this localized variance is definitively augmented via a static multiplier scalar $\lambda > 1$ directly purposed to systemically deprecate its analytical influence during the state updating framework:
   $$
   r_i =
   \begin{cases}
   r^i_{\text{new}}, & \text{if } is\_NLOS[i] = 0, \\
   \lambda \cdot r^i_{\text{new}}, & \text{if } is\_NLOS[i] = 1.
   \end{cases}
   $$
   For illustrative clarity, operating algebraically if $r_{\text{LOS}} = 0.5$ alongside $\lambda = 10$, subsequently for any degraded NLOS measurement array, $r_i = 5$.

2. **Structural Construction of the Matrix $\mathbf{R}$:**  
   Consequently, the finalized noise covariance error matrix manifests as explicitly diagonalized structurally, parameterized as:
   $$
   \mathbf{R} = \operatorname{diag}(r_1, r_2, \dots, r_m),
   $$
   where $m$ statistically correlates to the entirety of viable measurements accessible dynamically at the specified analytical interval.

3. **Innovation-Dictated Adaptive Update Formulations:**  
   Within the correction mechanism phase, the filtering logic explicitly outputs the definitive innovation construct:
   $$
   \mathbf{y} = \mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1}),
   $$
   coordinately generating an intermediary operational approximation mapping the systemic innovation covariance array:
   $$
   \mathbf{C}_{\text{innov}} = \mathbf{y}\,\mathbf{y}^T.
   $$
   For an arbitrary discrete measurement $i$, the algorithmic engine can proportionally re-calibrate its inherent component $r_i$ uniformly systematically extracting the correlative effects dictated natively by model-derived projected predictive stochastic matrices:
   $$
   r_{i,\text{new}} = \left|\mathbf{C}_{\text{innov}}(i,i) - \left(\mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T\right)(i,i)\right|.
   $$
   Following this derivation, the functional implementation of the explicit logic gating dictated sequentially by the **is\_los** descriptor is deployed:
   $$
   r_{i,\text{new}} =
   \begin{cases}
   r_{i,\text{new}}, & \text{if } is\_NLOS[i] = 0, \\
   \lambda \cdot r_{i,\text{new}}, & \text{if } is\_NLOS[i] = 1.
   \end{cases}
   $$
   
4. **Exponential Smoothing of the Finalized \(\mathbf{R}\) Mapping:**  
   To explicitly counter chaotic procedural oscillations functionally compromising estimation streams, a definitively smooth iterative mathematical adjustment governs the updated $\mathbf{R}$ tensor derivation:
   $$
   \mathbf{R} \leftarrow \alpha\,\mathbf{R} + (1 - \alpha)\,\mathbf{R}_{\text{new}},
   $$
   where the coefficient term $\alpha$ (deploying functionally at values such as $\alpha = 0.5$) precisely orchestrates the proportional integration degree linking trailing historical array values against the latest dynamically calculated estimate tensors.

---

### Analytical Implications Pertaining to the Filtering Infrastructure

- **Deprecation of Erroneous NLOS-Generated Measurement Impact:**  
  By artificially scaling and exacerbating the principal diagonal parameters belonging natively to $\mathbf{R}$ explicit to NLOS-identified measurement streams, the resulting proportional Kalman gain scalar mechanism:
    $$
  \mathbf{K} = \mathbf{P}_{k|k-1}\,\mathbf{H}^T\,\mathbf{S}^{-1}, \quad \text{where } \mathbf{S} = \mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T + \mathbf{R},
  $$
  is systematically analytically depressed for these precise data arrays, dynamically terminating their capacity to tangibly manipulate and skew geometric array state updates.

- **Dramatically Elevated Procedural System Robustness:**  
  Deploying this overarching parameter scheme analytically functionally allows the statistical tracking algorithm to overwhelmingly entrust structural reliability unto statistically proven pristine pristine LOS observations dynamically proven as profoundly robust, actively nullifying the destabilizing impacts universally tied towards severely biased measurement indices generated characteristically amid hostile NLOS physical degradation arrays.

Incorporating these systematic iterative adjustments mathematically transfigures the localization algorithm to become unyieldingly robust specifically within highly constrained indoor environments mathematically, particularly across spatial scenarios characterizing profound operational degradation of characteristic Time-Of-Flight (TOF) UWB measurement infrastructures governed definitively by pervasive NLOS disruption phenomena.


---



## Improve Adaptative Extended Kalman Filter (IAEKF)

### **1. Definition of State Variables**
- **State Vector** (Position and velocity errors):  
  $$
  X(k) = \begin{bmatrix}
  \delta x_x(k) \\
  \delta v_x(k) \\
  \delta x_y(k) \\
  \delta v_y(k)
  \end{bmatrix}
  $$
  - $\delta x_x, \delta x_y$: Estimation errors corresponding to the spatial coordinates $x$ and $y$.  
  - $\delta v_x, \delta v_y$: Estimation errors corresponding to orthogonal velocity components $x$ and $y$.

---

### **2. Process Model (Prediction Phase)**
- **State Transition Matrix** $F(k)$:  
  $$
  F(k) = \begin{bmatrix}
  1 & \Delta t & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & 1 & \Delta t \\
  0 & 0 & 0 & 1
  \end{bmatrix}
  $$
  where $\Delta t$ designates the discrete temporal sampling interval.

- **A Priori State Prediction**:  
  $$
  \hat{X}(k|k-1) = F(k) \cdot X(k-1|k-1)
  $$

- **Predicted Error Covariance Matrix**:  
  $$
  P(k|k-1) = F(k) \cdot P(k-1|k-1) \cdot F(k)^T + Q_{\text{process}}
  $$
  where $Q_{\text{process}}$ defines the intrinsic covariance of the dynamic process noise (structurally parameterized by fundamental noise characteristics intrinsic to the associated IMU or odometric infrastructure).

---

### **3. Measurement Model (Update Phase)**
- **Discrete Measurement Equation**:  
  $$
  Z(k) = H(k) \cdot X(k) + \eta(k), \quad \eta(k) \sim \mathcal{N}(0, Q(k))
  $$
  - $Z(k) = \begin{bmatrix} (d_1(k))^2 - (\hat{d}_1(k))^2 \\ (d_2(k))^2 - (\hat{d}_2(k))^2 \end{bmatrix}$ (configured specifically for dual-anchor operational modes).  
  - **Jacobian Matrix** $H(k)$:  
  $$
  H(k) = \begin{bmatrix}
  2(\hat{x}_x(k) - x_1) & 0 & 2(\hat{x}_y(k) - y_1) & 0 \\
  2(\hat{x}_x(k) - x_2) & 0 & 2(\hat{x}_y(k) - y_2) & 0
  \end{bmatrix}
  $$
  where $(\hat{x}_x, \hat{x}_y)$ analytically denotes the estimated spatial coordinate provided strictly by the auxiliary IMU or odometric hardware, whilst $(x_i, y_i)$ signifies the definitively known spatial anchor coordinates.

---

### **4. Adaptation of the Measurement Noise Covariance Matrix**
- **Measurement Innovation Residual**:  
  $$
  \tilde{Z}(k) = Z(k) - H(k) \cdot \hat{X}(k|k-1)
  $$

- **Empirically Derived Innovation Covariance** (leveraging an adaptive temporal window $M$):  
  $$
  P(k) = \frac{1}{M} \sum_{i=k-M+1}^k \tilde{Z}(i) \tilde{Z}(i)^T
  $$
  - **Magnitude of the Temporal Window $M$**:  
    $$
    M = 
    \begin{cases} 
    1 & \text{if } e(k) \geq \lambda_{\text{max}} \\
    \xi & \text{if } e(k) \leq \lambda_{\text{min}} \\
    \xi \cdot \mu^{(e(k) - \lambda_{\text{min}}) / \alpha} & \text{otherwise}
    \end{cases}
    $$
    
    incorporating the formal definition $e(k) = \tilde{Z}(k)^T \cdot E_{\tilde{Z}}^{-1} \cdot \tilde{Z}(k)$, wherein $E_{\tilde{Z}} = \mathbb{E}[\tilde{Z}(k)\tilde{Z}(k)^T]$ alongside strictly enforced operational thresholds $\lambda_{\text{min}}, \lambda_{\text{max}} \in [0, 1]$.
    
**Comprehensive Explication of Adaptive Measurement Noise Covariance Optimization Parameters**  
The explicit parameters dictating this rigorous algorithmic phase fundamentally facilitate the dynamic modulation of the measurement noise tensor $Q(k)$ exclusively to elevate structural robustness alongside analytical precision of the filtering pipeline:

1. **Analytical Innovation $\tilde{Z}(k)$**:  
   This term rigorously tracks the discrepancy separating verifiable empirical datasets $Z(k)$ from parametrically projected predictive models $H(k) \cdot \hat{X}(k|k-1)$. Mathematically, it isolates instantaneous disturbances generated dynamically by either noise interferences or systemic modeling errors.

2. **Adaptive Tracking Window $M$**:  
   Dictates the aggregate discrete temporal units effectively assimilated to construct the localized covariance mapping $P(k)$. Maintaining an artificially constrained small $M$ factor (e.g., $M=1$) elicits a hyper-reactive adjustment concerning instantaneous radical fluctuations (e.g., gross measurement aberrations or outliers), whilst an equivalently expanded factor (e.g., $M=\xi$) definitively smooths statistical discrepancies and effectively buffers random high-frequency noises.

3. **Normalized Discrepancy Scalar $e(k)$**:  
   Performs strictly as a normalized deterministic indicator tracking systemic divergence distinguishing practical observation streams against mathematically presumed covariance frameworks denoted by $E_{\tilde{Z}}$. Considerably magnified scalar values emphatically underline profound conflicts separating modeled expectations versus practical measurements, systematically prioritizing dynamically aggressive adaptive recalculations.

4. **Algebraic Saturation Thresholds $\lambda_{\text{min}}, \lambda_{\text{max}}$**:  
   Rigidly defines absolute operational constraints circumscribing allowable behaviors per $e(k)$. Upon a scenario wherein $e(k)$ surges past $\lambda_{\text{max}}$, the system forcibly restricts $M=1$ (yielding instantaneous adaptive responses). Conversely, operating below $\lambda_{\text{min}}$, the algorithm imposes $M=\xi$ (prioritizing mathematical stability).

5. **Structural Scaling Constants $\xi, \mu, \alpha$**:  
   - $\xi$: Formal maximum temporal span characteristic of the operational tracking window (governing the foundational trade-off resolving mathematical stability versus parametric complexity).  
   - $\mu$: Systematic convergence gradient mapping $M$ (exerting definitive control upon the transitional velocity governing localized physical adaptations).  
   - $\alpha$: Normalizing mathematical scaling parameter scaling the dynamic exponent inherent to $M$, formally tethered structurally to the established fundamental frequency characterizing prevailing operational noise frameworks.

6. **Cumulative Forgetting Coefficient $\tau(k)$ alongside Base Factor $\tau$**:  
   - $\tau$ (constrained fundamentally within $[0.9, 0.99]$): Deterministically limits the proportional weighting systematically allocated towards historical distributions natively inherent within the ongoing update cycles characterizing $Q(k)$. As $\tau$ asymptotically approaches 1, the computational algorithm "forgets" archaic temporal estimation matrices perceptibly slower.  
   - $\tau(k)$: Autonomously adjusts proportional indexing effectively managing transient smoothing frameworks connecting real-time generated localized covariance data $P(k)$ harmoniously onto accumulated historical indices recorded across $Q(k-1)$, fundamentally asserting guaranteed mathematical fluidity absent disruptive phase shocks.

**Cumulative Systemic Impact**: Sequentially leveraging these parametric formulations profoundly empowers the generalized filtration algorithmic architecture to intelligently execute the following sequence:  
- Robustly recognize alongside systematically diminish the cascading destructive consequences characteristic of unmitigated gross data errors leveraging dynamic responses driven by $M$ aligned with $e(k)$.  
- Iteratively restructure $Q(k)$ throughout real-time continuous operation precisely to objectively mirror current intrinsic statistical certitudes specific to dynamically evolving functional UWB hardware distance arrays.  
- Systemically resolve the foundational engineering balance differentiating parametric reactivity (small $M$) fundamentally opposing mathematical structural continuity (large $M$) consistently matching shifting operational noise patterns continuously.

**Iterative Mathematical Adaptation Protocol Dictating $Q(k)$**:  
  $$
  Q(k) = (1 - \tau(k)) \cdot Q(k-1) + \tau(k) \cdot \left( P(k) - H(k) \cdot P(k|k-1) \cdot H(k)^T \right)
  $$
  where $\tau(k) =  \frac{1 - \tau}{1 - \tau(k+1)}$ (delineating purely an exponential forgetting parameter tracking strictly $\tau \in [0.9, 0.99]$).

---

### **5. Extended Kalman Filter Computational Equations**
- **Kalman Gain Parameter**:  
  $$
  K(k) = P(k|k-1) \cdot H(k)^T \cdot \left( H(k) \cdot P(k|k-1) \cdot H(k)^T + Q(k) \right)^{-1}
  $$

- **Post-Priori State Error Matrix Correction**:  
  $$
  X(k|k) = \hat{X}(k|k-1) + K(k) \cdot \tilde{Z}(k)
  $$

- **Iterative Covariance Architecture Recalibration**:  
  $$
  P(k|k) = (I - K(k) \cdot H(k)) \cdot P(k|k-1)
  $$

---

### **6. Operational System Inputs & Definitive Outputs**
- **Empirical Measurement Inputs**:  
  - Aggregated UWB measurement data: $d_i(k)$ (dynamically tracking either 1 or 2 anchors).  
  - Pre-established Anchor Coordinates: $(x_i, y_i)$.  
  - Fixed Pre-Configuration Adaptive Scalar Settings: $\mu, \alpha, \xi, \lambda_{\text{min}}, \lambda_{\text{max}}, \tau$.

- **Deterministic Process Outputs**:  
  - Parametrically Calibrated Post-Priori Error Coordinates: $X(k|k)$.  
  - Dynamically Extrapolated Definitive Robot Positioning Mapping:  
    $$
    x_x(k) = \hat{x}_x(k) - \delta x_x(k|k), \quad x_y(k) = \hat{x}_y(k) - \delta x_y(k|k).
    $$  
  - Adaptively Smoothed Measurement Noise Covariance Matrix $Q(k)$ alongside structurally finalized Error Array $P(k|k)$.

---

### **Functional Architectural Flowchart**
1. **Prediction Phase Execution**:  
   - Effectively utilizes the parametric matrices $F(k)$ alongside associated stochastic error array $Q_{\text{process}}$ to compute continuous analytic predictions corresponding specifically to $\hat{X}(k|k-1)$.  
2. **Adaptive Modulations applied to $Q(k)$**:  
   - Functionally calibrates $M$ natively alongside the covariance tracking element derived explicitly per systemic innovation protocols, yielding $P(k)$.  
3. **Execution Update Protocol Phase**:  
   - Iteratively implements a final a posteriori parameter correction strictly using optimized configurations mapping the gain factor $K(k)$, innovation metrics dictating the deviation parameter specific to $\tilde{Z}(k)$, aligned perfectly symmetrically incorporating recalibrated arrays structurally defining $Q(k)$.

# References:
> - [1] Taha, M., Berder, O., Courtay, A., & Le Gentil, M. (2025, October). NA-AEKF: A NLOS-Aware Adaptive Extended Kalman Filter for Robust Indoor Localization. In 2025 21th International Conference on Wireless and Mobile Computing, Networking and Communications (WiMob) (pp. 1-6). IEEE. [https://ieeexplore.ieee.org/document/11257468](https://ieeexplore.ieee.org/document/11257468)
> - [2] Momtaz, M. R., Abolhasan, M., Lipman, J., & Ni, W. (2023). Adaptive Extended Kalman Filter Position Estimation Based on Ultra-Wideband Active-Passive Ranging Protocol. *Sensors*, *23*(5), 2669. https://doi.org/10.3390/s23052669
> - [3] Li, J., Wang, S., Hao, J., Ma, B., & Chu, H. K. (2024). UVIO: Adaptive Kalman Filtering UWB-Aided Visual-Inertial SLAM System for Complex Indoor Environments. *Remote Sensing*, *16*(17), 3245.
> - [4] Ahmed, M., Mosavi, A., Sardroudi, A., & Varkonyi-Koczy, A. R. (2021). Indoor Localization by Kalman Filter based Combining of UWB-Positioning and PDR. *Sensors*, *21*(5), 1745. https://doi.org/10.3390/s21051745
