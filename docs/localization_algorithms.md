# Trilateration

Dans le contexte de la détermination de la position d’un tag dans l’espace $(x, y, z)$, la trilatération consiste à utiliser les distances mesurées depuis ce tag vers plusieurs ancres dont les positions sont connues pour calculer ses coordonnées. Chaque ancre fournit une équation basée sur une sphère.

---

### 1. Équations des Sphères

Pour chaque ancre $i$ dont la position est donnée par $p_i = (x_i, y_i, z_i)$ et pour laquelle la distance mesurée au tag est $r_i$, on a l’équation d’une sphère :
$$
(x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2 = r_i^2.
$$
Pour déterminer une position unique en 3D, il est nécessaire d’avoir au moins quatre ancres (quatre équations).


### 2. Linéarisation du Système par Soustraction

Pour obtenir un système linéaire, on soustrait l’équation de la première ancre de celles des autres ancres. Par exemple, pour l’ancre 2 par rapport à l’ancre 1 :

$$
\begin{aligned}
&(x - x_2)^2 + (y - y_2)^2 + (z - z_2)^2 \\
&\quad - \left[(x - x_1)^2 + (y - y_1)^2 + (z - z_1)^2\right] = r_2^2 - r_1^2.
\end{aligned}
$$

En développant et en annulant les termes quadratiques communs, on obtient :

$$
-2x(x_2 - x_1) - 2y(y_2 - y_1) - 2z(z_2 - z_1) + \big[(x_2^2 - x_1^2) + (y_2^2 - y_1^2) + (z_2^2 - z_1^2)\big] = r_2^2 - r_1^2.
$$

En multipliant par \(-1\), on peut écrire :

$$
2x(x_2 - x_1) + 2y(y_2 - y_1) + 2z(z_2 - z_1) = r_1^2 - r_2^2 + (x_2^2 - x_1^2) + (y_2^2 - y_1^2) + (z_2^2 - z_1^2).
$$

On réalise le même procédé pour les ancres 3 et 4.


### 3. Formulation Matricielle du Système Linéaire

Les trois équations obtenues (pour les ancres 2, 3 et 4 par rapport à la première) se présentent sous la forme :
$$
A \begin{pmatrix} x \\ y \\ z \end{pmatrix} = b,
$$
avec :
$$
A = 2 \begin{pmatrix}
x_2 - x_1 & y_2 - y_1 & z_2 - z_1 \\
x_3 - x_1 & y_3 - y_1 & z_3 - z_1 \\
x_4 - x_1 & y_4 - y_1 & z_4 - z_1 
\end{pmatrix},
$$
et
$$
b = \begin{pmatrix}
r_1^2 - r_2^2 + (x_2^2 - x_1^2) + (y_2^2 - y_1^2) + (z_2^2 - z_1^2) \\
r_1^2 - r_3^2 + (x_3^2 - x_1^2) + (y_3^2 - y_1^2) + (z_3^2 - z_1^2) \\
r_1^2 - r_4^2 + (x_4^2 - x_1^2) + (y_4^2 - y_1^2) + (z_4^2 - z_1^2)
\end{pmatrix}.
$$

La solution du système fournit les coordonnées $(x, y, z)$ du tag :

$$
\begin{pmatrix} x \\ y \\ z \end{pmatrix} = A^{-1}b,
$$
ou, dans un cas pratique, on résoudra numériquement ce système (par exemple, via une méthode des moindres carrés en cas d’ancres supplémentaires ou de bruit dans les mesures).

### 4. Remarques Pratiques

- **Nombre d’ancres :** En 3D, un minimum de quatre ancres est nécessaire pour obtenir une solution unique. Si plus d’ancres sont disponibles, la solution peut être optimisée par des méthodes de régression (moindres carrés) afin de minimiser l’erreur.
- **Cas de matrice singulière :** Si les positions des ancres sont telles que la matrice \(A\) devient singulière ou mal conditionnée (par exemple, si elles sont alignées ou presque coplanaires), la solution sera imprécise ou non déterminable.

---

# Extended Kalman Filter (EKF) 

L’Extended Kalman Filter (EKF) est une méthode récursive qui permet d’estimer l’état d’un système dynamique non linéaire. Dans notre cas, l’objectif est d’estimer la position et la vitesse d’un tag évoluant dans un plan (2D) en fusionnant :

- Un **modèle dynamique** (pour la prédiction de l’état futur), et  
- Un **modèle de mesure** non linéaire (pour corriger cette prédiction à partir de mesures réelles).

### 1. Vecteur d’État et Notions de Base

**Vecteur d’état ($\mathbf{x}$) :**  
C’est une représentation des variables que nous souhaitons estimer. Pour un tag en 2D, nous définissons :

$$
\mathbf{x} = \begin{pmatrix} x \\ y \\ v_x \\ v_y \end{pmatrix},
$$

- $x,\, y$ : coordonnées de position,  
- $v_x,\, v_y$ : composantes de la vitesse.

**Covariance de l’état ($\mathbf{P}$) :**  
La matrice $\mathbf{P}$ quantifie l’incertitude (ou l’erreur) associée à l’estimation de l’état.

**Bruitage :**  
- **Bruit de processus ($\mathbf{w}_k$) :** incertitude dans l’évolution du système.  
- **Bruit de mesure ($v_i$) :** incertitude dans les mesures effectuées par les capteurs.  
Ces bruits sont souvent modélisés comme des variables aléatoires gaussiennes.

### 2. Modèle Dynamique (Prédiction)

Le modèle dynamique décrit comment l’état évolue dans le temps.

#### 2.1. Équation d’État

$$
\mathbf{x}_{k+1} = \mathbf{F}\,\mathbf{x}_k + \mathbf{w}_k,
$$

- $\mathbf{x}_k$ : état à l’instant $k$.  
- $\mathbf{w}_k$ : bruit de processus (incertitude dans la dynamique).

#### 2.2. Matrice de Transition d’État ($\mathbf{F}$)

Pour un déplacement à vitesse constante et avec un intervalle de temps $dt$, on définit :

$$
\mathbf{F} = \begin{pmatrix}
1 & 0 & dt & 0 \\
0 & 1 & 0 & dt \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{pmatrix}.
$$

Cette matrice traduit que la nouvelle position dépend de l’ancienne position et de la vitesse.

#### 2.3. Bruit de Processus et sa Covariance (\(\mathbf{Q}\))

La covariance du bruit de processus est donnée par :

$$
\mathbf{Q} = q\,\mathbf{I}_4,
$$

où $q$ est une constante (ex. $q=0.1$) et $\mathbf{I}_4$ est la matrice identité 4×4. Cette matrice modélise l’incertitude dans la prédiction.

### 2.4. Étape de Prédiction

La prédiction de l’état et de la covariance est effectuée par :

$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F}\,\hat{\mathbf{x}}_{k-1|k-1},
$$
$$
\mathbf{P}_{k|k-1} = \mathbf{F}\,\mathbf{P}_{k-1|k-1}\,\mathbf{F}^T + \mathbf{Q}.
$$

- $\hat{\mathbf{x}}_{k|k-1}$ : estimation prédite de l’état à l’instant $k$.  
- $\mathbf{P}_{k|k-1}$ : covariance prédite associée à cette estimation.


### 3. Modèle de Mesure et Linéarisation (Mise à Jour)

Les mesures proviennent d’ancres dont les positions sont connues. Chaque mesure est la distance entre le tag et une ancre.

#### 3.1. Équation de Mesure

Pour une ancre $i$ située en $p_i = (x_i, y_i)$, la distance mesurée est modélisée par :

$$
r_i = h_i(\mathbf{x}) + v_i = \sqrt{(x - x_i)^2 + (y - y_i)^2} + v_i,
$$

où $v_i$ est le bruit de mesure, modélisé par une variable gaussienne.

### 3.2. Bruit de Mesure et sa Covariance ($\mathbf{R}$)

La matrice $\mathbf{R}$ est la covariance associée au bruit de mesure. Par exemple, pour $m$ mesures, on peut poser :

$$
\mathbf{R} = \sigma_r^2\,\mathbf{I}_m,
$$

où $\sigma_r^2$ est la variance du bruit de mesure.

#### 3.3. Linéarisation : La Jacobienne ($\mathbf{H}$)

Comme la fonction de mesure $h_i(\mathbf{x})$ est non linéaire, on la linéarise autour de l’estimation courante $\hat{\mathbf{x}}$ en calculant la matrice Jacobienne. Pour chaque ancre $i$ :

$$
\frac{\partial h_i}{\partial x} = \frac{x - x_i}{d_i}, \quad \frac{\partial h_i}{\partial y} = \frac{y - y_i}{d_i},
$$

avec

$$
d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2}.
$$

Les dérivées par rapport aux vitesses sont nulles (puisque la mesure dépend uniquement de la position) :

$$
\frac{\partial h_i}{\partial v_x} = 0, \quad \frac{\partial h_i}{\partial v_y} = 0.
$$

Pour $m$ mesures, la matrice Jacobienne s’exprime ainsi :

$$
\mathbf{H} = \begin{pmatrix}
\frac{x - x_1}{d_1} & \frac{y - y_1}{d_1} & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots \\
\frac{x - x_m}{d_m} & \frac{y - y_m}{d_m} & 0 & 0 \\
\end{pmatrix}.
$$

La prédiction des mesures (le vecteur attendu) est donnée par :

$$
\mathbf{h}(\hat{\mathbf{x}}) = \begin{pmatrix}
\sqrt{(x - x_1)^2 + (y - y_1)^2} \\
\vdots \\
\sqrt{(x - x_m)^2 + (y - y_m)^2} \\
\end{pmatrix}.
$$


### 4. Mise à Jour de l’Estimation (Correction)

#### 4.1. Calcul du Gain de Kalman

Le gain de Kalman $\mathbf{K}$ détermine la pondération des mesures par rapport à la prédiction et se calcule ainsi :

$$
\mathbf{S} = \mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T + \mathbf{R},
$$
$$
\mathbf{K} = \mathbf{P}_{k|k-1}\,\mathbf{H}^T\,\mathbf{S}^{-1}.
$$

- $\mathbf{S}$ est la covariance innovation (ou covariance de la différence entre mesure prédite et réelle).

#### 4.2. Mise à Jour de l’État et de la Covariance

La correction de l’estimation se fait par :

$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}\,\left(\mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})\right),
$$
$$
\mathbf{P}_{k|k} = \left(\mathbf{I} - \mathbf{K}\,\mathbf{H}\right)\,\mathbf{P}_{k|k-1}.
$$

- $\mathbf{z}$ est le vecteur des mesures réelles.  
- $\mathbf{I}$ est la matrice identité.  
- La différence $\mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$ est appelée l’**innovation**.

### 5. Avantages de l’EKF par Rapport à la Trilateration

- **Intégration du Dynamisme :**  
  L’EKF prend en compte l’évolution temporelle du système grâce à son modèle dynamique. Cela permet de prédire la position entre deux mesures, ce qui est particulièrement utile pour suivre un tag en mouvement. La trilatération, en revanche, donne une position ponctuelle sans historique temporel.

- **Gestion du Bruit et des Incertitudes :**  
  L’EKF intègre explicitement les incertitudes (bruit de processus et bruit de mesure) via les matrices $\mathbf{Q}$ et $\mathbf{R}$, fournissant ainsi une estimation plus robuste face aux perturbations.

- **Estimation Complète :**  
  En plus de la position, l’EKF estime la vitesse du tag, offrant une information dynamique supplémentaire pour prédire les mouvements futurs.

- **Mise à Jour Itérative :**  
  L’EKF ajuste continuellement son estimation à mesure que de nouvelles mesures sont disponibles, améliorant ainsi la précision au fil du temps, contrairement à la trilatération qui ne tient compte que de la dernière mesure disponible.

---

### Conclusion

Ce modèle mathématique de l’Extended Kalman Filter en 2D présente une approche structurée en deux phases principales : la prédiction de l’état à l’aide d’un modèle dynamique et la correction basée sur des mesures non linéaires linéarisées. Chaque terme du modèle est défini afin d’assurer une compréhension claire du processus de filtrage. Cette méthode offre une solution dynamique, robuste et plus complète que la simple trilatération pour la localisation d’un tag évoluant dans le temps.

---

## Adaptive Extended Kalman Filter (AEKF) 

### 1. Vecteur d'État et Notions de Base

Les définitions du vecteur d'état, de la matrice de covariance et des bruits sont identiques à celles présentées dans la section précédente sur le filtre de Kalman étendu (EKF).

### 2. Modèle Dynamique (Prédiction)

Le modèle dynamique et l'équation d'état sont les mêmes que pour l'EKF standard, avec la matrice de transition $\mathbf{F}$ identique.

### 3. Modèle de Mesure Non Linéaire

Le modèle de mesure et sa linéarisation via la matrice Jacobienne $\mathbf{H}$ restent inchangés par rapport à l'EKF.

### 4. Étapes de l'Algorithme

Les étapes de prédiction et de correction suivent exactement le même processus que l'EKF standard décrit précédemment, avec les mêmes équations pour le gain de Kalman et la mise à jour de l'état et de la covariance.

La principale différence de l'AEKF réside dans l'ajout du mécanisme adaptatif décrit ci-après.


### 5. Mécanisme Adaptatif pour $\mathbf{R}$ et $\mathbf{Q}$

L’algorithme ajuste dynamiquement les matrices $\mathbf{R}$ et $\mathbf{Q}$ en fonction des innovations observées.

#### 5.1. Adaptation de la Matrice de Bruit de Mesure $\mathbf{R}$

1. **Calcul de l’innovation :**
   $$
   \mathbf{y} = \mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1}).
   $$
   
2. **Innovation covariance estimée :**
   $$
   \mathbf{C}_{\text{innov}} = \mathbf{y}\,\mathbf{y}^T.
   $$
   
3. **Estimation provisoire de $\mathbf{R}$ :**  
   En retirant l’effet de l’incertitude projetée :
   $$
   \mathbf{R}_{\text{new}} = \mathbf{C}_{\text{innov}} - \mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T.
   $$
   Pour assurer que la covariance reste positive, on ne retient que la diagonale (avec valeurs absolues) :
   $$
   \mathbf{R}_{\text{new}} = \operatorname{diag}\Bigl(\bigl|\operatorname{diag}(\mathbf{R}_{\text{new}})\bigr|\Bigr).
   $$
   
4. **Mise à jour lissée de $\mathbf{R}$ :**
   $$
   \mathbf{R} \leftarrow \alpha\,\mathbf{R} + (1 - \alpha)\,\mathbf{R}_{\text{new}},
   $$
   avec par exemple $\alpha = 0.5$ pour une mise à jour équilibrée.

#### 5.2. Adaptation de la Matrice de Bruit de Processus $\mathbf{Q}$

1. **Calcul d’un facteur de mise à l’échelle basé sur l’innovation :**  
   On mesure la norme de l’innovation :
   $$
   \|\mathbf{y}\| = \sqrt{\mathbf{y}^T \mathbf{y}},
   $$
   puis on définit un facteur de mise à l’échelle :
   $$
   \gamma = \max\left(1, \frac{\|\mathbf{y}\|}{m}\right),
   $$
   où $m$ est le nombre de mesures.
   
2. **Mise à jour de $\mathbf{Q}$:**  
   On ajuste la variance du bruit de processus :
  $q_{\text{new}} = \gamma,$
   et donc :
   $$
   \mathbf{Q}_{\text{new}} = q_{\text{new}} \times  \mathbf{I}
   $$
   
3. **Mise à jour lissée de $\mathbf{Q}$ :**
   $$
   \mathbf{Q} \leftarrow \beta\,\mathbf{Q} + (1 - \beta)\,\mathbf{Q}_{\text{new}},
   $$
   avec par exemple $\beta = 0.5$ pour assurer une transition en douceur.

### 6. Conclusion

L’algorithme Adaptive Extended Kalman Filter (AEKF) procède en deux grandes phases :

1. **Prédiction :**  
   Utilisation du modèle dynamique linéaire pour estimer l’état futur et mettre à jour la covariance :
   $$
   \hat{\mathbf{x}}_{k|k-1} = \mathbf{F}\,\hat{\mathbf{x}}_{k-1|k-1} \\
   \quad \mathbf{P}_{k|k-1} = \mathbf{F}\,\mathbf{P}_{k-1|k-1}\,\mathbf{F}^T + \mathbf{Q}.
   $$

2. **Mise à Jour (Correction) avec Adaptation :**  
   - Calcul de l’innovation : $\mathbf{y} = \mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1})$.  
   - Calcul du gain de Kalman $\mathbf{K}$ et mise à jour de l’état et de la covariance.  
   - **Adaptation** des matrices $\mathbf{R}$ et $\mathbf{Q}$ en fonction de la magnitude de l’innovation et de son écart par rapport aux incertitudes projetées.

Ce mécanisme adaptatif permet d’ajuster en temps réel la confiance accordée aux mesures et au modèle dynamique, améliorant ainsi la robustesse et la précision de la localisation dans des environnements où les conditions de mesure varient.

---
## NLOS-Aware Adaptive Extended Kalman Filter (LA-AEKF)

### Adaptation de la Matrice de Bruit de Mesure $\mathbf{R}$ en tenant compte du vecteur **is\_NLOS**

Pour chaque mesure $z_i$ provenant d’une ancre, le drapeau **is\_NLOS[i]** permet d’indiquer si la mesure est affectée par un problème de NLOS (1) ou non (0). On peut ainsi adapter individuellement la variance associée à chaque mesure dans $\mathbf{R}$ de la manière suivante :

1. **Définition des variances de base :**  
   Pour une mesure en condition LOS (ligne de vue dégagée), on définit une variance de référence :
   $$
   r_{\text{LOS}}
   $$
   Pour une mesure en condition NLOS, on augmente cette variance par un facteur $\lambda > 1$ afin de réduire son influence lors de la mise à jour :
   $$
   r_i =
   \begin{cases}
   r^i_{\text{new}}, & \text{si } is\_NLOS[i] = 0, \\
   \lambda \cdot r^i_{\text{new}}, & \text{si } is\_NLOS[i] = 1.
   \end{cases}
   $$
   Par exemple, si $r_{\text{LOS}} = 0.5$ et $\lambda = 10$, alors pour une mesure NLOS, $r_i = 5$.

2. **Construction de la matrice $\mathbf{R}$ :**  
   La matrice de covariance de bruit de mesure devient alors diagonale, avec :
   $$
   \mathbf{R} = \operatorname{diag}(r_1, r_2, \dots, r_m),
   $$
   où $m$ est le nombre total de mesures disponibles à l’instant considéré.

3. **Mise à jour adaptative basée sur l’innovation :**  
   Lors de la correction, le filtre calcule l’innovation :
   $$
   \mathbf{y} = \mathbf{z} - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1}),
   $$
   et une estimation provisoire de la covariance de l’innovation :
   $$
   \mathbf{C}_{\text{innov}} = \mathbf{y}\,\mathbf{y}^T.
   $$
   Pour chaque mesure $i$, on peut ajuster la composante $r_i$ en retirant l’effet de l’incertitude projetée par le modèle :
   $$
   r_{i,\text{new}} = \left|\mathbf{C}_{\text{innov}}(i,i) - \left(\mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T\right)(i,i)\right|.
   $$
   Ensuite, en appliquant la pondération par le drapeau **is\_los** :
   $$
   r_{i,\text{new}} =
   \begin{cases}
   r_{i,\text{new}}, & \text{si } is\_NLOS[i] = 0, \\
   \lambda \cdot r_{i,\text{new}}, & \text{si } is\_NLOSi] = 1.
   \end{cases}
   $$
   
4. **Mise à jour lissée de \(\mathbf{R}\) :**  
   Pour éviter des variations brusques, on peut effectuer une mise à jour lissée de la matrice \(\mathbf{R}\) :
   $$
   \mathbf{R} \leftarrow \alpha\,\mathbf{R} + (1 - \alpha)\,\mathbf{R}_{\text{new}},
   $$
   où $\alpha$ (par exemple, $\alpha = 0.5$) détermine le degré de lissage entre l’ancienne valeur et la nouvelle estimation.

---

### Effets sur le Filtrage

- **Réduction de l’influence des mesures NLOS :**  
  En augmentant les valeurs diagonales de $\mathbf{R}$ pour les mesures NLOS, le gain de Kalman :
    $$
  \mathbf{K} = \mathbf{P}_{k|k-1}\,\mathbf{H}^T\,\mathbf{S}^{-1}, \quad \text{avec } \mathbf{S} = \mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T + \mathbf{R},
  $$
  sera réduit pour ces mesures, ce qui diminue leur impact sur la mise à jour de l’état.

- **Robustesse améliorée :**  
  Cette stratégie permet d’accorder plus de confiance aux mesures en LOS, considérées comme plus fiables, et d’atténuer l’influence des mesures potentiellement biaisées par les conditions NLOS.

En appliquant ces ajustements, l’algorithme devient plus robuste en environnement intérieur, particulièrement dans les scénarios où le phénomène NLOS peut fortement dégrader la qualité des mesures UWB basées sur le TOF.



## Improve adaptative extended kalman filter (IAEKF)

### **1. Définition des Variables d'État**
- **État** (erreurs de position et vitesse) :  
  $$
  X(k) = \begin{bmatrix}
  \delta x_x(k) \\
  \delta v_x(k) \\
  \delta x_y(k) \\
  \delta v_y(k)
  \end{bmatrix}
  $$
  - $\delta x_x, \delta x_y$ : Erreurs de position en $x$ et $y$.  
  - $\delta v_x, \delta v_y$ : Erreurs de vitesse en $x$ et $y$.

---

### **2. Modèle de Processus (Prédiction)**
- **Matrice de transition d'état** $F(k)$ :  
  $$
  F(k) = \begin{bmatrix}
  1 & \Delta t & 0 & 0 \\
  0 & 1 & 0 & 0 \\
  0 & 0 & 1 & \Delta t \\
  0 & 0 & 0 & 1
  \end{bmatrix}
  $$
  où $\Delta t$ est le pas de temps.

- **Prédiction de l'état** :  
  $$
  \hat{X}(k|k-1) = F(k) \cdot X(k-1|k-1)
  $$

- **Covariance d'erreur prédite** :  
  $$
  P(k|k-1) = F(k) \cdot P(k-1|k-1) \cdot F(k)^T + Q_{\text{process}}
  $$
  où $Q_{\text{process}}$ est la covariance du bruit de processus (définie par les caractéristiques de l'IMU/odomètre).

---

### **3. Modèle de Mesure (Mise à Jour)**
- **Équation de mesure** :  
  $$
  Z(k) = H(k) \cdot X(k) + \eta(k), \quad \eta(k) \sim \mathcal{N}(0, Q(k))
  $$
  - $Z(k) = \begin{bmatrix} (d_1(k))^2 - (\hat{d}_1(k))^2 \\ (d_2(k))^2 - (\hat{d}_2(k))^2 \end{bmatrix}$ (pour 2 ancres).  
  - **Jacobienne** $H(k)$ :  
  $$
  H(k) = \begin{bmatrix}
  2(\hat{x}_x(k) - x_1) & 0 & 2(\hat{x}_y(k) - y_1) & 0 \\
  2(\hat{x}_x(k) - x_2) & 0 & 2(\hat{x}_y(k) - y_2) & 0
  \end{bmatrix}
  $$
  où $(\hat{x}_x, \hat{x}_y)$ est la position estimée par l'IMU/odomètre, et $(x_i, y_i)$ sont les positions des ancres.

---

### **4. Adaptation de la Covariance du Bruit de Mesure**
- **Innovation** :  
  $$
  \tilde{Z}(k) = Z(k) - H(k) \cdot \hat{X}(k|k-1)
  $$

- **Covariance de l'innovation** (fenêtre adaptative $M$) :  
  $$
  P(k) = \frac{1}{M} \sum_{i=k-M+1}^k \tilde{Z}(i) \tilde{Z}(i)^T
  $$
  - **Taille de fenêtre $M$** :  
    $$
    M = 
    \begin{cases} 
    1 & \text{si } e(k) \geq \lambda_{\text{max}} \\
    \xi & \text{si } e(k) \leq \lambda_{\text{min}} \\
    \xi \cdot \mu^{(e(k) - \lambda_{\text{min}}) / \alpha} & \text{sinon}
    \end{cases}
    $$
    
    où $e(k) = \tilde{Z}(k)^T \cdot E_{\tilde{Z}}^{-1} \cdot \tilde{Z}(k)$, avec $E_{\tilde{Z}} = \mathbb{E}[\tilde{Z}(k)\tilde{Z}(k)^T] $ et $\lambda_{\text{min}}, \lambda_{\text{max}} \in [0, 1]$.
    
**Explication des Paramètres d'Adaptation de la Covariance du Bruit de Mesure**  
Les paramètres clés de cette étape permettent d'ajuster dynamiquement la covariance du bruit de mesure $Q(k)$ pour améliorer la robustesse et la précision du filtre :  

1. **Innovation $\tilde{Z}(k)$** :  
   Représente l'écart entre les mesures réelles $Z(k)$ et les prédictions $H(k) \cdot \hat{X}(k|k-1)$. Elle reflète les perturbations instantanées dues au bruit ou aux erreurs de modèle.  

2. **Fenêtre adaptative $M$** :  
   Détermine le nombre de pas de temps utilisés pour calculer $P(k)$. Un $M$ petit (ex. $M=1$) réagit rapidement aux changements brusques (ex. erreurs grossières), tandis qu'un $M$ grand (ex. $M=\xi$) lisse les variations et réduit le bruit.  

3. **Résiduel $e(k)$** :  
   Mesure normalisée de la divergence de l'innovation par rapport à sa covariance attendue $E_{\tilde{Z}}$. Une valeur élevée indique une incohérence entre les mesures et le modèle, déclenchant une adaptation plus agressive.  

4. **Seuils $\lambda_{\text{min}}, \lambda_{\text{max}}$** :  
   Définissent des limites pour $e(k)$. Si $e(k)$ dépasse $\lambda_{\text{max}}$, $M=1$ (réponse immédiate). Si $e(k)$ est inférieur à $\lambda_{\text{min}}$, $M=\xi$ (stabilité).  

5. **Paramètres $\xi, \mu, \alpha$** :  
   - $\xi$ : Taille maximale de la fenêtre (équilibre stabilité/complexité).  
   - $\mu$ : Taux de convergence de $M$ (contrôle la vitesse d'adaptation).  
   - $\alpha$ : Facteur d'échelle pour normaliser l'exposant dans $M$, lié à la fréquence du bruit.  

6. **Coefficient d'oubli $\tau(k)$ et facteur $\tau$** :  
   - $\tau$ (0.9–0.99) : Pondère l'historique de $Q(k)$. Plus $\tau$ est proche de 1, plus le filtre "oublie" lentement les anciennes estimations.  
   - $\tau(k)$ : Ajuste dynamiquement le poids entre la covariance actuelle $P(k)$ et l'historique $Q(k-1)$, garantissant une transition fluide.  

**Impact Global** : Ces paramètres permettent au filtre de :  
- Détecter et atténuer les erreurs grossières via $M$ et $e(k)$.  
- Adapter $Q(k)$ en temps réel pour refléter la confiance dans les mesures UWB.  
- Équilibrer réactivité (petit $M$) et stabilité (grand $M$) selon les conditions de bruit.

**Mise à jour adaptative de $Q(k)$** :  
  $$
  Q(k) = (1 - \tau(k)) \cdot Q(k-1) + \tau(k) \cdot \left( P(k) - H(k) \cdot P(k|k-1) \cdot H(k)^T \right)
  $$
  où $\tau(k) =  \frac{1 - \tau}{1 - \tau(k+1)}$ (facteur d'oubli exponentiel, $\tau \in [0.9, 0.99]$).

---

### **5. Équations de Mise à Jour du Filtre de Kalman**
- **Gain de Kalman** :  
  $$
  K(k) = P(k|k-1) \cdot H(k)^T \cdot \left( H(k) \cdot P(k|k-1) \cdot H(k)^T + Q(k) \right)^{-1}
  $$

- **Mise à jour de l'état** :  
  $$
  X(k|k) = \hat{X}(k|k-1) + K(k) \cdot \tilde{Z}(k)
  $$

- **Mise à jour de la covariance** :  
  $$
  P(k|k) = (I - K(k) \cdot H(k)) \cdot P(k|k-1)
  $$

---

### **6. Entrées/Sorties**
- **Entrées** :  
  - Mesures UWB : $d_i(k)$ (1 ou 2 ancres).  
  - Positions des ancres : $(x_i, y_i)$.  
  - Paramètres adaptatifs : $\mu, \alpha, \xi, \lambda_{\text{min}}, \lambda_{\text{max}}, \tau$.

- **Sorties** :  
  - Erreurs d'état corrigées : $X(k|k)$.  
  - Position finale du robot :  
    $$
    x_x(k) = \hat{x}_x(k) - \delta x_x(k|k), \quad x_y(k) = \hat{x}_y(k) - \delta x_y(k|k).
    $$  
  - Covariance adaptative $Q(k)$ et matrice $P(k|k)$.

---

### **Schéma Synoptique**
1. **Prédiction** :  
   - Utilise $F(k)$ et $Q_{\text{process}}$ pour prédire $\hat{X}(k|k-1)$.  
2. **Adaptation de $Q(k)$** :  
   - Calcule $M$ et $P(k)$ via l'innovation.  
3. **Mise à jour** :  
   - Corrige l'état avec $K(k)$, $\tilde{Z}(k)$, et $Q(k)$.



# Amélioration du Filtre LA-AEKF par l'Intégration des Mesures IMU  
## IMU-Assisted NLOS-Aware EKF (IA-NAEKF)

Ce rapport présente le filtre LA-AEKF (NLOS-Aware Adaptive Extended Kalman Filter) en détaillant ses avantages pour la localisation en environnement non-ligne de visée (NLOS). Il expose également la problématique liée à la classification des mesures LOS/NLOS — qui, dans certains cas, n'atteint qu'une précision d'environ 90 % —, et décrit l'idée innovante d'intégrer les mesures IMU pour pallier ces incertitudes. La dernière partie du document détaille le modèle mathématique complet de cette approche hybride.


## 1. Problématique Rencontrée

Dans des scénarios réels, la classification des mesures en conditions LOS et NLOS repose souvent sur des algorithmes de détection qui, même s'ils atteignent en général une précision d'environ 90 %, laissent subsister une part d'incertitude non négligeable. Ce taux d'erreur peut conduire à :

- **Des erreurs dans l'estimation de la variance :**  
  Une classification erronée (par exemple, considérer une mesure NLOS comme LOS) entraîne une mauvaise pondération lors de la fusion des mesures, ce qui dégrade la précision de la localisation.
  
- **Une propagation de l'erreur dans l'état estimé :**  
  Une variance sous-estimée pour une mesure affectée par des conditions NLOS peut biaiser la correction de l'état dans le filtre de Kalman étendu.

---

## 2. Idée Proposée pour Résoudre le Problème

Afin d'atténuer les effets d'une classification imparfaite des conditions LOS/NLOS, l'idée est d'intégrer directement les mesures issues d'un capteur inertiel (IMU) dans le schéma de filtrage. Cette approche présente plusieurs avantages :

- **Amélioration de la prédiction :**  
  Les mesures IMU, notamment les accélérations mesurées, sont intégrées dans la phase de prédiction pour affiner l'estimation de l'accélération du système. Un blending (pondération) des accélérations prédites et mesurées permet de tirer parti de la rapidité des mesures IMU.
  
- **Renforcement de la robustesse en situation d'incertitude :**  
  Lorsque la classification LOS/NLOS est douteuse, la fusion des informations UWB (distance) avec les mesures IMU (accélération) permet d'obtenir une estimation plus fiable de l'état. De plus, en appliquant une détection de zéro vitesse (ZUPT) lorsque l'accélération est faible, des contraintes supplémentaires sont introduites pour renforcer la stabilité du système.

- **Réduction de l'impact des erreurs de classification :**  
  L'ajout des mesures IMU dans le vecteur de mesure unifié offre une redondance qui aide à compenser les erreurs potentielles dans l'adaptation de la matrice $\mathbf{R}$ basée sur le vecteur $\text{is\_los}$.

---

## 3. Modèle Mathématique

### 3.1. Modélisation de l'État et Dynamique du Système

Le vecteur d'état est défini comme :
$$
\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \\ a_x \\ a_y \end{bmatrix},
$$
représentant la position, la vitesse et l'accélération dans le plan.

La dynamique du système suit un modèle de mouvement à accélération constante :
$$
\mathbf{x}_{k+1} = \mathbf{F}\,\mathbf{x}_k + \mathbf{w}_k,
$$
avec la matrice de transition :
$$
\mathbf{F} = \begin{bmatrix}
1 & 0 & dt & 0 & \tfrac{1}{2} dt^2 & 0 \\
0 & 1 & 0 & dt & 0 & \tfrac{1}{2} dt^2 \\
0 & 0 & 1 & 0 & dt & 0 \\
0 & 0 & 0 & 1 & 0 & dt \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix},
$$
où $dt$ représente l'intervalle de temps entre deux mises à jour.

Le bruit de processus, supposé gaussien, est caractérisé par une covariance $\mathbf{Q}$ définie comme suit :
$$
\mathbf{Q} = \begin{bmatrix}
\frac{dt^4}{4}\,\sigma_a & 0 & 0 & 0 & 0 & 0 \\
0 & \frac{dt^4}{4}\,\sigma_a & 0 & 0 & 0 & 0 \\
0 & 0 & dt^2\,\sigma_a & 0 & 0 & 0 \\
0 & 0 & 0 & dt^2\,\sigma_a & 0 & 0 \\
0 & 0 & 0 & 0 & dt\,\sigma_j & 0 \\
0 & 0 & 0 & 0 & 0 & dt\,\sigma_j 
\end{bmatrix},
$$
où $\sigma_a$ est le bruit d'accélération et $\sigma_j$ celui du « jerk » (variation de l'accélération).

La prédiction de l'état est alors donnée par :
$$
\hat{\mathbf{x}}_{k|k-1} = \mathbf{F}\,\mathbf{x}_k.
$$

**Incorporation des mesures IMU :**  
Les mesures IMU, fournissant les accélérations $a_{x,\text{IMU}}$ et $a_{y,\text{IMU}}$, sont intégrées dans la prédiction à l'aide d'un blending :
$$
\hat{a}_x = (1-\omega)\,\hat{a}_{x,\text{pred}} + \omega\,a_{x,\text{IMU}},
$$
$$
\hat{a}_y = (1-\omega)\,\hat{a}_{y,\text{pred}} + \omega\,a_{y,\text{IMU}},
$$
avec $\omega$ (par exemple 0.7) représentant le poids attribué aux mesures IMU. La covariance prédite se calcule par :
$$
\mathbf{P}_{k|k-1} = \mathbf{F}\,\mathbf{P}_k\,\mathbf{F}^T + \mathbf{Q}.
$$

---

### 3.2. Modèle de Mesure

Les mesures proviennent de trois sources distinctes :

#### 3.2.1. Mesures UWB

Pour chaque ancre $i$, la mesure de distance est donnée par :
$$
z_i = \sqrt{(x - x_i)^2 + (y - y_i)^2} + v_i,
$$
où $v_i$ est un bruit gaussien de variance $r_i$.

**Adaptation de la variance en fonction de LOS/NLOS :**
$$
r_i = 
\begin{cases}
r_{\text{LOS}}, & \text{si } \text{is\_los}[i] = 0, \\
\lambda\,r_{\text{LOS}}, & \text{si } \text{is\_los}[i] = 1,
\end{cases}
$$
avec $\lambda > 1$ (par exemple, $\lambda = 10$).

**Mise à jour adaptative :**  
Après avoir calculé l'innovation pour la mesure $i$ :
$$
y_i = z_i - h_i(\hat{\mathbf{x}}_{k|k-1}),
$$
une estimation provisoire de la variance est obtenue par :
$$
r_{i,\text{new}} = \left| y_i^2 - \left[\mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T\right]_{ii} \right|.
$$
Cette valeur est ajustée en fonction du flag NLOS :
$$
r_{i,\text{new}} =
\begin{cases}
r_{i,\text{new}}, & \text{si } \text{is\_los}[i] = 0, \\
\lambda\,r_{i,\text{new}}, & \text{si } \text{is\_los}[i] = 1.
\end{cases}
$$
Enfin, une mise à jour lissée est appliquée :
$$
r_i \leftarrow \alpha\,r_i + (1-\alpha)\,r_{i,\text{new}},
$$
avec $\alpha$ déterminant le degré de lissage.

#### 3.2.2. Mesures IMU

Les mesures IMU fournissent directement les accélérations mesurées :
$$
z_{\text{IMU}} = \begin{bmatrix} a_{x,\text{IMU}} \\ a_{y,\text{IMU}} \end{bmatrix}.
$$
La fonction de mesure correspondante est :
$$
h_{\text{IMU}}(\hat{\mathbf{x}}_{k|k-1}) = \begin{bmatrix} \hat{a}_x \\ \hat{a}_y \end{bmatrix}.
$$
La covariance associée est définie par :
$$
\mathbf{R}_{\text{IMU}} = \operatorname{diag}\Big(0.1 + 0.05\|a_{\text{IMU}}\|,\; 0.1 + 0.05\|a_{\text{IMU}}\|\Big).
$$

#### 3.2.3. Mesures ZUPT (Zero-Velocity UPdaTe)

Lorsque l'intensité de l'accélération mesurée est inférieure à un seuil $\text{zupt\_threshold}$, le système est supposé être à vitesse nulle. On impose alors :
$$
z_{\text{ZUPT}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix},
$$
avec la fonction de mesure prédite :
$$
h_{\text{ZUPT}}(\hat{\mathbf{x}}_{k|k-1}) = \begin{bmatrix} \hat{v}_x \\ \hat{v}_y \end{bmatrix},
$$
et une covariance très faible :
$$
\mathbf{R}_{\text{ZUPT}} = \operatorname{diag}(0.001, 0.001).
$$

**Fusion des mesures :**  
Les mesures issues des différents capteurs sont concaténées dans un vecteur unifié :
$$
\mathbf{z} = \begin{bmatrix} z_{\text{UWB}} \\ z_{\text{IMU}} \\ z_{\text{ZUPT}} \end{bmatrix},
$$
de même que les fonctions de mesure correspondantes :
$$
h(\hat{\mathbf{x}}_{k|k-1}) = \begin{bmatrix} h_{\text{UWB}}(\hat{\mathbf{x}}_{k|k-1}) \\ h_{\text{IMU}}(\hat{\mathbf{x}}_{k|k-1}) \\ h_{\text{ZUPT}}(\hat{\mathbf{x}}_{k|k-1}) \end{bmatrix}.
$$
La matrice Jacobienne $\mathbf{H}$ et la covariance de mesure $\mathbf{R}$ sont construites en blocs, chacun correspondant aux différents types de mesures.

---

### 3.3. Mise à Jour du Filtre de Kalman Étendu

#### Innovation et Gain de Kalman

L'innovation globale est calculée par :
$$
\mathbf{y} = \mathbf{z} - h(\hat{\mathbf{x}}_{k|k-1}),
$$
et sa covariance par :
$$
\mathbf{S} = \mathbf{H}\,\mathbf{P}_{k|k-1}\,\mathbf{H}^T + \mathbf{R}.
$$
Le gain de Kalman est obtenu par :
$$
\mathbf{K} = \mathbf{P}_{k|k-1}\,\mathbf{H}^T\,\mathbf{S}^{-1}.
$$

#### Correction de l'État et de la Covariance

L'état corrigé est mis à jour selon :
$$
\hat{\mathbf{x}}_{k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}\,\mathbf{y},
$$
et la covariance de l'estimation se met à jour par :
$$
\mathbf{P}_{k} = \left(\mathbf{I} - \mathbf{K}\,\mathbf{H}\right)\mathbf{P}_{k|k-1}.
$$

---

## 4. Conclusion

Le filtre LA-AEKF offre une approche robuste pour la localisation en intégrant intelligemment les mesures UWB et en adaptant la matrice de covariance en fonction des conditions LOS/NLOS. Toutefois, la classification des mesures, avec une précision d'environ 90 %, peut encore laisser des incertitudes dans l'estimation. L'ajout des mesures IMU dans le processus de fusion, couplé à une détection de zéro vitesse (ZUPT), constitue une amélioration significative. Cette intégration permet de compenser les erreurs de classification et d'améliorer la prédiction de l'état, aboutissant à une estimation plus fiable de la position dans des environnements complexes.

Ce rapport a ainsi présenté en détail les avantages du LA-AEKF, la problématique liée aux mesures NLOS, l'idée innovante d'intégrer les mesures IMU, et a finalement établi le modèle mathématique complet de l'approche proposée.