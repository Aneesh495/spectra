# Mathematical Foundations of Spectra

Spectra synthesizes ideas from differential geometry, algebraic topology, continuous Hamiltonian mechanics, spectral theory, and functional operator learning into a unified mathematical computing framework. This document presents the formal definitions, theorems, and mathematical formulations underlying each core component.

---

## 1. Riemannian Geometry and Optimization on Manifolds

### 1.1 Riemannian Manifolds and Metric Tensors
A smooth manifold $\mathcal{M}$ of dimension $d$ endowed with a smoothly varying inner product on its tangent spaces $T_p \mathcal{M}$ is a Riemannian manifold $(\mathcal{M}, g)$:

$$g_p : T_p \mathcal{M} \times T_p \mathcal{M} \to \mathbb{R}$$

For any two tangent vectors $u, v \in T_p \mathcal{M}$, the inner product is symmetric, positive definite, and induces the Riemannian norm $\|u\|_g = \sqrt{g_p(u, u)}$.

### 1.2 Geodesics and Exponential Maps
A geodesic $\gamma: [0, 1] \to \mathcal{M}$ is a curve of zero acceleration with respect to the Levi-Civita connection $\nabla$:

$$\nabla_{\dot{\gamma}(t)} \dot{\gamma}(t) = 0$$

The Riemannian exponential map $\exp_p: T_p \mathcal{M} \to \mathcal{M}$ maps a tangent vector $v \in T_p \mathcal{M}$ to the point $\gamma(1)$ along the unique geodesic starting at $p$ with initial velocity $\dot{\gamma}(0) = v$. The logarithmic map $\log_p: \mathcal{M} \to T_p \mathcal{M}$ is its local inverse.

### 1.3 Retractions and Vector Transport
To avoid expensive numerical geodesic integration in iterative optimization, Spectra employs smooth first-order retractions $R_p: T_p \mathcal{M} \to \mathcal{M}$ satisfying:
1. $R_p(0) = p$
2. $D R_p(0) = \mathrm{id}_{T_p \mathcal{M}}$

For submanifolds embedded in Euclidean space $\mathbb{R}^n$, standard metric projections serve as retractions:
- **Hypersphere $S^n$**: $R_p(\xi) = \frac{p + \xi}{\|p + \xi\|_2}$
- **Stiefel Manifold $V_k(\mathbb{R}^n)$**: $R_X(\xi) = \mathrm{qf}(X + \xi)$, where $\mathrm{qf}(A)$ denotes the $Q$ factor of the thin QR decomposition of $A$.
- **Grassmannian $Gr(k, n)$**: Subspace representation via orthogonal projectors with polar decomposition retraction.
- **Symmetric Positive Definite $\mathrm{SPD}(n)$**: Under the affine-invariant metric $g_P(U, V) = \mathrm{tr}(P^{-1} U P^{-1} V)$, the geodesic is:
  $$\exp_P(\Xi) = P^{1/2} \exp(P^{-1/2} \Xi P^{-1/2}) P^{1/2}$$

Vector transport $\mathcal{T}_{p \to q}(\xi): T_p \mathcal{M} \to T_q \mathcal{M}$ transports tangent vectors across distinct tangent spaces, satisfying linearity and preservation of projection operators:

$$\mathcal{T}_{p \to q}(\xi) = \Pi_q(\xi)$$

where $\Pi_q$ denotes the orthogonal projection onto the tangent space $T_q \mathcal{M}$.

---

## 2. Cellular Sheaf Theory and Sheaf Laplacians

### 2.1 Cell Complexes
Let $X = (V, E, F)$ be a regular cell complex consisting of vertices (0-cells), edges (1-cells), and 2-cells. The incidence relation between cell $\sigma$ and cell $\tau$ with $\dim(\sigma) = \dim(\tau) - 1$ is denoted $\sigma \trianglelefteq \tau$, equipped with signed orientation $[\sigma : \tau] \in \{-1, +1\}$.

### 2.2 Cellular Sheaves
A cellular sheaf $\mathcal{F}$ over a cell complex $X$ associates:
1. To each cell $\sigma \in X$, a real vector space $\mathcal{F}(\sigma) \cong \mathbb{R}^{d_\sigma}$ called the stalk over $\sigma$.
2. To each incidence relation $v \trianglelefteq e$, a linear mapping $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$ called the restriction map.
3. For higher-order incidences $v \trianglelefteq e \trianglelefteq \sigma$, the restriction maps satisfy functorial composition:
   $$\mathcal{F}_{e \trianglelefteq \sigma} \circ \mathcal{F}_{v \trianglelefteq e} = \mathcal{F}_{e' \trianglelefteq \sigma} \circ \mathcal{F}_{v \trianglelefteq e'}$$

### 2.3 Cochain Spaces and Coboundary Operators
The space of 0-cochains $C^0(X; \mathcal{F})$ is the direct sum of stalk spaces over vertices:

$$C^0(X; \mathcal{F}) = \bigoplus_{v \in V} \mathcal{F}(v) \cong \mathbb{R}^{\sum_v d_v}$$

The space of 1-cochains $C^1(X; \mathcal{F})$ is the direct sum over edges:

$$C^1(X; \mathcal{F}) = \bigoplus_{e \in E} \mathcal{F}(e) \cong \mathbb{R}^{\sum_e d_e}$$

The cellular sheaf coboundary operator $\delta^0: C^0(X; \mathcal{F}) \to C^1(X; \mathcal{F})$ evaluates the discrepancy across each oriented edge $e = (u, v)$:

$$(\delta^0 x)_e = \mathcal{F}_{v \trianglelefteq e} x_v - \mathcal{F}_{u \trianglelefteq e} x_u$$

### 2.4 Sheaf Laplacian and Cohomology
The degree-0 cellular Sheaf Laplacian $L_\mathcal{F}: C^0(X; \mathcal{F}) \to C^0(X; \mathcal{F})$ is defined as:

$$L_\mathcal{F} = (\delta^0)^* \delta^0$$

In block form, the entry corresponding to vertices $u, v$ is given by:

$$(L_\mathcal{F})_{u, u} = \sum_{e \ni u} \mathcal{F}_{u \trianglelefteq e}^* \mathcal{F}_{u \trianglelefteq e}$$

$$(L_\mathcal{F})_{u, v} = -\sum_{e = (u, v)} \mathcal{F}_{u \trianglelefteq e}^* \mathcal{F}_{v \trianglelefteq e} \quad (u \neq v)$$

The zeroth cellular sheaf cohomology space $H^0(X; \mathcal{F})$ consists of global sections, which are 0-cochains that satisfy the local consistency relations across all edges simultaneously:

$$H^0(X; \mathcal{F}) = \ker(\delta^0) = \ker(L_\mathcal{F})$$

The Sheaf Dirichlet Energy for a cochain $x \in C^0(X; \mathcal{F})$ measures overall global inconsistency:

$$\mathcal{E}_\mathcal{F}(x) = \langle x, L_\mathcal{F} x \rangle = \|\delta^0 x\|^2 = \sum_{e = (u, v)} \|\mathcal{F}_{v \trianglelefteq e} x_v - \mathcal{F}_{u \trianglelefteq e} x_u\|^2$$

---

## 3. Continuous Geometric Dynamics and Symplectic Mechanics

### 3.1 Hamiltonian Mechanics and Symplectic Manifolds
Let $\mathcal{M} = \mathbb{R}^{2d}$ denote the phase space with canonical coordinates $(q, p)$, where $q \in \mathbb{R}^d$ are generalized positions and $p \in \mathbb{R}^d$ are conjugate momenta. The canonical symplectic 2-form $\omega$ is:

$$\omega = \sum_{i=1}^d dq_i \wedge dp_i = \begin{bmatrix} dq \\ dp \end{bmatrix}^T J \begin{bmatrix} dq \\ dp \end{bmatrix}, \quad J = \begin{bmatrix} 0 & I_d \\ -I_d & 0 \end{bmatrix}$$

Given a scalar Hamiltonian function $\mathcal{H}(q, p)$, Hamilton's equations of motion are generated by the Hamiltonian vector field $X_\mathcal{H}$:

$$\frac{d}{dt} \begin{bmatrix} q \\ p \end{bmatrix} = J \nabla \mathcal{H}(q, p) = \begin{bmatrix} \nabla_p \mathcal{H}(q, p) \\ -\nabla_q \mathcal{H}(q, p) \end{bmatrix}$$

By skew-symmetry of $J$, total energy is strictly conserved along all trajectories:

$$\frac{d\mathcal{H}}{dt} = \langle \nabla \mathcal{H}, \dot{z} \rangle = \nabla \mathcal{H}^T J \nabla \mathcal{H} = 0$$

### 3.2 Symplectic Geometric Integrators
A discrete numerical flow $\Phi_h: (q_n, p_n) \mapsto (q_{n+1}, p_{n+1})$ is symplectic if its Jacobian matrix $M = \frac{\partial (q_{n+1}, p_{n+1})}{\partial (q_n, p_n)}$ preserves the symplectic matrix:

$$M^T J M = J$$

Spectra implements multiple symplectic integration families:
- **Störmer-Verlet (Order 2)**:
  $$p_{n+1/2} = p_n - \frac{h}{2} \nabla V(q_n)$$
  $$q_{n+1} = q_n + h M^{-1} p_{n+1/2}$$
  $$p_{n+1} = p_{n+1/2} - \frac{h}{2} \nabla V(q_{n+1})$$
- **Forest-Ruth Symplectic Composition (Order 4)**:
  Composed of sub-steps with coefficients $\theta = \frac{1}{2 - 2^{1/3}}$:
  $$c_1 = c_4 = \frac{\theta}{2}, \quad c_2 = c_3 = \frac{1 - \theta}{2}$$
  $$d_1 = d_3 = \theta, \quad d_2 = 1 - 2\theta, \quad d_4 = 0$$
- **Gauss-Legendre Implicit Runge-Kutta (Orders 4 and 6)**:
  A Runge-Kutta method with Butcher tableau $(A, b, c)$ is unconditionally symplectic if and only if:
  $$b_i a_{ij} + b_j a_{ji} - b_i b_j = 0 \quad \forall i, j$$

### 3.3 Continuous Adjoint Sensitivity for Neural ODEs
For a state evolution governed by a parametric vector field $\frac{dz}{dt} = f(z(t), t, \theta)$ with scalar terminal loss $\mathcal{L}(z(T))$, the continuous adjoint state $a(t) = \frac{\partial \mathcal{L}}{\partial z(t)}$ satisfies the backward linear ODE:

$$\frac{da(t)}{dt} = -a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial z}$$

The parameter gradients are evaluated by continuous integration along the trajectory:

$$\frac{d\mathcal{L}}{d\theta} = -\int_T^0 a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial \theta} dt$$

---

## 4. Spectral Neural Operators in Infinite Dimensions

### 4.1 Operator Learning Problem Formulation
Let $\mathcal{A} \subset C(D; \mathbb{R}^{d_a})$ and $\mathcal{U} \subset C(D; \mathbb{R}^{d_u})$ denote Banach spaces of functions defined over a bounded domain $D \subset \mathbb{R}^d$. We seek to approximate a nonlinear map $G^\dagger: \mathcal{A} \to \mathcal{U}$ from paired observations $\{a_j, u_j\}_{j=1}^N$ such that the model is mesh-independent and evaluates continuously at arbitrary target resolutions.

### 4.2 Fourier Neural Operator (FNO)
The Fourier Neural Operator defines iterative hidden representations $v_0(x) \mapsto v_1(x) \mapsto \dots \mapsto v_L(x)$ via:

$$v_{l+1}(x) = \sigma \left( W_l v_l(x) + (\mathcal{K}_l v_l)(x) \right)$$

where $W_l$ is a local channel-mixing affine map, $\sigma$ is an element-wise activation function, and $\mathcal{K}_l$ is a non-local kernel integral operator defined in frequency space:

$$(\mathcal{K}_l v)(x) = \mathcal{F}^{-1} \left( R_l(k) \cdot (\mathcal{F} v)(k) \right)(x)$$

Here $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote forward and inverse Fourier transforms, and $R_l(k) \in \mathbb{C}^{d_{out} \times d_{in}}$ is a parameterized complex weight tensor truncated to the lowest $K_{max}$ frequency modes:

$$R_l(k) = 0 \quad \text{for } \|k\|_\infty > K_{max}$$

### 4.3 Chebyshev Spectral Collocation Operators
For non-periodic domains with boundary conditions on $x \in [-1, 1]$, Fourier expansions experience Gibbs ringing. Chebyshev Neural Operators replace the trigonometric basis with Chebyshev polynomials of the first kind $T_n(x) = \cos(n \arccos(x))$ using Gauss-Lobatto collocation nodes:

$$x_j = \cos\left(\frac{\pi j}{N}\right), \quad j = 0, \dots, N$$

Polynomial evaluation is carried out via the Clenshaw-Smith backward recurrence relation:

$$b_{k}(x) = c_k + 2x b_{k+1}(x) - b_{k+2}(x)$$

---

## 5. Koopman and Ergodic Operator Theory

### 5.1 The Koopman Operator
Given an autonomous discrete dynamical system $x_{t+1} = F(x_t)$ on state space $\Omega$, the Koopman operator $\mathcal{K}$ is an infinite-dimensional linear composition operator acting on scalar observable functions $g \in L^2(\Omega)$:

$$\mathcal{K} g = g \circ F, \quad (\mathcal{K} g)(x) = g(F(x))$$

Even when the underlying dynamics $F$ are strongly nonlinear or chaotic, the operator $\mathcal{K}$ is strictly linear.

### 5.2 Dynamic Mode Decomposition (DMD) and HAVOK
Given snapshot matrices $X = [x_1, \dots, x_{m-1}]$ and $Y = [x_2, \dots, x_m]$, Exact DMD finds the best-fit linear operator $A \approx Y X^\dagger$ via truncated SVD:

$$X \approx U \Sigma V^* \implies \tilde{A} = U^* Y V \Sigma^{-1}$$

Eigenvalues $\lambda_i$ and eigenvectors $w_i$ of $\tilde{A}$ yield the continuous growth rates $\omega_i = \frac{\ln(\lambda_i)}{\Delta t}$ and dynamic spatial modes $\phi_i = Y V \Sigma^{-1} w_i$.

Hankel Alternative View of Koopman (HAVOK) constructs a delay-coordinate embedding matrix:

$$H = \begin{bmatrix} x_1 & x_2 & \dots & x_{m-q+1} \\ x_2 & x_3 & \dots & x_{m-q+2} \\ \vdots & \vdots & \ddots & \vdots \\ x_q & x_{q+1} & \dots & x_m \end{bmatrix}$$

Singular value decomposition of $H$ provides an intrinsic coordinate basis that linearizes chaotic attractors (such as the Lorenz-63 system) with closed-form linear dynamics and sparse nonlinear intermittent forcing.
