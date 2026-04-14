# Macro Placement: Formal Problem Statement

---

## 1. Inputs

**Canvas**
A bounded rectangular region $\Omega = [0, W] \times [0, H] \subset \mathbb{R}^2$, with $W, H > 0$ given in microns.

**Macros**
A set of $N$ rectangular blocks indexed $i = 1, \ldots, N$. Each macro $i$ has:
- Fixed dimensions $(w_i, h_i) \in \mathbb{R}^2_+$ (width, height — not decision variables)
- A type flag: hard macro or soft macro
- A mobility flag: movable or fixed

Let $N_{\text{hard}} \subset \{1,\ldots,N\}$ denote hard macros (indices $1$ through $n_h$) and $N_{\text{soft}} \subset \{1,\ldots,N\}$ denote soft macros (indices $n_h+1$ through $N$). Fixed macros have prescribed positions that cannot change.

**Netlist**
A hypergraph $G = (V, E)$ where $V = \{1,\ldots,N\}$ and $E = \{S_1, S_2, \ldots, S_M\}$ is a collection of $M$ nets. Each net $S_j \subseteq V$ is a subset of macros (typically $|S_j| \in [2, 10]$) that must be electrically connected. This is not a standard graph — nets connect subsets, not pairs.

**Density Grid**
A partition of $\Omega$ into a uniform grid of $R$ rows $\times$ $C$ columns. Each grid cell has area $A_{\text{cell}} = (W/C) \times (H/R)$.

---

## 2. Decision Variables

For each movable macro $i$, choose a center position:

$$p_i = (x_i, y_i) \in \mathbb{R}^2$$

The full solution is a vector $p \in \mathbb{R}^{2N}$ containing positions of all macros. Fixed macros contribute known constants. Soft macro positions are typically inherited from the input and not optimized.

**The effective degrees of freedom** are $2 \times |\{\text{movable hard macros}\}|$, typically 400–1000 real-valued variables.

Positions are continuous. There is no discretization or grid snapping.

---

## 3. Objective Function

Minimize the proxy cost:

$$f(p) = 1.0 \cdot WL(p) + 0.5 \cdot D(p) + 0.5 \cdot C(p)$$

where:

### 3.1 Wirelength: $\text{WL}(p)$

$$\text{WL}(p) = \sum_{j=1}^{M} \text{HPWL}_j(p)$$

$$\text{HPWL}_j(p) = \left[\max_{i \in S_j} x_i - \min_{i \in S_j} x_i\right] + \left[\max_{i \in S_j} y_i - \min_{i \in S_j} y_i\right]$$

This is the half-perimeter of the bounding box of net $j$.

**Properties:**
- Piecewise linear in $p$
- Convex (max and min are convex/concave; their difference over disjoint variable sets is convex)
- Non-differentiable at points where the argmax/argmin of a net changes
- Separable across nets but coupled across macros sharing nets

**Smooth approximation** (used by all gradient-based solvers):

$$\text{HPWL}_j \approx \gamma \cdot \ln\!\left(\sum_{i \in S_j} \exp(x_i / \gamma)\right) + \gamma \cdot \ln\!\left(\sum_{i \in S_j} \exp(-x_i / \gamma)\right) + [\text{same for } y]$$

where $\gamma > 0$ is a smoothing parameter. As $\gamma \to 0$, this recovers exact HPWL.

### 3.2 Density: $D(p)$

For each grid cell $(r,c)$, compute area utilization:

$$\rho(r,c) = \frac{1}{A_\text{cell}} \sum_{i=1}^{N} \mathrm{overlapArea}(\mathrm{macro}_i, \mathrm{cell}(r,c))$$

where $\mathrm{overlapArea}$ is the intersection area of the macro's footprint $[x_i - w_i/2,\; x_i + w_i/2] \times [y_i - h_i/2,\; y_i + h_i/2]$ with the grid cell.

$$D(p) = \text{mean of top 10\% of } \{\rho(r,c)\} \text{ values}$$

**Properties:**
- Piecewise polynomial in $p$ (intersection area is piecewise bilinear)
- Non-convex (top-10% selection creates non-smooth, non-convex behavior)
- Measures local crowding — penalizes hotspots, not total area

### 3.3 Congestion: $C(p)$

For each grid cell $(r,c)$, estimate routing demand:

$$\text{demand}(r,c) = |\{j : \text{cell}(r,c) \cap \text{bbox}(S_j) \neq \emptyset\}|$$

where $\text{bbox}(S_j)$ is the bounding box of net $j$.

$$C(p) = \text{smoothed mean of top 5\% of } \{\text{demand}(r,c) / \text{capacity}(r,c)\} \text{ values}$$

**Properties:**
- Piecewise constant in raw form (demand changes discretely as net bboxes cross grid boundaries)
- Non-convex, non-smooth
- Proxies for downstream routing difficulty

---

## 4. Hard Constraints

### 4.1 Canvas Bounds

For all $i = 1, \ldots, N$:

$$x_i - w_i/2 \geq 0$$
$$x_i + w_i/2 \leq W$$
$$y_i - h_i/2 \geq 0$$
$$y_i + h_i/2 \leq H$$

These are $4N$ linear inequalities. They define a box constraint and are trivially satisfiable.

### 4.2 Non-Overlap (Hard Macros Only)

For every pair $(i, k)$ with $i, k \in N_{\text{hard}}$ and $i < k$, at least one of:

$$x_i + w_i/2 \leq x_k - w_k/2 \quad (i \text{ left of } k)$$
$$x_k + w_k/2 \leq x_i - w_i/2 \quad (k \text{ left of } i)$$
$$y_i + h_i/2 \leq y_k - h_k/2 \quad (i \text{ below } k)$$
$$y_k + h_k/2 \leq y_i - h_i/2 \quad (k \text{ below } i)$$

This is a **disjunctive constraint**: one of four linear inequalities must hold, but which one is not specified. There are $n_h \cdot (n_h - 1)/2$ such constraints.

**This is the sole source of NP-hardness.**

### 4.3 Fixed Macros

For each fixed macro $i$:

$$p_i = p_i^0 \quad (\text{given constant})$$

Linear equality constraints.

### 4.4 Soft Macros

Soft macros may overlap with each other and with hard macros. They represent pre-placed standard cell clusters and are not subject to non-overlap constraints.

---

## 5. Formal Problem Statement

$$
\begin{aligned}
\text{minimize} \quad & f(p) = \text{WL}(p) + 0.5 \cdot D(p) + 0.5 \cdot C(p) \\
\text{subject to} \quad & p_i \in B_i & \text{for all } i & \quad (\text{canvas bounds}) \\
& \text{nonoverlap}(i,k) & \text{for all } i < k \in N_{\text{hard}} & \quad (\text{disjunction}) \\
& p_i = p_i^0 & \text{for all fixed } i & \quad (\text{fixed positions})
\end{aligned}
$$

where $B_i = [w_i/2,\; W - w_i/2] \times [h_i/2,\; H - h_i/2]$ is the feasible box for macro $i$.

---

## 6. Output

A position tensor $p^* \in \mathbb{R}^{N \times 2}$ satisfying all constraints, with lowest achievable $f(p^*)$. Specifically:
- Zero hard-macro overlaps
- All macros within canvas
- Fixed macros unmoved
- Soft macros at provided or initial positions

---

## 7. Key Structural Properties

### 7.1 The Decomposition Theorem

**Given a fixed assignment of all disjunctive constraints** (i.e., for each pair $(i,k)$, one of the four separation conditions is chosen), the remaining problem is:

$$
\begin{aligned}
\text{minimize} \quad & \text{WL}(p) & (\text{convex, piecewise linear}) \\
\text{subject to} \quad & p \in P & (\text{intersection of half-spaces} = \text{convex polyhedron})
\end{aligned}
$$

This is a **linear program**, solvable in polynomial time. The entire computational difficulty resides in selecting the disjunctive assignments.

### 7.2 Search Space Size

With $n_h$ hard macros, the number of possible disjunctive assignments is:

$$4^{n_h \cdot (n_h - 1)/2}$$

For $n_h = 300$: $4^{44{,}850} \approx 10^{26{,}970}$. Not all assignments yield feasible polyhedra, but the feasible subset is still exponentially large.

### 7.3 Feasible Region Geometry

The feasible region $F$ is a **union of convex polyhedra**:

$$F = \bigcup_{\sigma \in \Sigma} P_\sigma$$

where $\Sigma$ is the set of consistent disjunctive assignments and $P_\sigma$ is the convex polyhedron defined by assignment $\sigma$. The polyhedra may share boundaries but their interiors are disjoint. The number $|\Sigma|$ is exponential in $n_h$.

This means:
- $F$ is non-convex and disconnected (in general)
- Local search can get trapped in a single polyhedron
- Moving between polyhedra requires crossing an infeasible region (overlapping configurations)
- The global minimum may be in a polyhedron that is not reachable by continuous deformation from the current solution

### 7.4 Objective Landscape

$\text{WL}(p)$ is convex but $D(p)$ and $C(p)$ are not. The combined objective $f(p)$ is non-convex even within a single polyhedron $P_\sigma$. Combined with the disconnected feasible region, this creates a landscape with:

- Exponentially many local minima (one or more per polyhedron)
- Discontinuous jumps in quality between adjacent polyhedra
- Long-range dependencies (moving one macro changes costs for all nets it participates in)
- Competing gradients (wirelength pulls toward clustering; density pushes toward spreading)
