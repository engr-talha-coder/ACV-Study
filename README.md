# Comprehensive Guide to Applied Computer Vision

## Course Overview (CAP5415 - University of Central Florida)

This guide synthesizes lecture materials covering fundamental to advanced topics in computer vision, from image basics to deep learning and generative models.

---

## Part 1: Image Fundamentals

### 1.1 What is Computer Vision?
- **Definition**: Enabling computers to understand visual data (images, videos) and automate tasks the human visual system performs
- **Goal**: Bridge the gap between image pixels and semantic meaning

### 1.2 Image Digitization

**Sampling**: Discretization of spatial domain (pixels)
**Quantization**: Discretization of intensity range (e.g., 0-255 for 8-bit)

**Image Types**:
- **Binary**: 1 bit per pixel (0 or 1)
- **Grayscale**: Typically 8-bit (0 = black, 255 = white)
- **RGB Color**: 3 channels (Red, Green, Blue), each typically 8-bit

**Color Perception**:
- Human cones: ~64% Red, ~32% Green, ~2% Blue
- Wavelength ranges:
  - Red: 625-780 nm
  - Green: 400-585 nm
  - Blue: 440-485 nm

### 1.3 Image as a Function
An image P is a function defined on a rectangular subset G of a regular planar orthogonal array:
- **G** = 2D grid
- Each element = pixel
- P(p) assigns a value to each pixel p ∈ G

---

## Part 2: Image Filtering and Processing

### 2.1 Correlation and Convolution

**Correlation**:
$$(f \otimes h)[m,n] = \sum_{k}\sum_{l} f[k,l] \cdot h[m+k, n+l]$$

**Convolution**:
$$(f * h)[m,n] = \sum_{k}\sum_{l} f[k,l] \cdot h[m-k, n-l]$$

**Key Properties of Linear Filters**:
- Linearity: filter(f₁ + f₂) = filter(f₁) + filter(f₂)
- Shift invariance: same behavior regardless of location
- Any linear, shift-invariant operator = convolution
- Commutative: a * b = b * a
- Associative: a * (b * c) = (a * b) * c
- Distributes over addition: a * (b + c) = (a * b) + (a * c)

### 2.2 Common Filters

**Box Filter (Averaging)**:
- Replaces each pixel with average of neighborhood
- Effect: Smoothing (removes sharp features)

**Gaussian Filter**:
$$g(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

Properties:
- Smooth with infinite derivatives
- Symmetric
- Fourier Transform of Gaussian is Gaussian
- Separable: 2D convolution = two 1D convolutions
- Rule of thumb: filter half-width ≈ 3σ

**Median Filter**:
- Non-linear filter
- Takes median intensity in window
- Excellent for salt-and-pepper noise removal
- Preserves edges better than linear smoothing

**Sharpening Filter**:
- Accentuates differences with local average
- Kernel example: `[0 -1 0; -1 5 -1; 0 -1 0]`

### 2.3 Image Derivatives

**Discrete Derivatives (Finite Differences)**:
- Backward: f'(x) = f(x) - f(x-1) → Kernel `[-1 1]`
- Forward: f'(x) = f(x) - f(x+1) → Kernel `[1 -1]`
- Central: f'(x) = f(x+1) - f(x-1) → Kernel `[-1 0 1]`

**2D Gradient**:
$$\nabla f(x,y) = \begin{bmatrix} f_x \\ f_y \end{bmatrix}$$

**Gradient Magnitude**:
$$|\nabla f(x,y)| = \sqrt{f_x^2 + f_y^2}$$

**Gradient Direction**:
$$\theta = \tan^{-1}\left(\frac{f_y}{f_x}\right)$$

**Sobel Operator** (Horizontal edge detection):
```
[-1 0 1]     [-1 -2 -1]
[-2 0 2]     [ 0  0  0]
[-1 0 1]     [ 1  2  1]
```
Horizontal    Vertical

### 2.4 Image Histogram

- Distribution of pixel intensities
- Useful for: contrast adjustment, thresholding, feature extraction

---

## Part 3: Edge Detection

### 3.1 Why Edge Detection?
- Extract useful information for object recognition
- Recover geometry
- Edges = rapid changes in image intensity

### 3.2 Design Criteria (Canny)
1. **Good detection**: Find all real edges, ignore noise
2. **Good localization**: As close as possible to true edges
3. **Single response**: One point per true edge point

### 3.3 Edge Detection Methods

**Prewitt Operator**:
- Simple derivative approximation
- Kernel size 3×3

**Sobel Operator**:
- Better noise suppression than Prewitt
- Weighted average for smoother derivative

**Marr-Hildreth (LoG - Laplacian of Gaussian)**:

Steps:
1. Smooth with Gaussian: $g(x,y) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2+y^2}{2\sigma^2}}$
2. Apply Laplacian: $\Delta^2 S = \frac{\partial^2 S}{\partial x^2} + \frac{\partial^2 S}{\partial y^2}$
3. Find zero-crossings

LoG formula:
$$\Delta^2 g(x,y) = -\frac{1}{\sqrt{2\pi}\sigma^3}\left(2 - \frac{x^2+y^2}{\sigma^2}\right)e^{-\frac{x^2+y^2}{2\sigma^2}}$$

**Canny Edge Detector** (Most widely used):

Algorithm:
1. Smooth image with Gaussian filter
2. Compute gradient magnitude and orientation
3. Apply Non-Maximum Suppression (NMS)
4. Apply Hysteresis thresholding

**Non-Maximum Suppression**:
- For each pixel, check if gradient magnitude is maximum along gradient direction
- Suppress non-maximum pixels

**Hysteresis Thresholding [Low, High]**:
- Above High → edge pixel
- Below Low → non-edge
- Between Low and High → edge if connected to high edge pixel (8-connected)

### 3.4 Derivative Theorem of Convolution
$$\frac{d}{dx}(f * g) = f * \frac{d}{dx}g$$

This allows combining smoothing and differentiation.

---

## Part 4: Feature Detection and Description

### 4.1 Interest Points (Keypoints)

**Characteristics of Good Features**:
- **Distinctiveness**: Uniquely identifiable
- **Repeatability**: Found in multiple images under transformation
- **Compactness**: Fewer features than pixels
- **Efficiency**: Fast to compute

**Types of Interest Points**:
- Corners (abrupt change in boundary direction)
- Blobs
- Edge intersections

### 4.2 Harris Corner Detector

**Core Idea**: Shift window in any direction → large intensity change indicates corner

**Auto-correlation Function**:
$$E(u,v) = \sum_{x,y} w(x,y)[I(x+u, y+v) - I(x,y)]^2$$

**Taylor Series Approximation**:
$$I(x+u, y+v) \approx I(x,y) + I_x u + I_y v$$

**Second Moment Matrix (M)**:
$$M = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

**Cornerness Measure**:
$$R = \det(M) - k \cdot \text{trace}(M)^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2$$

Where:
- λ₁, λ₂ are eigenvalues of M
- k is empirical constant (typically 0.04-0.06)

**Classification based on eigenvalues**:
- |R| small → flat region (λ₁, λ₂ small)
- R < 0 (large magnitude) → edge (one eigenvalue large)
- R large positive → corner (both eigenvalues large)

**Alternative Harris variants**:
- Shi-Tomasi: R = min(λ₁, λ₂)
- Harmonic mean: R = det(M)/trace(M)

### 4.3 Properties of Harris Detector

**Invariant to**:
- Translation (covariant)
- Rotation (covariant - ellipse rotates but eigenvalues unchanged)
- Affine intensity changes (partial invariance)

**Not invariant to**:
- Scaling (fails when scale changes significantly)

### 4.4 Scale-Invariant Feature Transform (SIFT)

**Authors**: David Lowe, 2004 (IJCV)

**Major Stages**:
1. **Scale-space extrema detection** (using Difference-of-Gaussian)
2. **Keypoint localization** (fit detailed model, select stable points)
3. **Orientation assignment** (based on local gradient directions)
4. **Descriptor formation** (128-dim vector)

**Difference of Gaussian (DoG)**:
- Approximation of LoG
- $DoG = G(x,y,k\sigma) - G(x,y,\sigma)$

**SIFT Descriptor** (128 dimensions):
- Divide 16×16 region into 4×4 sub-patches
- Compute 8-bin gradient orientation histogram per sub-patch
- $4 \times 4 \times 8 = 128$ dimensions
- Normalize, clip values > 0.2, renormalize

**Advantages**:
- Invariant to scale and rotation
- Robust to affine distortion, noise, illumination change
- Highly distinctive

### 4.5 Histogram of Oriented Gradients (HOG)

**Application**: Human detection (Dalal & Triggs, CVPR 2005)

**Computation Steps**:
1. Extract square block (e.g., 16×16 pixels)
2. Divide into 2×2 cells (each 8×8 pixels)
3. Compute orientation histogram per cell (9 bins, 0-180°)
4. Concatenate 4 histograms → 36-dim vector
5. Normalize vector

**Parameters**:
- Orientation range: 0-180° (for pedestrians)
- 9 bins (20° each)
- Cell size: 8×8 pixels
- Block size: 2×2 cells (16×16 pixels)
- Final descriptor: 36 dimensions

**Normalization Options**:
1. Euclidean norm: v / ||v||₂
2. L₁ norm: v / ||v||₁
3. L₂ norm, clip > 0.2, renormalize

---

## Part 5: Image Noise

### 5.1 Noise Types

**Gaussian Noise**:
$$n \sim \mathcal{N}(0, \sigma^2)$$
$$I_{\text{observed}} = I_{\text{original}} + n$$

**Salt-and-Pepper Noise**:
- Random pixels become black or white
- Uniform probability distribution

**Multiplicative Noise**:
$$I_{\text{observed}} = I_{\text{original}} \times n$$

### 5.2 Noise Reduction Strategies
- Smoothing filters (Gaussian, Box)
- Median filter (excellent for salt-and-pepper)
- Bilateral filter (preserves edges)

---

## Part 6: Fitting and Model Estimation

### 6.1 Least Squares Line Fitting

**Problem**: Given points (x₁,y₁),...,(x_n,y_n), find line y = mx + b that minimizes:
$$E = \sum_{i=1}^{n} (y_i - mx_i - b)^2$$

**Matrix Form**:
$$\begin{bmatrix} x_1 & 1 \\ \vdots & \vdots \\ x_n & 1 \end{bmatrix} \begin{pmatrix} m \\ b \end{pmatrix} = \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix}$$

**Solution**: $(X^T X) B = X^T Y$

**Problem with slope-intercept**: Fails for vertical lines

### 6.2 Total Least Squares

**Line representation**: $ax + by = d$ with $a^2 + b^2 = 1$

**Perpendicular distance**: $|ax_i + by_i - d|$

**Objective**:
$$E = \sum_{i=1}^{n} (ax_i + by_i - d)^2$$

**Solution**: d = a·x̄ + b·ȳ

Plugging back:
$$E = \sum_{i=1}^{n} [a(x_i - \bar{x}) + b(y_i - \bar{y})]^2$$

Minimize ||UN||² subject to ||N||² = 1, where N = [a b]ᵀ

**Solution**: Eigenvector of UᵀU associated with smallest eigenvalue

**Second Moment Matrix**:
$$U^T U = \begin{bmatrix} \sum (x_i - \bar{x})^2 & \sum (x_i - \bar{x})(y_i - \bar{y}) \\ \sum (x_i - \bar{x})(y_i - \bar{y}) & \sum (y_i - \bar{y})^2 \end{bmatrix}$$

### 6.3 Robust Fitting

**Problem**: Least squares heavily penalizes outliers

**Robust Objective**:
$$\min_{\theta} \sum_{i} \rho_{\sigma}(r(x_i;\theta))$$

where $\rho_{\sigma}$ is a robust function, e.g.:
$$\rho_{\sigma}(u) = \frac{u^2}{\sigma^2 + u^2}$$

**Properties**:
- Less sensitive to outliers
- Requires iterative optimization

### 6.4 RANSAC (Random Sample Consensus)

**Algorithm** (Repeat N times):
1. Randomly select s points (minimum needed for model)
2. Fit model to these s points
3. Find inliers (points with residual < t)
4. If #inliers ≥ d, accept and refit using all inliers

**Choosing Number of Iterations N**:
$$N = \frac{\log(1-p)}{\log(1-(1-e)^s)}$$

Where:
- p = desired probability (e.g., 0.99)
- e = expected outlier ratio
- s = sample size

**Example table** (p = 0.99):

| e (%) | s=2 | s=3 | s=4 | s=5 | s=6 | s=7 | s=8 |
|-------|-----|-----|-----|-----|-----|-----|-----|
| 5 | 2 | 3 | 3 | 4 | 4 | 4 | 5 |
| 10 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| 20 | 5 | 7 | 9 | 12 | 16 | 20 | 26 |
| 30 | 7 | 11 | 17 | 26 | 40 | 61 | 93 |
| 40 | 11 | 19 | 34 | 57 | 97 | 163 | 272 |

**Pros**: Simple, general, works well in practice
**Cons**: Many parameters, iterations grow exponentially with outlier ratio

### 6.5 Hough Transform

**Core Idea**: Voting in parameter space

**Line representation** (polar):
$$\rho = x\cos\theta + y\sin\theta$$

Where:
- ρ = perpendicular distance from origin
- θ = angle of normal vector

**Algorithm**:
1. Initialize accumulator H(θ,ρ) = 0
2. For each feature point (x,y):
   - For θ = 0 to 180°:
     - ρ = x cos θ + y sin θ
     - H(θ,ρ) += 1
3. Find local maxima in H → detected lines

**Pros**:
- Handles occlusion and multiple instances
- Robust to noise
- Can detect any shape with parametric representation

**Cons**:
- Complexity grows exponentially with model parameters
- Hard to choose grid size

### 6.6 Generalized Hough Transform

**Application**: Detecting arbitrary shapes using displacement vectors

**Training**:
1. Build codebook of patches around interest points (k-means clustering)
2. Map each interest point to closest codebook entry
3. Store displacement vectors from each codebook entry to object center

**Detection**:
1. Extract features in test image
2. Look up codebook entries
3. Vote for possible center locations
4. Find peaks → object detections

**K-Means Clustering**:
- Initialize k cluster centers (randomly from data or k-means++)
- Iterate:
  - Assign each point to nearest center
  - Update centers to mean of assigned points
- Stop when centers stabilize

---

## Part 7: Image Alignment and Transformations

### 7.1 Transformation Models

| Transformation | DOF | Parameters | Properties |
|---------------|-----|------------|------------|
| Euclidean | 3 | rotation, translation | Preserves lengths |
| Similarity | 4 | rotation, translation, scale | Preserves angles |
| Affine | 6 | 2×2 matrix + translation | Preserves parallelism |
| Projective (Homography) | 8 | 3×3 matrix | Preserves lines |

### 7.2 Affine Transformations

**Matrix form**:
$$\begin{pmatrix} x_i' \\ y_i' \end{pmatrix} = \begin{bmatrix} m_1 & m_2 \\ m_3 & m_4 \end{bmatrix} \begin{pmatrix} x_i \\ y_i \end{pmatrix} + \begin{pmatrix} t_1 \\ t_2 \end{pmatrix}$$

**Least squares solution** (known correspondences):
- Minimize $\sum_{i=1}^{n} \|x_i' - M x_i - t\|^2$
- Closed form solution exists
- Need at least 3 point correspondences (6 equations, 6 unknowns)

### 7.3 Homography (Projective Transformation)

**For N=2 (plane-to-plane)**:
$$x' = \frac{ax + by + c}{gx + hy + i}, \quad y' = \frac{dx + ey + f}{gx + hy + i}$$

**Matrix form** (homogeneous coordinates):
$$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

**Solving for homography**:
- Each correspondence gives 2 equations
- Need at least 4 points (8 equations)
- Solve using DLT (Direct Linear Transform)

**Equations**:
$$x'(h_{31}x + h_{32}y + h_{33}) = h_{11}x + h_{12}y + h_{13}$$
$$y'(h_{31}x + h_{32}y + h_{33}) = h_{21}x + h_{22}y + h_{23}$$

### 7.4 Feature Matching

**SSD (Sum of Squared Differences)**:
$$\text{SSD}(\mathbf{u},\mathbf{v}) = \sum_i (u_i - v_i)^2$$

**Normalized Correlation**:
$$\rho(\mathbf{u},\mathbf{v}) = \frac{\sum_i (u_i - \bar{u})(v_i - \bar{v})}{\sqrt{(\sum_j (u_j - \bar{u})^2)(\sum_j (v_j - \bar{v})^2)}}$$

**Why normalized correlation over SSD?**
- Invariant to linear brightness changes
- Compensates for differences in contrast

**Lowe's Ratio Test** (reject ambiguous matches):
- Compare distance to nearest neighbor vs second nearest neighbor
- Ratio < 0.8 → accept (empirically determined)

### 7.5 Iterative Closest Points (ICP)

**Algorithm** (start with initial transformation):
1. For each point in transformed source, find nearest neighbor in target
2. Reestimate transformation using those correspondences
3. Apply new transform to transformed source
4. Repeat

### 7.6 Robust Alignment (Iteratively Reweighted Least Squares)

**Weighted least squares**: minimize $\sum_i w_i [x_i - My_i - t]^T[x_i - My_i - t]$

**Robust loss**: minimize $\sum_i \rho([x_i - My_i - t]^T[x_i - My_i - t])$

At solution: $\sum_i \rho' [x_i - My_i - t] = 0$

**IRLS algorithm**:
1. Start with weights = 1
2. Align using weights
3. Update weights to ρ' (derivative of robust function)
4. Iterate

**Example** (L1 loss): ρ(s) = √s → ρ' = 1/(2√s)

### 7.7 Large-Scale Alignment

**Vocabulary Tree** (Hierarchical k-means):
- Build hierarchy of cluster centers
- For query, walk tree to find matching images
- Efficient approximate similarity search

---

## Part 8: Neural Networks and Deep Learning

### 8.1 Bias-Variance Tradeoff

$$\text{Total Loss} = \text{Bias} + \text{Variance} + \text{Noise}$$

- **Bias**: Error from model lacking capacity to represent concept (underfitting)
- **Variance**: Error from overreacting to training noise (overfitting)

### 8.2 Universality Theorem
Any continuous function f: ℝᴺ → ℝᴹ can be realized by a network with one hidden layer (given enough hidden neurons).

**Implication**: Neural networks have very high capacity (millions of parameters)
**Challenge**: Mitigating overfitting

### 8.3 Convolutional Neural Networks (CNNs)

**Key Innovations**:

1. **Local Connectivity**: Neurons only connected to small region of previous layer
2. **Weight Sharing**: Same parameters across spatial positions (shift-invariant filters)

**Convolution Layer**:

Input: $W_{in} \times H_{in} \times D_{in}$
Filter: $W_f \times H_f \times D_{in}$ (same depth as input)
Number of filters: $D_{out}$

**Output dimensions** (stride = s):
$$W_{out} = \frac{W_{in} - W_f}{s} + 1$$
$$H_{out} = \frac{H_{in} - H_f}{s} + 1$$
$$D_{out} = \text{number of filters}$$

**Number of parameters per filter**: $W_f \times H_f \times D_{in}$
**Total parameters**: $D_{out} \times (W_f \times H_f \times D_{in})$

**Example**: 7×7 filter, 3 input channels, 5 filters
- Per filter: 7×7×3 = 147 weights
- Total: 5 × 147 = 735 weights

### 8.4 Pooling

**Purpose**:
- Aggregate multiple values into single value
- Provide invariance to small transformations
- Reduce spatial dimensions
- Increase receptive field

**Common pooling**: Max pooling, Average pooling

**Dimensions after pooling** (kernel size = k, stride = s):
$$W_{out} = \frac{W_{in} - k}{s} + 1$$

### 8.5 Activation Functions

**Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
**Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
**ReLU**: $f(x) = \max(0, x)$

### 8.6 Neural Network Training

**Forward Pass**: Compute predictions from input through layers
**Backpropagation**: Compute gradients of loss w.r.t parameters using chain rule

**Cross-Entropy Loss** (classification):
$$C(\Theta) = -\sum_{i=1}^{n} [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

**Softmax** (multi-class):
$$P(y=c|x,\Theta) = \frac{e^{z_c}}{\sum_{j} e^{z_j}}$$

### 8.7 Optimization Algorithms

**SGD (Stochastic Gradient Descent)**:
$$x_{t+1} = x_t - \alpha \nabla f(x_t)$$

**Mini-batch SGD**:
$$\frac{\sum_{j=1}^{m} \nabla C_{X_j}}{m} \approx \nabla C$$

**SGD with Momentum**:
$$v_{t+1} = \rho v_t + \nabla f(x_t)$$
$$x_{t+1} = x_t - \alpha v_{t+1}$$

**ADAM** (Adaptive Moment Estimation):
- Combines momentum with adaptive learning rates
- Typically converges faster but SGD may give better generalization

### 8.8 Hyperparameter Tuning

**Data Split**:
- Training set: learn parameters
- Validation set: tune hyperparameters
- Test set: evaluate final performance

### 8.9 Data Augmentation

Common augmentations for images:
- Horizontal flips
- Random crops (e.g., sample 224×224 patch from resized image)
- Color jitter (PCA-based color augmentation)
- Rotation, shear, scaling
- Motion blur, lens distortions

---

## Part 9: Generative Models

### 9.1 Generative vs Discriminative

**Supervised Learning**: Learn mapping x → y
**Unsupervised Learning**: Learn hidden structure in x

**Generative Modeling Goal**: Learn P_model(x) similar to P_data(x)

### 9.2 Autoencoders

**Structure**:
- Encoder: maps x to latent z
- Decoder: maps latent z to reconstructed x̂

**Loss**: $\mathcal{L}(x, \hat{x}) = \|x - \hat{x}\|^2$

**Purpose**:
- Dimensionality reduction
- Feature learning (unsupervised)
- Compression (smaller latent space = more compression)

### 9.3 Variational Autoencoders (VAEs)

**Key Difference from Traditional Autoencoder**:
- Encoder outputs distribution parameters (μ, σ) rather than point
- Sample from distribution: $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$

**VAE Loss** (ELBO - Evidence Lower Bound):
$$\mathcal{L}(\phi,\theta,x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization}}$$

**Prior**: $p(z) = \mathcal{N}(0, I)$

**Reparameterization Trick**:
- Enables backpropagation through sampling layer
- $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$

**Properties of good latent space**:
- **Continuity**: Points close in latent space → similar decoded content
- **Completeness**: Sampling from latent space → meaningful content

**β-VAE**:
$$\mathcal{L}(\theta,\phi;x,z,\beta) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z))$$
- β > 1 encourages disentanglement

### 9.4 Generative Adversarial Networks (GANs)

**Core Idea**: Two networks compete:
- **Generator G**: creates fake samples from noise z ∼ P_z(z)
- **Discriminator D**: distinguishes real from fake

**Objective** (min-max game):
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim P_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim P_z(z)}[\log(1 - D(G(z)))]$$

**Optimal Solution**:
When $P_G(x) = P_{data}(x)$:
$$D_G^*(x) = \frac{P_{data}(x)}{P_{data}(x) + P_G(x)} = \frac{1}{2}$$

**At optimal**:
$$C(G) = \max_D V(G,D) = -2\log 2 + 2 \cdot \text{JSD}(P_{data} \| P_G)$$

Where JSD is Jensen-Shannon divergence:
$$\text{JSD}(P\|Q) = \frac{1}{2} D_{KL}(P\|M) + \frac{1}{2} D_{KL}(Q\|M), \quad M = \frac{1}{2}(P+Q)$$

**Training** (alternating):
1. Update D to maximize V(D,G)
2. Update G to minimize V(D,G)

**Mode Collapse Problem**:
- Generator learns to map multiple z values to same output
- Results in low diversity
- Solutions: Mini-batch discrimination, feature matching, TTUR

### 9.5 Conditional GAN (cGAN)

**Idea**: Control generation by conditioning on label y

**Implementation**:
- Concatenate one-hot encoded label y to:
  - Input x for discriminator
  - Noise z for generator

**Objective**:
$$\min_G \max_D V(D,G) = \mathbb{E}_{x,y \sim P_{data}}[\log D(x,y)] + \mathbb{E}_{z \sim P_z, y \sim P_y}[\log(1 - D(G(z,y), y))]$$

### 9.6 Diffusion Models

**Inspiration**: Non-equilibrium thermodynamics

**Key Idea**: Markov chain that gradually adds noise, then learns to reverse

**Forward Process** (add noise):
$$q(x_t|x_{t-1}) = \mathcal{N}(x_t | \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

Where β_t is noise schedule: $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$

**Let**: $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$

**Closed-form sampling**:
$$q(x_t|x_0) = \mathcal{N}(x_t | \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$$

**Reverse Process** (denoise):
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1} | \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Joint distribution**:
$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$

Where $p(x_T) = \mathcal{N}(0, I)$

**Training Objective** (simplified):
$$\mathcal{L}_t = \mathbb{E}_{x_0,\epsilon}\left[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar{\alpha}_t)\|\Sigma_\theta\|_2^2} \|\epsilon_t - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_t, t)\|^2\right]$$

**DDPM Training Algorithm**:
```
while not converged:
    x₀ ~ q₀(x₀)
    t ~ Uniform({1,...,T})
    ε ~ N(0,I)
    Take gradient step on ∇_θ ||ε - ε_θ(√ᾱ_t x₀ + √(1-ᾱ_t)ε, t)||²
```

**DDPM Sampling Algorithm**:
```
x_T ~ N(0, I)
for t = T down to 1:
    ε_t ~ N(0, I) (if t > 1 else ε_t = 0)
    x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t, t)) + σ_t ε_t
return x₀
```

### 9.7 Adversarial Autoencoders (AAE)

- Uses adversarial learning to impose distribution on latent variable
- Alternative to KL divergence in VAE
- Enables semi-supervised learning

**Applications**:
- Unsupervised clustering
- Semi-supervised classification
- Dimensionality reduction

### 9.8 GAN Applications

**Image-to-Image Translation**:
- **PatchGAN**: coloring sketches, day↔night, aerial↔map
- **CycleGAN**: unpaired translation (zebra↔horse, summer↔winter)
- **DeepFaceDrawing**: generate faces from sketches

**Text-to-Image Generation**:
- StackGAN: generate images from descriptive captions

**Vector Arithmetic in Latent Space** (DCGAN):
- e.g., "man with glasses" - "man without glasses" + "woman without glasses" ≈ "woman with glasses"

---

## Part 10: Common Computer Vision Tasks

### 10.1 Recognition Tasks (increasing complexity)

| Task | Description |
|------|-------------|
| Object Recognition | Does image contain object X? |
| Object Localization | Where is the object? (bounding box) |
| Object Detection | Recognize AND localize multiple objects |
| Semantic Segmentation | Label each pixel with class |
| Instance Segmentation | Segment each object instance individually |
| Panoptic Segmentation | Semantic + instance segmentation |

### 10.2 Video Tasks

- Action detection
- Video segmentation
- Object tracking (single/multi-object)
- Video security and monitoring
- Cross-view action synthesis

### 10.3 3D Vision Tasks

- Mesh recovery
- 3D scene reconstruction (NeRF)
- Structure from motion
- Stereo depth estimation

---

## Part 11: Exam Topics Summary

**Linear Algebra**:
- Rank, null space, range, invertible matrices
- Eigen decomposition, SVD, pseudo-inverse
- Basic matrix calculus

**Optimization**:
- Least squares
- Low-rank approximation
- Statistical interpretation of PCA

**Image Formation**:
- Diffuse/specular reflection
- Lambertian lighting equation

**Filtering**:
- Linear filters, convolution vs correlation
- Filter properties, filter banks
- Median filter

**Statistics**:
- Bias, variance, bias-variance tradeoff
- Overfitting, underfitting

**Neural Networks**:
- Linear classifier, softmax
- Why linear classifier is insufficient
- Activation functions, feed-forward pass
- Universality theorem
- Backpropagation
- SGD, momentum
- CNN concepts (local connectivity, weight sharing, pooling)
- Hyperparameter tuning
- Data augmentation

**Generative Models**:
- VAE, GAN, Diffusion models
- ELBO, reparameterization trick

---

## Key Formulas Reference Sheet

| Concept | Formula |
|---------|---------|
| Gaussian filter | $g(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$ |
| Gradient magnitude | $|\nabla f| = \sqrt{f_x^2 + f_y^2}$ |
| Harris cornerness | $R = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)^2$ |
| LoG | $\Delta^2 g = -\frac{1}{\sqrt{2\pi}\sigma^3}(2-\frac{x^2+y^2}{\sigma^2})e^{-\frac{x^2+y^2}{2\sigma^2}}$ |
| Hough transform | $\rho = x\cos\theta + y\sin\theta$ |
| RANSAC iterations | $N = \frac{\log(1-p)}{\log(1-(1-e)^s)}$ |
| Convolution output size | $W_{out} = \frac{W_{in} - W_f}{s} + 1$ |
| Cross-entropy loss | $C = -\sum_i [y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| VAE loss | $\mathcal{L} = \mathbb{E}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\|p(z))$ |
| GAN objective | $\min_G\max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ |
| Diffusion forward | $q(x_t|x_{t-1}) = \mathcal{N}(x_t\|\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ |

---

## Recommended Reading

- Szeliski, "Computer Vision: Algorithms and Applications" (Chapters 3, 4)
- Shah, "Computer Vision" (Chapter 2)
- Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (IJCV 2004)
- Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection" (CVPR 2005)
- Goodfellow et al., "Generative Adversarial Nets" (2014)
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
