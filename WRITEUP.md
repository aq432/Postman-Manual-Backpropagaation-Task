# Technical Write-up: Tasks 1.1 – 1.3

## 1. Network Architecture & Formulation (Deliverable 1.1)

The network is a 2-layer Multilayer Perceptron (MLP) mapping an input batch $X \in \mathbb{R}^{N \times D_{\text{in}}}$ to output class probabilities $P \in \mathbb{R}^{N \times D_{\text{out}}}$ across a hidden layer of size $H$[cite: 1]. The architecture relies strictly on matrix multiplications and vector additions broadcast across the batch dimension[cite: 1].

### Dimension Flow
* **Input Data ($X$):** Shape $(N, D_{\text{in}})$, where $N$ is the batch size and $D_{\text{in}}$ is the number of input features.
* **Layer 1 Weights ($W_1$) & Bias ($b_1$):** 
  * $W_1 \in \mathbb{R}^{D_{\text{in}} \times H}$
  * $b_1 \in \mathbb{R}^{1 \times H}$
* **Hidden Pre-activations ($Z_1$) & Activations ($A_1$):** Shape $(N, H)$.
* **Layer 2 Weights ($W_2$) & Bias ($b_2$):** 
  * $W_2 \in \mathbb{R}^{H \times D_{\text{out}}}$
  * $b_2 \in \mathbb{R}^{1 \times D_{\text{out}}}$
* **Output Logits ($Z_2$) & Probabilities ($P$):** Shape $(N, D_{\text{out}})$.

### Parameter Initialization
Weights cannot be initialized to zero, as symmetric weights cause every neuron in a layer to compute identical gradients and updates during backpropagation. To maintain activation variance across layers and prevent vanishing or exploding gradients, weights are initialized from a scaled Gaussian distribution using He/Xavier scaling:

$$W_1 \sim \mathcal{N}\left(0, \, \frac{2}{D_{\text{in}}}\right), \quad W_2 \sim \mathcal{N}\left(0, \, \frac{2}{H}\right)$$

Biases are safely initialized to zeros ($b_1 = \mathbf{0}_{1 \times H}$, $b_2 = \mathbf{0}_{1 \times D_{\text{out}}}$) because randomized weights are sufficient to break symmetry.

---

## 2. Forward Pass Mechanics & Loss (Deliverable 1.2)

The forward propagation pipeline computes intermediate representations sequentially, caching them for gradient evaluation[cite: 1]:

$$X \xrightarrow{\text{Linear}} Z_1 \xrightarrow{\text{ReLU}} A_1 \xrightarrow{\text{Linear}} Z_2 \xrightarrow{\text{Softmax}} P \xrightarrow{\text{Cross-Entropy}} L$$

### Linear Layers & Non-Linearity
The first layer applies an affine transformation followed by a Rectified Linear Unit (ReLU) activation[cite: 1]:

$$Z_1 = X W_1 + b_1$$
$$A_1 = \max(0, Z_1)$$

The hidden representation $A_1$ is then projected to raw output logits:

$$Z_2 = A_1 W_2 + b_2$$

### Numerically Stable Softmax
Softmax maps unconstrained logits $Z_2 \in \mathbb{R}^{N \times D_{\text{out}}}$ into normalized probability distributions where each row sums to $1.0$[cite: 1]. Directly calculating $e^{Z_{2, ij}}$ risks floating-point overflow (`inf`) when logits are large. To maintain numerical stability, each row is shifted by its row-wise maximum:

$$C_i = \max_j (Z_{2, ij})$$
$$P_{ik} = \frac{e^{Z_{2, ik} - C_i}}{\sum_{j=1}^{D_{\text{out}}} e^{Z_{2, ij} - C_i}}$$

Because $\frac{e^{z_k - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_k}e^{-c}}{e^{-c}\sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$, shifting by a constant does not alter the resulting probabilities.

### Categorical Cross-Entropy Loss
For true class indices $y \in \{0, \dots, D_{\text{out}}-1\}^N$, the batch loss measures the negative log-likelihood assigned to the target classes[cite: 1]:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log(P_{i, y_i} + \epsilon)$$

A small smoothing factor ($\epsilon = 10^{-12}$) prevents numerical instability ($\log(0) \to -\infty$).

---

## 3. Backward Pass Derivations (Deliverable 1.3)

Gradients are propagated backwards through the computation graph via the chain rule, as detailed in Nielsen's matrix formulation[cite: 1].

### Step 1: Output Gradient ($\frac{\partial L}{\partial Z_2}$)
Differentiating the Categorical Cross-Entropy loss with respect to the input logits $Z_2$ combines the derivatives of both the log-loss and the Softmax function, simplifying to the difference between predictions and targets[cite: 1]:

$$\frac{\partial L}{\partial Z_2} = \frac{1}{N} (P - Y)$$

Where $Y \in \mathbb{R}^{N \times D_{\text{out}}}$ is the one-hot encoded matrix of ground-truth labels ($Y_{i, j} = 1$ if $j = y_i$, else $0$). In implementation, this is achieved by taking a copy of $P$, subtracting $1.0$ at the target indices $(i, y_i)$, and scaling by $\frac{1}{N}$.

### Step 2: Layer 2 Parameter Gradients ($\frac{\partial L}{\partial W_2}, \frac{\partial L}{\partial b_2}$)
Using the incoming error $dZ_2 = \frac{\partial L}{\partial Z_2}$ and cached activations $A_1$:

$$\frac{\partial L}{\partial W_2} = A_1^T \cdot dZ_2 \quad \in \mathbb{R}^{H \times D_{\text{out}}}$$
$$\frac{\partial L}{\partial b_2} = \sum_{i=1}^N (dZ_2)_{i, :} \quad \in \mathbb{R}^{1 \times D_{\text{out}}}$$

### Step 3: Backpropagation Through Hidden Layer ($dA_1, dZ_1$)
The gradient flowing into the hidden activations is computed by projecting back through $W_2$:

$$\frac{\partial L}{\partial A_1} = dZ_2 \cdot W_2^T \quad \in \mathbb{R}^{N \times H}$$

Because $\text{ReLU}(z) = \max(0, z)$, its derivative is piecewise:
$$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

Applying the element-wise product ($\odot$):

$$\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbb{I}(Z_1 > 0) \quad \in \mathbb{R}^{N \times H}$$

### Step 4: Layer 1 Parameter Gradients ($\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial b_1}$)
Using $dZ_1 = \frac{\partial L}{\partial Z_1}$ and the cached input data $X$:

$$\frac{\partial L}{\partial W_1} = X^T \cdot dZ_1 \quad \in \mathbb{R}^{D_{\text{in}} \times H}$$
$$\frac{\partial L}{\partial b_1} = \sum_{i=1}^N (dZ_1)_{i, :} \quad \in \mathbb{R}^{1 \times H}$$

---

## 4. Engineering & Debugging Insights

Tracking false starts and shape mismatches directly in the commit history provides verifiable documentation of the debugging process[cite: 2]:

* **Dimension Inversion in Layer 2:** Initial attempts multiplied $X$ directly with $W_2$ instead of passing the activated hidden states $A_1$, generating matrix dimension mismatches (`ValueError: matmul mismatch (size 4 is different from 3)`). Tracing matrix shapes explicitly ($(N, D_{\text{in}}) \to (N, H) \to (N, D_{\text{out}})$) resolved the transformation flow.
* **Vector Indexing Bug:** Accessing target probabilities using `Prob[np.array(N), y]` caused an `IndexError` due to scalar broadcasting rather than 2D coordinate indexing. Replacing this with `np.arange(N)` correctly extracted the true label probabilities: `(0, y[0]), (1, y[1]), ..., (N-1, y[N-1])`.
* **State Caching:** The manual backward pass requires access to quantities calculated during the forward pass ($X, Z_1, A_1, P$). Encapsulating these variables in a forward `cache` dictionary resolved scope retention without polluting global memory.