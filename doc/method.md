# Implementation Guide for Boundary-Wandering Learning (BWL)

This document provides a comprehensive guide for implementing the Boundary-Wandering Learning (BWL) framework, a novel defense against Label Inference Attacks in Vertical Federated Learning (VFL).

## 1. Overview

BWL introduces a new defense paradigm that shifts from "controlling information content" to "reshaping geometric structure". Instead of merely hiding or compressing information, BWL actively dismantles the geometric patterns in the embedding space that attackers exploit.

It achieves this through two core, synergistic components:
1.  **Boundary-Wandering Loss ($L_{bw}$):** An innovative loss function that enforces "intra-class repulsion," forcing embeddings of the same class to be far apart in angular distance. This breaks the link between embedding similarity and label identity.
2.  **Shadow Model Proxy (SMP) Architecture:** A novel architecture that decouples the privacy-preserving operations from the primary task learning. It uses a public-facing "obfuscated space" for federal interaction and a local "fidelity space" for high-accuracy prediction, resolving the classic privacy-utility trade-off.

## 2. System Setup & Prerequisites

-   **VFL Setting:** A standard two-party VFL setup.
    -   **Party A (Attacker):** The passive party holding features $X_A$. This party is "honest-but-curious."
    -   **Party B (Defender):** The active party holding features $X_B$ and the private labels $Y$.
    -   **Server:** A central coordinator that hosts the top model, computes the global loss, and distributes gradients. (Often, the active party also acts as the server).
-   **Model Structure:**
    -   Party A has a bottom model $M_A$.
    -   Party B (Defender) implements the special SMP architecture (see Section 3.1).
    -   The server has a top model $M_{top}$.

## 3. Core Component Implementation

### 3.1. Shadow Model Proxy (SMP) Architecture on Defender Side

The defender (Party B) needs to set up a specific local architecture before training begins.

#### Step 1: Feature Partitioning

The defender must partition their local features $X_B$ into two disjoint sets:
-   **Public Features ($x_{public}$):** A subset of features with a weaker correlation to the labels. These will be used for federal interaction.
-   **Private Features ($x_{private}$):** A subset of features with a strong correlation to the labels. These will be kept strictly local and never shared.

**Implementation Notes:**
-   This partitioning is a pre-processing step performed locally by the defender.
-   Standard feature selection techniques can be used for this purpose, such as calculating Mutual Information scores between each feature and the labels, or using feature importance derived from a pre-trained local model (e.g., Gradient Boosting).
-   The ratio of private to public features is a key hyperparameter. A recommended starting point is to designate 10-30% of the most informative features as private.

#### Step 2: Model Instantiation

The defender needs to instantiate three distinct local models:
1.  **Shadow Model ($M_{shadow}$):** A bottom model that takes $x_{public}$ as input and produces the public embedding $E_{shadow}$. This is the only model whose output is sent to the server.
2.  **Private Model ($M_{private}$):** A bottom model that takes $x_{private}$ as input and produces the private embedding $E_{private}$. This model and its output are strictly local.
3.  **Local Head ($M_{local\_head}$):** A local classification head. It is used exclusively within the defender's local training loop to update the Private Model.

### 3.2. Boundary-Wandering Loss ($L_{bw}$)

This loss is calculated by the server (or the active party acting as the server) using the public embeddings received from the defender ($E_{shadow}$).

**Objective:** To maximize the angular distance between embeddings of the same class. In a minimization-based optimization framework, this is achieved by minimizing the cosine similarity between same-class embedding pairs.

**Formula:**
$$L_{bw} = \frac{1}{N_{pairs}} \sum_{\text{class } k} \sum_{i, j \in \mathcal{C}_k, i \neq j} \left( \frac{z_i}{\|z_i\|} \cdot \frac{z_j}{\|z_j\|} \right)$$
where $z_i, z_j$ are shadow embeddings ($E_{shadow}$) from the same class $\mathcal{C}_k$ within a training batch.

**Implementation Notes:**
-   For each batch, first normalize all shadow embeddings to the unit hypersphere.
-   Group the normalized embeddings by their corresponding labels.
-   For each group (class) containing more than one embedding, calculate the sum of cosine similarities (dot products of normalized vectors) for all unique pairs.
-   Sum these values across all classes and normalize by the total number of pairs computed.
-   It is important to use a sufficiently large batch size to ensure that batches frequently contain multiple samples from the same class, making the loss calculation effective.

## 4. The Complete Training Workflow (Single Batch)

The training process for a single batch is divided into three distinct phases.

#### **Phase 1: Forward Pass**

1.  **Party A (Attacker):**
    -   Computes its embedding: $E_A = M_A(x_A)$.
    -   Sends $E_A$ to the server.
2.  **Party B (Defender):**
    -   Splits its batch features: $x_B \rightarrow (x_{public}, x_{private})$.
    -   Computes public embedding: $E_{shadow} = M_{shadow}(x_{public})$.
    -   Computes private embedding: $E_{private} = M_{private}(x_{private})$.
    -   Sends **only $E_{shadow}$** to the server.
3.  **Server (Top Model):**
    -   Receives $E_A$ and $E_{shadow}$.
    -   Fuses them into a single vector: $E_{fused\_top} = \text{Concat}(E_A, E_{shadow})$.
    -   Calculates the final prediction: $p = \text{Softmax}(M_{top}(E_{fused\_top}))$.

#### **Phase 2: Loss Calculation on Server**

1.  **Prediction Loss:** The server calculates the standard cross-entropy loss based on the model's prediction: $L_{pred} = \text{CrossEntropy}(p, y)$.
2.  **Wandering Loss:** The server calculates the Boundary-Wandering loss **exclusively on the defender's public embeddings**, $E_{shadow}$, as described in Section 3.2.
3.  **Total Federal Loss:** The server combines these two losses using a hyperparameter $\alpha$: $L_{total} = L_{pred} + \alpha \cdot L_{bw}$.

#### **Phase 3: Backward Pass & Decoupled Updates**

This phase ensures that the disruptive privacy loss only affects the public-facing model.

1.  **Server:**
    -   Computes the gradients of $L_{total}$ with respect to the top model's inputs: $\nabla E_A$ and $\nabla E_{shadow}$.
    -   Sends $\nabla E_A$ back to Party A and $\nabla E_{shadow}$ back to Party B.
2.  **Party A (Attacker):**
    -   Receives $\nabla E_A$.
    -   Performs backpropagation to update its bottom model $M_A$.
3.  **Party B (Defender) - The Decoupled Update:** The defender performs two independent updates in parallel.
    -   **Track 1 (Update Shadow Model):**
        -   Receives $\nabla E_{shadow}$ from the server.
        -   Uses this gradient to perform backpropagation and update the weights of the **Shadow Model $M_{shadow}$ only**.
    -   **Track 2 (Update Private Model):**
        -   **Isolate the gradient flow:** The defender must detach $E_{shadow}$ from the federal computation graph. This critical step prevents gradients from this local loop from flowing back into the shadow model.
        -   Fuse the detached public embedding with the private one: $E_{fused\_local} = \text{Concat}(E_{shadow}.\text{detach}(), E_{private})$.
        -   Pass the fused vector through the local head to get a local prediction: $p_{local} = M_{local\_head}(E_{fused\_local})$.
        -   Calculate a local-only cross-entropy loss: $L_{local} = \text{CrossEntropy}(p_{local}, y)$.
        -   Perform backpropagation from $L_{local}$ to update the weights of the **Private Model $M_{private}$ and the Local Head $M_{local\_head}$ only**.

By following this procedure, the defender ensures that the privacy-enhancing but disruptive $L_{bw}$ only influences the public-facing $M_{shadow}$, while the performance-critical $M_{private}$ is trained cleanly using a standard objective.

## 5. Hyperparameters and Configuration

Key parameters to configure and tune during experimentation:

-   `alpha` ($\alpha$): This is the most crucial hyperparameter. It controls the strength of the privacy defense.
    -   `alpha = 0` corresponds to no defense.
    -   Larger values of `alpha` increase the geometric obfuscation but may impact model utility if not properly balanced.
    -   It should be tuned by evaluating the trade-off between attack success rate and primary task accuracy on a validation set.
-   **Feature Split Ratio:** The percentage of features allocated to $x_{private}$ vs. $x_{public}$. The optimal ratio may vary across different datasets.
-   **Batch Size:** A larger batch size is recommended to improve the stability and effectiveness of the Boundary-Wandering Loss.
-   **Learning Rates:** Separate learning rates can be used for the shadow and private models if necessary.