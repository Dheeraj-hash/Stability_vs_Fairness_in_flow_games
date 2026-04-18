# Network Flow Games: Core Stability and the Shapley Value

This repository contains Python scripts used to experiment with cooperative flow games, analyzing network topologies, core stability (deficit), the Shapley value, and the Nucleolus. These experiments support the findings detailed in the accompanying paper on Series-Parallel (SP) balanced networks and topological stability.

## 📂 File Overview

Below is a description of the primary scripts included in this repository:

### Core Experiments
* **`sigma2_deficit.py`**
    Investigates the relationship between the core deficit and the variance in sizes across different min-cuts in a network. 
* **`shapley_nucleolus.py`**
    An extension of `sigma2_deficit.py`. In addition to the core deficit, this script calculates and monitors the L1 distance between the Shapley value and the Nucleolus with respect to the variance of min-cut sizes.
* **`unequal_edge_weight.py`**
    Conducts two primary sensitivity experiments:
    1.  Analyzes how the core deficit and the Shapley-Nucleolus L1 distance change as the variance among edge capacities *within the same min-cut* increases.
    2.  Evaluates how scaling the bottleneck capacity in Funnel topologies affects the Shapley value, the Nucleolus, and the resulting core deficit.
* **`SP_surgery.py`**
    Explores graph modification techniques. It tests whether generic (non-convex/unstable) graphs can be structurally converted into SP-balanced networks to yield Shapley values that successfully lie within the core of the original graph.

### Visualization
* **`topology-visualize.py`**
    Uses the `networkx` library to generate and export visual representations of all the network topologies analyzed in the paper (e.g., Funnel, SP-Balanced, Wheatstone Bridge).

---

## 🚀 How to Run

Ensure you have Python 3 installed along with the necessary dependencies (such as `networkx`, `numpy`, and `matplotlib`). 

To execute any of the experiments or generate the visualizations, run the following command in your terminal:

```bash
python3 <file-name>.py