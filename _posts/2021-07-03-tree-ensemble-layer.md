---
layout: post
title: An Explainer on Tree Ensemble Layer
comments: true
---
*This article was originally published on medium on 03-Jul-2021 at this [link](https://medium.com/towards-data-science/an-explainer-on-tree-ensemble-layer-b7e445621a3f)*


Both Neural Networks and Decision Trees have worked exceptionally on multiple machine learning problems. What if we could get the best of both in one model? This is what the [Google Research](https://github.com/google-research/google-research) team tried to seek in their paper titled [“The Tree Ensemble Layer: Differentiability meets Conditional Computation”](https://arxiv.org/abs/2002.07772).

Trees support *conditional computation* i.e. they are able to route each sample through a small number of nodes. This can lead to both performance benefits, enhanced statistical properties, and helps in interpretability. But the performance of a tree is heavily dependent on feature engineering since they lack a mechanism for representation learning. This is where neural networks have excelled, especially in image and speech recognition problems albeit lacking support for *conditional computation*. In this paper, a layer of additive differentiable decision trees, *Tree Ensemble Layer (TEL)*, for neural networks have been proposed. This layer can be inserted anywhere in a neural network and is trainable by standard gradient-based optimization methods (e.g. SGD).

## Differentiable Decision Tree

![Soft Routing, https://arxiv.org/abs/2002.07772](/assets/2021-07-03-fig1.png)


In classical decision trees, each sample is directed to exactly one direction at every node (*hard routing*), which introduces a discontinuity in the loss function. Since continuous optimization techniques can’t be applied, a greedy approach is taken to build a tree. Soft trees are a variant of decision trees that perform *soft routing* i.e. route each sample to both left and right with different proportions. In this structure, the loss function is differentiable, and gradient-based optimization methods can be used.

But how do we model the probability that a particular sample, x, reaches a leaf node, l? To reach node $l$, sample $x$ has to visit all of its ancestor nodes. And at each node, it will be sent to both left and right subtree with a certain probability. The total probability of reaching node l is then the joint probability of moving to subtree containing $l$ at each node.

$$
\begin{aligned}
	P({x \rightarrow l}) &= \Pi_{i \in A(l)} r_{i,l} (x), \\
	r_{i,l} (x) &= S(\left< x, w_i \right>) ^ {1[l \swarrow i]} \left( 1 - S(\left< x, w_i \right>)\right) ^ {1[i \searrow l]} \\
	\text{where} ~A(l) & \text{: all ancestor nodes of }l \\
	S() & \text{: activation function} \\
	1[i \swarrow l] & \text{: indicator function that $l$ is in left subtree}
\end{aligned}
$$


The probability of sample x to reach node l
$r_i,l(x)$ is the probability that at node $i$, sample $x$ with will move toward subtree containing leaf $l$. The logistic function is a popular choice of activation function but it doesn’t exactly give 0 or 1. This means all nodes would need to be calculated, computation for which increases exponentially with tree depth. Following **continuous** and **differentiable** activation function is proposed to get around this:

$$
S(t)= 
\begin{cases}
    0, & \text{if } t < -\gamma / 2 \\
    \frac{-2}{\gamma^3} t^3 + \frac{3}{2\gamma}t + \frac{1}{2},& \text{if } -\gamma/2 \leq t < \gamma/2\\
    1, & \text{if } t \geq \gamma / 2
\end{cases}
$$

Choice of $\gamma$ controls the number of samples hard routed to 0 or 1. The function closely approximates the logistic function.

![Smooth Step vs Logistic Function (1 / (1 + e^(-6t))) : https://arxiv.org/abs/2002.07772](/assets/2021-07-03-fig2.png)

## Conditional Computation
To optimize TEL, first-order optimization methods such as variants of Stochastic Gradient Descent (SGD) can be used. Computation of gradient increases exponentially with tree depth and this has been a major bottleneck. Efficient forward and backward propagation is developed by exploiting sparsity in the activation function defined above and its gradient.

**Conditional Forward Pass**: Prior to computing gradient, a forward pass over the tree is required. Here sparsity is leveraged by dropping any subtree, and ensuing computation, where the activation function is hard routed to 0.

**Conditional Backward Pass**: Backward pass traverses through the tree to update each node with the gradient. A critical observation of note is that the gradient is 0 for any node where the value is hard routed to 0 or 1. The number of nodes to be visited in the backward pass is hence even lower than what was visited in the forward pass. This is leveraged by creating a *fractional tree* with a reduced number of nodes, leading to faster computation.

Results of experiments from the paper indicate that the TEL achieves competitive performance to Gradient Boosted Decision Trees (GBDT) and Dense Neural Network (DNN) layers while leading to significantly more compact models. This paper is really interesting as it takes a new approach to bring interpretability into neural network models. Each hidden layer in DNN learns a representation and bringing the TEL layer before the output layer will definitely help in understanding the direct link of that representation with output.

## References
1. Hazimeh, H., Ponomareva, N., Mol, P., Tan, Z., & Mazumder, R. (2020, November). The tree ensemble layer: Differentiability meets conditional computation. In International Conference on Machine Learning (pp. 4138–4148). PMLR.