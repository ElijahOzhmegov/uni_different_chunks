# Backpropagation
## by Ilia Ozhmegov

## Questions:

First of all, in the following text Stanford notation is used regarding to all used indexes.

1. **If we Backpropagate the Error, how much Error does a Node get from the Previous Errors?**
    Let us assume that the Node $j$ in the layer $l$ has weights $w_{1j}, ..., w_{S_{l+1}j}$ and the errors from the previous layer are $\epsilon_{i}^{(l+1)}$. Then we can get the exact error for the Node $j$:
    $$ \epsilon_j^{(l)} = \sum\limits_{i=1}^{S_{l+1}} w_{ij}^{(l)} \epsilon_i^{(l+1)} $$, 
    $ S_{l+1} $ is the number of Nodes in the layer $ l+1 $.
    
1. **Why can we simply take the Transposed Weight Matrix times the Error Function to calculate the Error for the previous Layer?**
    To answer this question firstly we need to look at the Weight Matrix:

    $${\displaystyle \mathbf {W}^{(l)} =
        {\begin{bmatrix}
            w_{11} & w_{12} & \cdots & w_{1S_l} \\
            w_{21} & w_{22} & \cdots & w_{2S_l} \\
            \vdots & \vdots & \ddots & \vdots \\
            w_{S_{l+1} 1} & w_{S_{l+1} 2} & \cdots & w_{S_{l+1} S_l}
        \end{bmatrix}}}$$

    We can see that weights are placed in certain order. The number of values in each line correspond to the number of nodes $S_l$ in the previous layer, when the number of rows correspond to the number of nodes $S_{l+1}$ in the next layer. It was made, so that we could get the input values $ z_{i}^{l+1}$ as the vector $ \mathbf z^{(l+1)} $ in the following way:
    $$ \mathbf z^{(l+1)} = \mathbf {W}^{(l)} \mathbf {h}^{(l)} + \mathbf{b}^{(l)}$$,
    where $\mathbf {h}^{(l)} $ is an output vector of the previous layer and $\mathbf{b}^{(l)}$ is a bias vector. The order of values in the Weight Matrix can be explained by the feed-forward process to be more accurate by the system of equations. 

    Now, we will come back to the question. As you could notice the formula for getting the certain part of errors for the $l$ layer:
    $$ \epsilon_j^{(l)} = \sum\limits_{i=1}^{S_{l+1}} w_{ij}^{(l)} \epsilon_i^{(l+1)} $$.
    At the same time the dimension of the error vector $ \mathbf{\Epsilon}^{(l+1)} $ is $[S_{l+1} \times 1]$, the dimension of the Weight Matrix $\mathbf {W}^{(l)} $ is $[S_{l+1} \times S_{l}]$, so besides dimension problem during multiplication of the matrix and the vector, we see that wrong weights are multiplicated with values of the error vector. That could two main reasons behind the following formular:
    $$  \mathbf{\Epsilon}^{(l)} = {\mathbf {W}^{(l)}}^T \mathbf{\Epsilon}^{(l+1)} $$.

1. **What is the Goal of Gradient Decent and how do we achieve this?**
    The Goal of Gradient Decent is to minimize the error by tuning the weights
    The gradient decent is one of the numerical optimization methods.

    Firstly, this method requires an activation function that can be derivatived. Then we take a derivative from the loss function by some weight during backpropagation, it allows as to get anti gradient (as we look for a minimum). After that we update the weight matrix in the following way for each layer:
    $$ {\mathbf {W}^{(l)}} = {\mathbf {W}^{(l)}} - \alpha \frac{\partial}{\partial W_{ij}^{(l)} } J(W, b, x, y)$$,
    $$ {\mathbf {b}^{(l)}} = {\mathbf {b}^{(l)}} - \alpha \frac{\partial}{\partial W_{ij}^{(l)} } J(W, b, x, y)$$,
    where $\alpha$ is learning rate and $ J(W, b, x, y) $ is loss (error) function.

1. **Can Backpropagation (theoretically) be used for an infinite number of Layers? Why?**
    No, because in that case we will never reach the output layer during feed-forward process and will not be able to compare the desired value with output values. It then will not be followed by calculating errors and the backpropagation process.

1. **Why do we need a Learning Rate?**
    The learning rate in the Gradient Decent determines the size of the step in the method. We can look at the derivative as on the direction and at the rate as the size of the step. So if the size is too big, we can miss the minimum, however if it is too small there will be too much iterations to reach it.


# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

## by Ilia Ozhmegov

## Questions:
1. **What is the problem with overfitting in neural networks?**
    When we have a large neural network and a tiny dataset for training, the NN cannot ignore the statistical noise in the dataset and as a result we will have an overfitted model. That is what I understood under the term **Overfitting**.

1. **How does dropout work?**
    Firts of all, dropout is a method to solve a problem highlited in the previous question. It allows to drop certain NN units, it temporarily removes certain neurons from the network with its incoming and outgoing connections during training. As the result the NN can ignore the statistical noise.

1. **What effect does dropout have on loss and accuracy?**
    The main effect is that it increases the accuracy and decreases the error compared with the NN without dropout.

1. **How to choose hyperparameter p?**
    The hyperparameter p takes values in the range from 0 to 1, where 1 means no dropout. For input layers, the choice for p depends on the type of input unit. For hidden layers p usually takes values in the range from 0.5 to 0.8, but we should also take in the consideration the number of units in the hidden layer. As if we drop too many units in the layer we will see underfitting, but if we drop not enough we will see overfitting.

1. **What other methods used to prevent neural networks from overfitting do you know?**
    1. Make the model simpler.
    2. As in most cases we use gradient descent we can stop the iterative process before overfitting.
