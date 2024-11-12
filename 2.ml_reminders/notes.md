# Machine Learning

In this lesson, we’ll introduce machine learning by implementing a two-layer feed-forward neural network to predict handwritten digits. We’ll build it from scratch, using only basic tensor operations.

## Handwritten Digit Recognition Revisited

Previously, we recognized digits by calculating mean images for each digit, creating a tensor of shape `(10, 28, 28)` with pixel values from 0 (black) to 1 (white). We normalized these images by subtracting their mean and dividing by their standard deviation, resulting in values centered around 0. For inference, we compared test images with these templates using matrix-vector multiplication, yielding a score indicating agreement with each template.

## Machine Learning Differences

Machine learning adds two key ideas:
1. **Hierarchical Feature Calculation**: Features are not calculated in a single step but hierarchically, allowing reuse of shared features.
2. **Learned Weights**: Weights are not set using averages but learned from data.

### Hierarchical Features

We can detect shared features like circles or lines using a hidden layer of feature templates. Each template captures a different feature, stored in a matrix \( W_1 \) of shape `(784, H)`. By multiplying test images with \( W_1 \), we produce feature scores that can be further combined for digit classification.

For example, a hidden feature could be an "upper-half-circle," which might correspond to parts of 8 or 0. By hierarchically combining these, we can detect distinct digit patterns with a second weight matrix \( W_2 \) of shape `(10, H)`.

### Non-Linearities

Without non-linearities, hierarchical layers collapse into a single linear transformation, losing the benefits of multiple layers. To prevent this, we apply a non-linear function (like ReLU) between layers, ensuring that the layers don’t collapse. The ReLU function sets all negative values to zero, enabling more complex feature interactions. Our new formulation becomes:

\[ z = W_2 \cdot \text{ReLU}(W_1 \cdot x) \]

Adding biases \( b_1 \) and \( b_2 \) makes this an affine linear transformation. Biases allow the model to account for class imbalances by learning to adjust class scores.

## The Loss Function

Our final network is:

\[ z = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2 \]

To optimize it, we need a loss function that quantifies prediction quality. We map scores to [0, 1] with a sigmoid function, making the output a probability-like value. Using one-hot encoding, the correct output for a digit like 4 would be a vector like `(0, 0, 0, 0, 1, 0, 0, 0, 0, 0)`. We’ll measure prediction accuracy with the **L2 loss**:

\[ L = \frac{1}{N} \sum_i (y_i - \hat{y_i})^2 \]

where \( y_i \) are the true labels, and \( \hat{y_i} \) are the predictions. L2 loss is simple to compute and works well for this introductory model.

## Gradient Descent

Gradient descent optimizes model parameters by adjusting them in the direction that reduces the loss. By calculating the derivative of the loss with respect to each parameter, we identify the optimal direction for reducing error. We update parameters by subtracting a small fraction of this derivative (scaled by the **learning rate** \( \alpha \)).

### Forward Pass Summary

1. **Compute the output**: 
   \[
   z_i = W_2 \cdot \text{ReLU}(W_1 \cdot x_i + b_1) + b_2
   \]
2. **Apply Sigmoid**:
   \[
   y_i = \sigma(z_i)
   \]
3. **Calculate Loss**:
   \[
   L = \frac{1}{N} \sum_i (y_i - \hat{y_i})^2
   \]

### Gradient Descent Update

For each parameter \( W_j \) and \( b_j \):

\[
W_j \leftarrow W_j - \alpha \frac{\partial L}{\partial W_j}
\]

where \( \alpha \) is the learning rate.

## Calculating the Derivatives

Using the chain rule, we trace gradients layer by layer:

1. **L2-loss**: Computes the difference between predictions and true values.
2. **Affine Linear Layer**: Matrix product with input and addition of bias.
3. **ReLU**: Passes only positive values, with a derivative of zero for negative inputs.
4. **Sigmoid**: A smooth function with derivative \( \sigma'(x) = \sigma(x)(1 - \sigma(x)) \).

## Conclusion

Machine learning enhances our naive approach by introducing hierarchical features and automatic parameter adjustment through gradient descent. We used three types of layers:

1. **Affine Linear Layers**: Matrix multiplication with bias addition.
2. **ReLU**: Sets all negative values to zero.
3. **Sigmoid**: Maps values to [0, 1].

Our two-layer classifier structure is: **Affine -> ReLU -> Affine -> Sigmoid**.

Using one-hot encoding, our labels are 10-element vectors where only the correct index is set to 1. We then optimize the parameters using gradient descent and calculate derivatives through each layer.

## Next Steps

This concludes our foundational exploration. In the tutorial, we’ll implement this model for handwritten digit recognition. For a slower-paced, visual introduction, check out tutorials by **3Blue1Brown** or **Sebastian Lague**.

Modern frameworks handle these calculations automatically, so while understanding gradients is useful, it’s not essential for using tools like PyTorch. Let’s dive into the implementation next!
