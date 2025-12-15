# NeuralNet - Simple Feedforward Neural Network for Image Classification

This project implements a fully-connected feedforward neural network in Python using **NumPy**. It supports mini-batch stochastic gradient descent, computes categorical cross-entropy loss, and can generate predictions for Kaggle submissions.

---

## Network Architecture

| Layer        | Neurons | Activation |
|-------------|--------|-----------|
| Input       | 784    | -         |
| Hidden      | Configurable | ReLU      |
| Output      | 10     | Softmax   |

---

## Forward Propagation

- **Hidden Layer Pre-activation:**  
  `Z₁ = X · W₁ + b₁`  
- **Hidden Layer Activation (ReLU):**  
  `A₁ = ReLU(Z₁) = max(0, Z₁)`  
- **Output Layer Pre-activation:**  
  `Z₂ = A₁ · W₂ + b₂`  
- **Output Layer Activation (Softmax):**  
  `yᵢ = exp(Z₂ᵢ) / Σⱼ exp(Z₂ⱼ)`

---

## Loss Function

- **Categorical Cross-Entropy Loss:**  
  `L = -(1 / N) Σᵢ log(yᵢ, tᵢ)`  
  - `N` = batch size  
  - `yᵢ, tᵢ` = predicted probability for the true class `tᵢ` of example `i`  

- **Gradient of loss w.r.t. pre-softmax output:**  
  `dZ₂ = (y - one_hot(t)) / N`

---

## Backpropagation

- **Output layer gradients:**  
  `dW₂ = A₁ᵀ · dZ₂`  
  `db₂ = Σ dZ₂`  

- **Hidden layer gradients:**  
  `dA₁ = dZ₂ · W₂ᵀ`  
  `dZ₁ = dA₁ ∘ ReLU'(Z₁)`  
  `dW₁ = Xᵀ · dZ₁`  
  `db₁ = Σ dZ₁`  

> (∘ denotes element-wise multiplication)

---

## Accuracy

`Accuracy = Number of correct predictions / Batch size`

---

## Training

- Mini-batch stochastic gradient descent (random sampling with replacement)  
- **Hyperparameters:**  
  - `BATCHSIZE`: number of examples per batch  
  - `LEARNING_RATE`: step size for gradient updates  
  - `epochs`: number of training iterations  
  - `hidden_layer_size`: number of neurons in hidden layer  

---

## Kaggle Submission

- Output CSV format:  

| ImageId | Label |
|---------|-------|
| 1       | 7     |
| 2       | 2     |
| ...     | ...   |

- Use the function `kaggleTest(testing)` to generate `submission.csv`.

---

## Dependencies

- Python 3.x  
- NumPy  
- Pandas  

Install dependencies with:  
```bash
pip install numpy pandas
