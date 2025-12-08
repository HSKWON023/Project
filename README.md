# Diffusion Model on MNIST

This project implements a simple **Denoising Diffusion Probabilistic Model (DDPM)** to generate MNIST-like handwritten digits using PyTorch.

The project is based on the content covered in the course **Machine Learning and Applications**, which introduced generative models such as Autoencoders, Variational Autoencoders, GANs, and Diffusion Models.  
Diffusion models represent the current state-of-the-art in image generation and form the basis of modern models such as Stable Diffusion.

---

## 📌 1. Project Goals

The goal of this project is to:

1. Implement the **forward (noising)** and **reverse (denoising)** processes of DDPM.
2. Train a neural network to predict noise at arbitrary timesteps.
3. Generate new MNIST digit samples from pure Gaussian noise.
4. Visualize and analyze the behavior of diffusion-based generative models.

---

## 📂 2. Dataset

We use the MNIST dataset:

- 60,000 training images (28×28 grayscale)
- Loaded automatically via `torchvision.datasets.MNIST`
- Only images are used (labels are ignored) because this is an **unconditional generative model**

Images are scaled from `[0, 1]` to `[-1, 1]` during preprocessing.

---

## ⚙️ 3. Method

### **3.1 Forward Process**

Starting from a clean image \( x_0 \), noise is gradually added:

\[
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t ; \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I)
\]

This can be sampled in a single step from \( x_0 \):

\[
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon
\]

where \( \epsilon \sim \mathcal{N}(0, I) \).

---

### **3.2 Reverse Process**

The reverse process starts from pure noise and iteratively denoises the image:

\[
p_\theta(x_{t-1} \mid x_t)
\]

Instead of modeling this distribution directly, the network learns to **predict the noise** added in the forward process:

\[
L(\theta) = 
\mathbb{E}\left[
  \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2
\right]
\]

This simple loss is one of the reasons diffusion models are stable and easy to train compared to GANs.

---

### **3.3 Network Architecture**

A small **U-Net-like architecture** is used:

- Conv layers with residual blocks  
- Timestep embedding added to each block  
- Skip connections between downsampling and upsampling layers  

Model implemented in `src/model.py`.

---

## 🧪 4. Experiments

### **Training settings**
- Timesteps: 300  
- Beta schedule: linear \( \beta_t \in [10^{-4}, 0.02] \)
- Optimizer: Adam (lr=2e-4)  
- Epochs: 20 (adjustable)  
- Batch size: 128  
- Loss: MSE between predicted noise and true noise  

Checkpoint is saved at:

