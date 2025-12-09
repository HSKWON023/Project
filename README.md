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

## ✅ 3. Method

### 3.1 Forward Process

Starting from a clean image \( x_0 \), noise is gradually added according to:

$$
q(x_t \mid x_{t-1})
= \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)
$$

This can also be sampled in closed form from \( x_0 \):

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 \;+\; \sqrt{1 - \bar{\alpha}_t}\, \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I)
$$

---

### 3.2 Reverse Process

The reverse process starts from pure noise and iteratively denoises the image:

$$
p_\theta(x_{t-1} \mid x_t)
$$

Instead of directly modeling this distribution, the model learns to **predict the noise**:

$$
L(\theta)
= \mathbb{E}\left[ \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 \right]
$$

This simple objective makes diffusion models stable and easy to train, compared to GANs.

---

### 3.3 Network Architecture

A small **U-Net-like architecture** is used, with:

- convolutional layers  
- residual blocks  
- timestep embeddings  
- skip connections between encoder and decoder


---

## 🧪 4. Experiments

### **Training Settings**

- **Timesteps:** 300  
- **Beta schedule:** linear  
```math
β_t ∈ [10^{-4}, 0.02]
```
- **Optimizer:** Adam (lr = 2e-4)  
- **Epochs:** 20 (adjustable)  
- **Batch size:** 128  
- **Loss:** MSE between predicted noise and true noise  

---

### **Checkpoints**
Trained model weights are saved at:
./checkpoints/model_latest.pth


---

### **Generated Samples**
Images produced during training are saved under:
./samples/


---

## 5. Results

### Training Samples (Epoch 1 → 20)

The following images show the progression of the model’s denoising ability during training.

#### Epoch 1 (initial reverse diffusion)
![epoch_001](samples/epoch_001.png)

#### Epoch 20 (final training step)
![epoch_020](samples/epoch_020.png)

### Final Generated Samples

After training for 20 epochs on MNIST, the diffusion model successfully generates clear and diverse handwritten digits from pure Gaussian noise.

![generated](samples/generated.png)




---

## 🚀 6. DDIM Fast Sampling (Extension)

In the original DDPM formulation, sampling requires running the reverse process for all diffusion steps (T = 300 in this project). This makes generation relatively slow on CPU.

To accelerate sampling, I additionally implemented **Denoising Diffusion Implicit Models (DDIM)** based on Song et al. (2020).  
DDIM treats the reverse process as a deterministic mapping and allows us to use a **subsequence of timesteps** instead of all 300 steps.

### 6.1 Implementation

- I added a new method `ddim_sample()` in the `Diffusion` class.
- Sampling now supports two modes:

  - **DDPM (baseline)**  
    ```bash
    python sample.py --checkpoint checkpoints/model_latest.pth --method ddpm
    ```
    Uses all 300 steps of the reverse process.

  - **DDIM (fast sampling)**  
    ```bash
    # 50-step DDIM
    python sample.py --checkpoint checkpoints/model_latest.pth --method ddim --num_steps 50

    # 20-step DDIM
    python sample.py --checkpoint checkpoints/model_latest.pth --method ddim --num_steps 20
    ```

  - The same trained model is reused; only the sampling process is changed.

### 6.2 Qualitative Results

Below are example samples generated with different methods:

- **DDPM (300 steps)**  
  ![ddpm_full](samples/generated_ddpm_full.png)

- **DDIM (50 steps)**  
  ![ddim_50](samples/generated_ddim_50.png)

- **DDIM (20 steps)**  
  ![ddim_20](samples/generated_ddim_20.png)

The 50-step DDIM samples are visually almost indistinguishable from the original 300-step DDPM samples, while being significantly faster to generate.  
Even with only 20 steps, the model still produces recognizable digits, demonstrating the efficiency of DDIM-style sampling.

### 6.3 Discussion

- DDIM reuses the same noise-prediction model ε_θ but changes the reverse dynamics.
- By using a deterministic update rule with a carefully chosen timestep subsequence, we can **trade off generation speed and image quality**.
- This experiment shows that diffusion models can generate high-quality MNIST digits with far fewer sampling steps than the original DDPM, which is especially important in resource-limited environments (like CPU-only training).

---

## ▶️ 7. How to Run

### **Install dependencies**
```bash
pip install -r requirements.txt

python train.py

python sample.py --checkpoint ./checkpoints/model_latest.pth

./samples/generated.png
```

## 8. Conclusion

In this project, I implemented a Denoising Diffusion Probabilistic Model (DDPM) from scratch using PyTorch and trained it on the MNIST dataset. The model successfully learned the reverse diffusion process, transforming pure Gaussian noise into realistic handwritten digits.

Through the training progression, we observed that:

- Early-stage samples contained almost no recognizable structure.
- As training proceeded, the model gradually improved its denoising ability.
- By the final epoch, the model generated clear and diverse digit samples.

This demonstrates the core principle of diffusion models: learning to predict noise at each timestep enables the reconstruction of complex data distributions. Despite being trained on a CPU and with limited computational resources, the results show that even a relatively small U-Net architecture can effectively model the MNIST distribution.

Overall, this project helped me better understand:
- The forward and reverse diffusion processes,
- Noise prediction as a learning objective,
- Practical implementation of a state-of-the-art generative model,
- And the stability advantages of diffusion models compared to GANs.

Future improvements could include experimenting with different noise schedules, training with more epochs on GPU, or extending the model to conditional or higher-resolution datasets. Nevertheless, the final results confirm that diffusion models are powerful and robust generative frameworks.










