# **Linear Ensemble nGPT: Combining Normalized Transformer with Linear Ensemble Attention**

This repository implements a hybrid model that combines two innovative transformer architecture advancements:

1. **nGPT (Normalized Transformer)** from NVIDIA Research [arXiv:2410.01131](https://arxiv.org/abs/2410.01131)
2. **Linear Ensemble Attention Transformer (LET)** a generalization of the Microsoft Research Differential Transformer work [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)

This hybrid architecture aims to leverage the benefits of both approaches:
- **nGPT's** unit-normalized hypersphere representation learning for faster convergence
- **Linear Ensemble Attention's** multiple attention maps for enhanced context modeling

## **Architecture Overview**

### **Key Components**

1. **Normalized Representation Learning (from nGPT)**
   - All vectors are normalized to lie on a unit hypersphere
   - Matrix-vector multiplications represent cosine similarities
   - Updates follow geodesic paths with eigen learning rates
   - No need for weight decay or warmup

2. **Linear Ensemble Attention Mechanism**
   - Uses multiple attention maps with separate query-key projections but shared values
   - Combines attention maps with learned weights
   - Allows the model to capture different attention patterns simultaneously
   - The ensemble approach provides more expressive power to the attention mechanism

3. **Combined Benefits**
   - Faster convergence from normalized representation
   - Enhanced attention representation with multiple attention patterns
   - Better optimization dynamics with hypersphere constraints
   - More expressive attention through the ensemble approach

## **Getting Started**

### **Dependencies**

- **PyTorch**: version 2.0+ recommended for best performance
- **FlashAttention**: from [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
- **Data Preparation**: Follow [nanoGPT repository](https://github.com/karpathy/nanoGPT) instructions for preparing the OpenWebText dataset

### **Running the Code**

To start the training process with defined hyperparameters:

```bash
# Modify problem_name in launcher.sh to select configuration
./launcher.sh
```

Available configurations:
- `LETNGPT_1kctx_10k_lr30e-4`: Linear Ensemble nGPT with 1k context (3 attention maps)
- `LETNGPT_4kctx_10k_lr30e-4`: Linear Ensemble nGPT with 4k context (3 attention maps)
- `LETNGPT_1kctx_10k_lr30e-4_maps5`: Linear Ensemble nGPT with 1k context and 5 attention maps
- `LETTransformer_1kctx_10k_lr30e-4`: Standard Transformer with linear ensemble attention (no nGPT normalization)

### **Implementation Details**

- **Model Architecture**: See `model.py` for the implementation of Linear Ensemble nGPT
- **Training Loop**: See `train.py` for the training procedure with normalization
- **Configuration**: Use `launcher.sh` to customize training parameters
- **Attention Maps**: By default, the model uses 3 attention maps, but this can be adjusted via the `--num_maps` parameter

## **Expected Benefits**

Based on the original papers, this hybrid model may exhibit:

1. **Faster Convergence**:
   - nGPT demonstrated 4-20x faster convergence depending on context length
   - The combined model should maintain this advantage

2. **Enhanced Attention Representation**:
   - Linear Ensemble Attention allows the model to capture multiple attention patterns
   - Each attention map can focus on different aspects of the input
   - Learned weights combine these patterns optimally

3. **Better Generalization**:
   - The multiple attention maps may help prevent overfitting to specific attention patterns
   - The nGPT normalization approach stabilizes training dynamics

## **Customizing the Model**

You can adjust several key parameters to experiment with the model:

1. **Number of Attention Maps**:
   - Modify the `--num_maps` parameter (default: 3)
   - More maps increase representational power but also computational cost

2. **nGPT Features**:
   - Toggle hypersphere normalization with `--use_ngpt` (1 for enabled, 0 for disabled)
   - Adjust eigen learning rates via initialization values in the model

3. **Training Dynamics**:
   - Learning rate, warmup, and decay can be customized in the launcher script
   - nGPT typically performs well without warmup when normalization is enabled

## **Acknowledgements**

This work builds directly upon:

- **nGPT**: [arXiv:2410.01131](https://arxiv.org/abs/2410.01131) by Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg (NVIDIA)
- **Differential Transformer**: [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) by Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, and Furu Wei (Microsoft Research), which includes the Linear Ensemble Attention approach
- **nanoGPT**: by Andrej Karpathy, which serves as the foundational codebase for nGPT

This implementation is a research exploration combining these two innovative approaches to transformer architecture. The codebase is meant to serve as a reference implementation to illustrate the concepts rather than a production-ready solution.

## **Repository Goals**

The main goal of this repository is to explore the potential synergies between two cutting-edge transformer innovations:

1. **Research Exploration**: This is a proof-of-concept implementation demonstrating how these approaches can be combined
2. **Educational Resource**: The code aims to illustrate the concepts of both architectures clearly
3. **Reference Implementation**: While not optimized for production use, it provides a foundation for further exploration

We welcome contributions and discussions about the hybrid approach and potential improvements.
