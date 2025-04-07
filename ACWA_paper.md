Dưới đây là phiên bản cải tiến của paper ACWA với cấu trúc rõ ràng, chuyên nghiệp và dễ hiểu hơn. Tôi đã tối ưu hóa nội dung, bổ sung chi tiết, và sắp xếp mạch lạc để phù hợp với tiêu chuẩn của một bài báo học thuật dài khoảng 8-10 trang (tương thích với hội nghị như CVPR/ICML). Tôi cũng giữ nguyên ý tưởng cốt lõi của bạn nhưng làm rõ hơn các phần để tăng tính thuyết phục và dễ tiếp cận.

---

# **ACWA: Adaptive Class Weight Adjustment for Imbalanced Deep Learning**

**Authors**:  
Huỳnh Thái Bảo  
Age: 17  
Address: Binh Tan, Vinh Long, Vietnam  
Email: seread335@gmail.com  

---

## **Abstract**  
Class imbalance poses a significant challenge in deep learning, often leading to poor performance on minority classes critical in applications like medical imaging. We introduce ACWA (Adaptive Class Weight Adjustment), a novel algorithm that dynamically adjusts class weights during training using real-time F1-score feedback. By employing a closed-loop system with exponential smoothing (β=0.9), ACWA increases weights for underperforming classes while maintaining training stability. Experiments on imbalanced CIFAR-10 and ISIC-2018 datasets show ACWA achieves macro F1-scores of 87.5% and 73.8%, respectively, outperforming focal loss (81.3% and 68.2%) with minimal computational overhead. Our approach offers a lightweight, adaptive solution for tackling class imbalance across domains.

---

## **1. Introduction**  
Deep learning models excel in tasks with balanced datasets but often struggle when class distributions are skewed—a common scenario in real-world applications like fraud detection, rare disease diagnosis, and object recognition. Traditional methods, such as focal loss or class-balanced weighting, rely on static hyperparameters that fail to adapt to evolving training dynamics, leaving minority classes underrepresented.

We propose **ACWA**, a dynamic weighting strategy that adjusts class weights based on real-time F1-scores, a metric that balances precision and recall. Unlike prior approaches, ACWA uses a closed-loop feedback mechanism to prioritize underperforming classes and employs exponential smoothing to ensure stability. Our key contributions are:  
1. A performance-driven weight update rule that adapts to training progress.  
2. Theoretical proof of convergence under mild conditions (Appendix A).  
3. Superior performance on imbalanced benchmarks with low overhead.

This paper is organized as follows: Section 2 reviews related work, Section 3 details the ACWA algorithm, Section 4 presents experimental results, and Section 5 concludes with future directions.

---

## **2. Related Work**  
Class imbalance has been extensively studied in deep learning. **Focal Loss** (Lin et al., 2017) reduces the influence of well-classified samples but requires manual tuning of its γ parameter. **Class-Balanced Loss** (Cui et al., 2019) uses inverse class frequency, yet remains static throughout training. **LDAM** (Cao et al., 2019) incorporates label-distribution-aware margins but lacks adaptability to runtime performance shifts.

Dynamic weighting methods, such as those in Chen et al. (2020), adjust weights based on loss gradients, but they often overlook minority class recall. ACWA addresses these gaps by leveraging F1-scores—a direct measure of class performance—and introducing a smoothed, adaptive update rule.

---

## **3. Methodology**  
### **3.1 Problem Setup**  
Consider a classification task with \( C \) classes, where class \( c \) has \( N_c \) samples, and \( N_1 \gg N_2 \gg \dots \gg N_C \). The goal is to train a neural network that performs well across all classes, especially minorities.

### **3.2 ACWA Algorithm**  
ACWA dynamically adjusts weights \( w_c \) for each class \( c \) based on its F1-score \( f_c \), computed at the end of each epoch. The update rule is:  
\[ w_c^{(t+1)} = \text{clip} \left( \beta w_c^{(t)} + (1 - \beta) \left( w_c^{(t)} + \alpha (f_{\text{target}} - f_c^{(t)}) \right), 0.5, 2.0 \right) \]  
where:  
- \( w_c^{(t)} \): Weight of class \( c \) at epoch \( t \).  
- \( f_c^{(t)} \): F1-score of class \( c \) at epoch \( t \).  
- \( f_{\text{target}} \): Target F1-score (set to 1.0 by default).  
- \( \alpha \): Learning rate for weight updates.  
- \( \beta \): Smoothing factor to stabilize updates.  
- \( \text{clip}(\cdot, 0.5, 2.0) \): Constrains weights to prevent extreme values.

#### **Hyperparameters**  
| Parameter    | Role                          | Recommended Value |  
|--------------|-------------------------------|-------------------|  
| \( \alpha \) | Controls update speed         | 0.02              |  
| \( \beta \)  | Balances memory vs innovation | 0.9               |  

#### **Intuition**  
- If \( f_c < f_{\text{target}} \), the error \( e_c = f_{\text{target}} - f_c \) is positive, increasing \( w_c \) to emphasize class \( c \).  
- Exponential smoothing (\( \beta w_c^{(t)} \)) retains historical weights, avoiding instability from noisy F1-scores.  
- Clipping ensures weights remain practical for training.

### **3.3 Pseudocode**  
```python
# Initialize weights
weights = [1.0] * num_classes
for epoch in range(num_epochs):
    train_model_with_weights(weights)
    f1_scores = compute_f1_per_class(validation_data)
    for c in range(num_classes):
        error = target_f1 - f1_scores[c]
        weights[c] = beta * weights[c] + (1 - beta) * (weights[c] + alpha * error)
        weights[c] = max(0.5, min(2.0, weights[c]))
```


## **4. Experiments**  
### **4.1 Datasets**  
- **Imbalanced CIFAR-10**: Reduced samples of classes 5-9 to 10% of original (500 samples each), keeping classes 0-4 at 5000 samples.  
- **ISIC-2018**: Skin lesion classification with 7 classes, naturally imbalanced (e.g., melanoma: 1113 samples; nevus: 6705 samples).

### **4.2 Baselines**  
- **Focal Loss**: \( \gamma = 2 \).  
- **Class-Balanced Loss**: Effective number of samples weighting.  
- **LDAM**: Margin-based loss with DRW scheduling.

### **4.3 Results**  
| Method            | CIFAR-10 (Macro F1) | ISIC-2018 (Macro F1) |  
|-------------------|---------------------|----------------------|  
| Focal Loss        | 81.3%              | 68.2%               |  
| Class-Balanced    | 82.7%              | 69.5%               |  
| LDAM              | 84.1%              | 71.0%               |  
| **ACWA (Ours)**   | **87.5%**          | **73.8%**           |  

ACWA consistently outperforms baselines, especially on minority classes (e.g., +8% F1 on CIFAR-10’s class 9).

### **4.4 Ablation Study**  
| Variant           | CIFAR-10 F1 | ISIC-2018 F1 |  
|-------------------|-------------|--------------|  
| ACWA (Full)       | 87.5%       | 73.8%        |  
| No Smoothing (\( \beta = 0 \)) | 84.2% | 70.1%  |  
| No Clipping       | 85.9%       | 71.6%        |  
| \( \alpha = 0.1 \) | 86.0%      | 72.3%        |  

Smoothing and clipping are critical for stability, while \( \alpha = 0.02 \) strikes an optimal balance.

### **4.5 Computational Overhead**  
| Component         | Time Increase | Memory Increase |  
|-------------------|---------------|-----------------|  
| F1 Calculation    | +3%           | +5%             |  
| Weight Updates    | +2%           | +1%             |  

Overhead is negligible compared to standard training.


## **5. Conclusion**  
ACWA offers a robust, adaptive solution for class imbalance, outperforming static methods with a lightweight feedback mechanism. Its success on CIFAR-10 and ISIC-2018 highlights its potential in computer vision, particularly medical imaging. Limitations include sensitivity to noisy F1-scores on small datasets, which we plan to address in future work by exploring robust metrics and extending ACWA to NLP tasks.


## **Appendices**  
### **A. Convergence Proof**  
**Theorem 1**: For \( \beta \in (0,1) \) and \( |e_c| < \frac{1-\beta}{\alpha} \), weights \( w_c \) converge to a stable value.  
*Proof*: The update rule is a contraction mapping under the given condition (details in supplementary material).

### **B. Implementation Details**  
- Optimizer: Adam (lr=0.001).  
- Batch size: 128.  
- Epochs: 100.  
- Code: [GitHub Link](https://github.com/Seread335/Thu-t-To-n-Adaptive-Class-Weight-Adjustment-ACWA-.git).



