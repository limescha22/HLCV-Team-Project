**1. Architecture Improvements**
* Increased filters: 8→14→20→24 became 16→24→32→38 (more capacity)
* Added BatchNorm: Stabilizes training and helps convergence
* Used Depthwise Separable Convolutions: Key innovation that saves parameters
    * Splits convolutions into depthwise (spatial) + pointwise (channels)
    * Gets ~85% parameter savings while maintaining capacity
    * Layers 2 and 4 use this technique
* Final count: 9,986 params (was 8,380)

**2. Training Strategy**
* Minimal augmentation (like you suggested!): Only horizontal flip + 10° rotation
* Stronger regularization: weight_decay = 5e-4 (up from 1e-4)
* Higher dropout: 0.4 (up from 0.3)
* More epochs: 150 with patience=35 - best model found at epoch 122
* AdamW optimizer: Better weight decay handling than Adam

**3. Results**
* Best classes: Orange 90.7%, Pineapple 84.6%, Kiwi 83.8%
* Still struggling:
    * Mango 39% (confusing with kiwi/avocado - similar shape/color)
    * Apple 55% (confusing with cherry/strawberries - all round and red)
