# Handwritten Digit Recognition

A web-based handwritten digit recognizer powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset.

**Live site:** https://dr-wangzhiyuan.github.io/handwritten-digit-recognition/

## How it works

1. Draw a digit (0–9) on the canvas
2. Press **READ** to classify
3. The CNN model runs entirely in your browser via [ONNX Runtime Web](https://onnxruntime.ai/) — no server required

## Model

- Architecture: 3× Conv2D + MaxPooling → Dense(128) → Softmax(10)
- Trained on 60,000 MNIST images for 10 epochs
- **Test accuracy: 99.1%**
- Model size: ~517 KB (ONNX format)

## Training

```bash
python train_model.py
```

Requires TensorFlow, tf2onnx, and onnxruntime.

---

Developed by Dr. Wang Zhiyuan at Singapore University of Social Sciences (SUSS).
