# Model Weights

Trained model weights are not stored in this repository due to file size.

## Download Instructions

### Custom CNN (`cnn_currency.h5`)
After training via `notebooks/currency_cnn.ipynb`, save the model:
```python
model.save("models/cnn_currency.h5")
```

### ResNet34 FastAI (`resnet34_currency.pkl`)
After training via `notebooks/currency_resnet34.ipynb`, export:
```python
learn.export("models/resnet34_currency.pkl")
```

## Performance Summary

| Model | Val Accuracy | File Size |
|---|---|---|
| Custom CNN | 83.33% | ~1.8 MB |
| ResNet34 (FastAI) | **95.24%** | ~85 MB |

Use the ResNet34 model for production inference.
