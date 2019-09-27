# Variational Autoencoder applied to Anomaly Detection

## Benchmark of some classifiers using the autoencoder feature extraction:

> *Feature dimensions reduced from 122 to 16*

1. Vanilla Neural Network:
    - Normal data:
        - __Accuracy__: 99.62%
        - __Loss__: 0.014535
    - Encoded data:
        - __Accuracy__: 97.63%
        - __Loss__: 0.064207

2. Naive Bayes Classifier:
    - Normal data:
        - __Accuracy__: 85.90%
    - Encoded data:
        - __Accuracy__: 87.21%

3. Support Vector Classifier:
    - Normal data:
        - __Accuracy__: 98.49%
    - Encoded data:
        - __Accuracy__: 96.60%

4. Random Forest Classifier:
    - Normal data:
        - __Accuracy__: 99.89%
    - Encoded data:
        - __Accuracy__: 99.74%

5. Threshold Approach:
    - __Accuracy__: 93.51%

