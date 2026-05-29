"""
Please modify this file and implement function _train

_train(model, sig, label) - Trains a model in place using MindRove signal and the associated labels
    Parameters:
        model: Torch model (CNNLSTMClassifier)
        sig:   NumPy array (3D: 1 batch, 200 rows, 8 columns)
        label: NumPy array (1D: 200 elements)
    Returns:
        Nothing

Notes:
1. _train currently takes in one batch at a time.
2. Training is always in place. Whether this model is saved as a new file is not dealt with here.
"""

def _train(model, sig, label):
    print("UNIMPLEMENTED ModelManager.train received:")
    print(f"\tSig:   {sig.shape}")
    print(f"\tLabel: {label.shape}")
    print(f"\tContact Justin for more info")
    print()