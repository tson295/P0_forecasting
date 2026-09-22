"""Discretized classification of future mid-price displacement on the WF3 contract.

delta_h(t) = target_mid(t+h) - origin_mid(t), in USD, is mapped to a frozen class
interval, a model predicts class logits, and the argmax class is decoded back to a
displacement and a predicted mid. Split, samples, features and normalization are the
WF3 code's (src/data); only the target, the output heads and the loss change.
"""
HORIZONS = (60, 120, 180)
HORIZON_NAMES = {60: "h1", 120: "h2", 180: "h3"}
