"""
Binary Perceptron Trainer for Tiny Tapeout 4-Stage Pipelined Perceptron
========================================================================
Trains a binary perceptron on your dataset, then generates the bit pattern
to load into the chip via the input pins.

Mathematical model:
    y = 1  if  sum(w[i] * x[i]) >= theta   for i = 0..7
    y = 0  otherwise

Since all values are binary:
    w[i] * x[i] = w[i] AND x[i]
    sum = count of bits where both input and weight are 1
"""

import numpy as np


# ─────────────────────────────────────────────
# 1. TRAINING
# ─────────────────────────────────────────────

def train_perceptron(X_train, y_train, learning_rate=0.1, epochs=100):
    """
    Train a binary perceptron on 8-bit input vectors.

    Args:
        X_train: numpy array of shape (n_samples, 8), values in {0, 1}
        y_train: numpy array of shape (n_samples,), values in {0, 1}
        learning_rate: float, step size for weight updates
        epochs: int, number of training passes

    Returns:
        binary_weights: numpy array of shape (8,), values in {0, 1}
        threshold: int, firing threshold (0-8)
        history: list of accuracy per epoch
    """
    weights = np.zeros(8, dtype=float)
    history = []

    for epoch in range(epochs):
        correct = 0
        for x, y in zip(X_train, y_train):
            weighted_sum = np.dot(weights, x)
            threshold = 4  # midpoint default
            prediction = 1 if weighted_sum >= threshold else 0
            error = y - prediction
            weights += learning_rate * error * x
            if prediction == y:
                correct += 1

        accuracy = correct / len(y_train)
        history.append(accuracy)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Accuracy: {accuracy:.2%} | "
                  f"Weights: {weights.round(2)}")

    # Binarise weights — positive = 1, zero or negative = 0
    binary_weights = (weights > 0).astype(int)

    # Find optimal threshold by trying all values 0-8
    best_threshold = 0
    best_accuracy = 0
    for t in range(9):
        preds = [(1 if np.dot(binary_weights, x) >= t else 0) for x in X_train]
        acc = np.mean([p == y for p, y in zip(preds, y_train)])
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    return binary_weights, best_threshold, history


# ─────────────────────────────────────────────
# 2. CHIP LOADING INSTRUCTIONS
# ─────────────────────────────────────────────

def generate_load_instructions(binary_weights, threshold):
    """
    Generate the bit patterns to load into the chip.

    On the chip:
    - Weight registers share IN0-IN7 with input data
    - Threshold registers use IN0-IN3
    - Both load on every clock edge

    Args:
        binary_weights: numpy array of shape (8,), values in {0, 1}
        threshold: int, firing threshold (0-8)

    Returns:
        dict with loading instructions
    """
    weight_byte = sum(int(w) << i for i, w in enumerate(binary_weights))
    threshold_nibble = threshold & 0x0F

    print("\n" + "="*50)
    print("CHIP LOADING INSTRUCTIONS")
    print("="*50)
    print(f"\nTrained weights:  {binary_weights}")
    print(f"Threshold:        {threshold}")
    print(f"\nWeight byte:      {bin(weight_byte)} ({weight_byte})")
    print(f"Threshold nibble: {bin(threshold_nibble)} ({threshold_nibble})")

    print("\n--- DIP Switch Settings ---")
    print("Step 1: Set switches to load weights:")
    for i in range(8):
        state = "ON " if binary_weights[i] else "OFF"
        print(f"  Switch {i+1} (IN{i}): {state}  [weight bit {i} = {binary_weights[i]}]")

    print(f"\nStep 2: Clock once to latch weights")

    print(f"\nStep 3: Set switches for threshold {threshold} = {bin(threshold_nibble)}:")
    for i in range(4):
        bit = (threshold_nibble >> i) & 1
        state = "ON " if bit else "OFF"
        print(f"  Switch {i+1} (IN{i}): {state}  [threshold bit {i} = {bit}]")
    print(f"  Switches 5-8: OFF")

    print(f"\nStep 4: Clock once to latch threshold")
    print(f"\nStep 5: Set switches to your input pattern and clock 4 times")
    print(f"        OUT7 (fire LED) = 1 means neuron fired!")

    return {
        "weight_byte": weight_byte,
        "threshold": threshold,
        "binary_weights": binary_weights.tolist()
    }


# ─────────────────────────────────────────────
# 3. INFERENCE SIMULATION
# ─────────────────────────────────────────────

def simulate_chip(binary_weights, threshold, test_inputs):
    """
    Simulate the chip's behaviour for given inputs.

    Args:
        binary_weights: numpy array of shape (8,)
        threshold: int
        test_inputs: numpy array of shape (n, 8)

    Returns:
        list of (input, sum, fire) tuples
    """
    print("\n" + "="*50)
    print("CHIP SIMULATION")
    print("="*50)
    print(f"Weights:   {binary_weights}")
    print(f"Threshold: {threshold}")
    print(f"\n{'Input':<20} {'Sum':>5} {'Fire':>6}")
    print("-"*35)

    results = []
    for x in test_inputs:
        weighted_sum = int(np.dot(binary_weights, x))
        fire = 1 if weighted_sum >= threshold else 0
        input_str = "".join(map(str, x.astype(int)))
        print(f"{input_str:<20} {weighted_sum:>5} {'YES' if fire else 'NO':>6}")
        results.append((x, weighted_sum, fire))

    return results


# ─────────────────────────────────────────────
# 4. EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Binary Perceptron Trainer for Tiny Tapeout")
    print("==========================================\n")

    # ── Example 1: Learn to detect patterns with more than 4 active bits ──
    print("Example 1: Detect patterns with > 4 active bits")
    print("-"*50)

    np.random.seed(42)
    X = np.random.randint(0, 2, (200, 8))
    y = (X.sum(axis=1) > 4).astype(int)

    weights, threshold, history = train_perceptron(X, y, epochs=50)
    instructions = generate_load_instructions(weights, threshold)

    # Test on some examples
    test_cases = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],  # 5 ones → should fire
        [1, 1, 1, 0, 0, 0, 0, 0],  # 3 ones → should not fire
        [1, 1, 1, 1, 1, 1, 0, 0],  # 6 ones → should fire
        [0, 0, 0, 0, 0, 0, 0, 0],  # 0 ones → should not fire
        [1, 1, 1, 1, 0, 0, 0, 0],  # 4 ones → borderline
        [1, 1, 1, 1, 1, 1, 1, 1],  # 8 ones → should fire
    ])
    simulate_chip(weights, threshold, test_cases)

    # ── Example 2: Learn to detect a specific pattern ──
    print("\n\nExample 2: Detect if first 4 bits match 1010")
    print("-"*50)

    X2 = np.random.randint(0, 2, (200, 8))
    # Fire if first 4 bits are 1,0,1,0
    y2 = ((X2[:, 0] == 1) & (X2[:, 1] == 0) &
          (X2[:, 2] == 1) & (X2[:, 3] == 0)).astype(int)

    weights2, threshold2, _ = train_perceptron(X2, y2, epochs=100)
    generate_load_instructions(weights2, threshold2)

    test_cases2 = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0],  # exact match → should fire
        [1, 0, 1, 0, 1, 1, 1, 1],  # match + noise → should fire
        [0, 1, 0, 1, 0, 0, 0, 0],  # inverse → should not fire
        [1, 1, 1, 1, 0, 0, 0, 0],  # wrong pattern → should not fire
    ])
    simulate_chip(weights2, threshold2, test_cases2)
