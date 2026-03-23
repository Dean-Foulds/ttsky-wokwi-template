# Version 2 — Systolic Binary Neural Network Accelerator

## Why we upgraded

Version 1 was a valid binary perceptron — it worked, it synthesised, and it produced correct classifications. But it had a fundamental limitation: it duplicated the same hardware 16 times. Every neuron had its own AND array, its own adder tree, its own comparator. This is expensive in silicon.

Version 2 takes the same mathematical idea and reimplements it as a proper hardware accelerator — one that reuses compute units, implements the correct BNN mathematics, and includes hardware feature engineering. The result is a design that uses less silicon area while being more capable and more aligned with how real AI chips work.

---

## What changed and why

### 1. AND → XNOR (the most important change)

**Version 1 computed:**
```
S[n] = popcount( weights[n] AND input )
```

**Version 2 computes:**
```
S[n] = popcount( XNOR( weights[n], input ) )
```

This is not a minor tweak — it changes what the neuron actually measures.

AND only fires when both bits are 1. It counts "how many features are both present in the input AND marked important by the weight." This means a weight of 0 is meaningless — the neuron ignores that bit entirely regardless of the input.

XNOR fires when both bits match — either both 1, or both 0. It measures bit-level similarity between the weight pattern and the input pattern. A weight of 0 now means "I expect this feature to be absent" and contributes positively when the input bit is also 0.

| w | x | AND | XNOR |
|---|---|-----|------|
| 1 | 1 |  1  |  1   |
| 1 | 0 |  0  |  0   |
| 0 | 1 |  0  |  0   |
| 0 | 0 |  0  |  1   |

XNOR essentially asks "does this input match my learned template?" rather than "how many active features did I find?" This is the standard computation used in virtually all Binary Neural Network research hardware, including chips from IBM, Microsoft Research, and academic groups worldwide.

Hardware cost: XNOR is only marginally larger than AND (one extra inverter), so this improvement is essentially free.

### 2. Parallel compute → systolic engine

**Version 1:**
All 8 bits of the dot product computed simultaneously across 16 neurons — 16 × 8 = 128 AND gates firing at once.

**Version 2:**
One bit of the dot product computed per clock cycle, accumulated over 8 cycles — 16 XNOR operations per cycle, reused 8 times.

The hardware cost drops significantly because the expensive part (the XNOR + accumulate logic) is shared across all 8 cycles rather than replicated. The tradeoff is latency: results update every 8 clock cycles instead of every cycle. At 10 MHz this is 800 nanoseconds — imperceptible for most applications.

This is the same fundamental tradeoff that drives every serious AI accelerator: trade time for area, reuse compute units, process data sequentially rather than spatially.

### 3. Threshold → signed bias

**Version 1:**
```
fire = ( sum >= theta )     theta in {0..8}, unsigned
```

**Version 2:**
```
fire = ( sum + bias >= 0 )  bias in {-8..7}, signed
```

Mathematically equivalent — you can always convert between them. But the bias formulation has a practical advantage: it is the form used by every major machine learning framework (PyTorch, TensorFlow, JAX). When you train a perceptron in Python, the output is weights and biases, not weights and thresholds. Version 2 accepts the trained values directly without any conversion step.

The bias is stored as a signed 5-bit integer, loaded via the same pin interface as weights.

### 4. Hardware feature expansion

Version 2 adds a small combinational block before the neuron array that generates 8 derived features from the 8 raw inputs:

```
feat[0] = x[0]               (raw)
feat[1] = x[1]               (raw)
feat[2] = x[2]               (raw)
feat[3] = x[3]               (raw)
feat[4] = x[4] XOR x[5]     (detects difference between bits 4 and 5)
feat[5] = x[6] XOR x[7]     (detects difference between bits 6 and 7)
feat[6] = x[0] AND x[7]     (detects both ends of the input active)
feat[7] = x[2] XOR x[6]     (detects a diagonal pattern)
```

A single-layer binary perceptron is a linear classifier — it can only learn linearly separable patterns. By adding XOR and AND combinations of input bits as extra features, the neuron can now represent simple non-linear relationships without any additional silicon cost beyond a handful of logic gates.

This technique is known as explicit feature engineering and is used in classical machine learning pipelines to improve the expressive power of simple classifiers.

### 5. Balanced popcount tree

**Version 1** summed the 8 product bits in a linear chain:
```
((((((a+b)+c)+d)+e)+f)+g)+h
```

This creates a long critical timing path — the signal must ripple through 7 additions in sequence.

**Version 2** uses a balanced binary tree:
```
Level 1:  (a+b), (c+d), (e+f), (g+h)      — 4 additions in parallel
Level 2:  (a+b+c+d), (e+f+g+h)             — 2 additions in parallel
Level 3:  (a+b+c+d+e+f+g+h)               — 1 final addition
```

The critical path is now only 3 levels deep instead of 7. This allows the circuit to run at a higher clock frequency and is easier for the ASIC synthesis tool to meet timing constraints. The balanced tree is standard practice in high-performance arithmetic circuit design.

---

## The science behind these improvements

### McCulloch and Pitts (1943)

Both v1 and v2 ultimately implement the same mathematical object: the McCulloch-Pitts neuron, first described in a paper titled "A Logical Calculus of the Ideas Immanent in Nervous Activity" published in the Bulletin of Mathematical Biophysics in 1943.

Warren McCulloch was a neurophysiologist and Walter Pitts was a self-taught mathematician who joined him at the University of Illinois. Their model abstracted the biological neuron into a simple threshold logic unit: sum the weighted inputs, fire if the sum exceeds a threshold. They proved that networks of such units could compute any logical function — a result that prefigured the modern theory of neural computation by four decades.

The same mathematical framework underlies every neural network trained today, from the simplest logistic regression to the largest language models.

### Frank Rosenblatt and the Perceptron (1957)

Frank Rosenblatt at Cornell implemented the McCulloch-Pitts model in hardware as the Perceptron — a physical machine with 400 photocells, variable resistors for weights, and motor-driven update mechanisms. It was the first learning machine: given labelled training examples, it adjusted its own weights to improve classification accuracy.

Rosenblatt's perceptron learning rule is:
```
w[i] = w[i] + learning_rate * (target - output) * x[i]
```

The Python training script included with this project implements exactly this rule, producing binary weights that can be loaded directly into the chip.

### H.T. Kung and the Systolic Array (1978–1982)

The systolic architecture that powers v2 was invented by H.T. Kung (then at Carnegie Mellon University) and Charles Leiserson in a seminal paper titled "Systolic Arrays (for VLSI)" presented at the Infotech State of the Art Conference in 1978, and developed further through the early 1980s.

The key insight was that the bottleneck in VLSI computation is not arithmetic — transistors are fast. The bottleneck is moving data: fetching operands from memory and distributing results. A systolic array eliminates this bottleneck by keeping data moving rhythmically through a grid of simple processing elements, each passing partial results to its neighbour like a heartbeat (from the Greek systole, the contraction of the heart).

Kung's vision was radical for its time: instead of one powerful processor and a lot of memory, build many simple processors and let data flow through them. Each processor does one small operation and passes results along. The computation emerges from the rhythm of the data flow.

This architecture proved highly influential. The principle of reusing simple compute units through rhythmic data movement now underlies:

- Google's Tensor Processing Unit (TPU) — the chip that runs most of Google's AI inference
- NVIDIA's Tensor Core units inside modern GPUs
- Most dedicated neural network accelerators in phones, cameras, and sensors

Version 2 of this chip implements a one-dimensional systolic dot product engine: a row of 16 accumulators, each receiving one XNOR result per clock cycle, processing one bit of the input vector at a time. After 8 cycles the accumulators hold the complete dot products for all 16 neurons simultaneously.

### Binary Neural Networks (2015–present)

The XNOR-popcount computation in v2 was formalised as Binary Neural Networks (BNNs) in the paper "BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" by Matthieu Courbariaux and Yoshua Bengio (2016), building on earlier work by Itay Hubara and colleagues.

BNNs replace 32-bit floating point multiply-accumulate with 1-bit XNOR and integer popcount. The result:

- Energy reduction: 32× less power per operation
- Area reduction: 16× smaller than 8-bit integer, 32× smaller than float
- Speed increase: bitwise operations on modern hardware are much faster than floating point

This makes BNNs particularly attractive for edge AI — deploying neural networks in resource-constrained devices like sensors, cameras, hearing aids, and microcontrollers where battery life and silicon area are the binding constraints.

Your chip implements this principle directly in custom silicon, which is the most efficient possible form of BNN inference.

---

## Version comparison

| Property | V1 | V2 |
|----------|----|----|
| Dot product | AND | XNOR |
| Compute style | Fully parallel | Systolic (1 bit/cycle) |
| Latency | 1 clock cycle | 8 clock cycles |
| Feature input | Raw 8 bits | Expanded (XOR/AND features) |
| Decision | sum >= threshold | sum + bias >= 0 |
| Threshold/bias | 4-bit unsigned | 5-bit signed |
| Popcount | Linear chain | Balanced binary tree |
| Silicon area | ~2000 cells | ~800 cells (estimated) |
| Tile size | 1×2 | 1×2 (with room to spare) |
| Python training | Threshold conversion needed | Direct bias export |
| BNN standard | Non-standard | Industry standard XNOR |

---

## What this chip represents

Both versions implement ideas that have been independently discovered and rediscovered across the history of AI and computer architecture:

- McCulloch and Pitts formalised the neuron in 1943
- Rosenblatt built it in hardware in 1957
- Kung made the hardware efficient with systolic arrays in 1978
- Courbariaux made it trainable at scale with BNNs in 2016
- This chip puts all of it together on 130nm silicon in 2025

The chip is a working demonstration that the fundamental ideas of neural computation and hardware acceleration are simple enough to implement from first principles in a weekend, using only the logic gates and flip-flops available in any digital library — the same building blocks that Rosenblatt used in 1957, just 10,000 times smaller.

---

## Practical use case — anomaly detection with a Raspberry Pi

This section shows how you would connect the chip to a Raspberry Pi and use it to classify sensor data in real time.

### The scenario

Imagine a factory machine that produces an 8-bit status reading every few milliseconds — vibration levels, temperature bands, pressure readings, all encoded as a single byte. You want to detect whether the machine is behaving normally or showing signs of failure, without running a full processor.

The chip handles the classification directly in silicon. The Raspberry Pi loads the trained weights once at startup, then feeds sensor readings continuously. The chip responds within 8 clock cycles — at 10 MHz that is under 1 microsecond per classification.

### Hardware connections

```
Raspberry Pi GPIO        Tiny Tapeout chip pin
─────────────────────    ─────────────────────
GPIO 2  (SDA / data)  →  ui_in[0]
GPIO 3                →  ui_in[1]
GPIO 4                →  ui_in[2]
GPIO 17               →  ui_in[3]
GPIO 27               →  ui_in[4]
GPIO 22               →  ui_in[5]
GPIO 10               →  ui_in[6]
GPIO 9                →  ui_in[7]
GPIO 11 (SCK)         →  clk
GPIO 0                →  uio_in[0]  (mode)
GPIO 5                →  uio_in[1]  (target)
GPIO 6                →  uio_in[2]  (sel bit 0)
GPIO 13               →  uio_in[3]  (sel bit 1)
GPIO 19               →  uio_in[4]  (sel bit 2)
GPIO 26               →  uio_in[5]  (sel bit 3)
GPIO 14               →  rst_n
GPIO 15               →  uo_out[0]  (neuron 0 fire)
GPIO 18               →  uo_out[7]  (neuron 7 fire)
```

### Step 1 — Train weights offline in Python

```python
import numpy as np
from perceptron_trainer import train_perceptron, generate_load_instructions

# Load your sensor data
# X: each row is one 8-bit sensor reading (binarised)
# y: 1 = normal, 0 = anomaly
X_train = np.load("sensor_readings.npy")
y_train = np.load("labels.npy")

# Train neuron 0 to detect normal operation
weights, bias, _ = train_perceptron(X_train, y_train, epochs=200)

print(f"Weights: {bin(int(''.join(map(str, weights)), 2))}")
print(f"Bias:    {bias}")
```

### Step 2 — Load weights onto the chip at startup

```python
import RPi.GPIO as GPIO
import time

# Pin definitions
UI_IN  = [2, 3, 4, 17, 27, 22, 10, 9]   # ui_in[0..7]
CLK    = 11
MODE   = 0    # uio_in[0]
TARGET = 5    # uio_in[1]
SEL    = [6, 13, 19, 26]                  # uio_in[2..5]
RST_N  = 14

GPIO.setmode(GPIO.BCM)
for pin in UI_IN + [CLK, MODE, TARGET, RST_N] + SEL:
    GPIO.setup(pin, GPIO.OUT)

def pulse_clock():
    GPIO.output(CLK, 1)
    time.sleep(0.0001)
    GPIO.output(CLK, 0)
    time.sleep(0.0001)

def set_byte(pins, value):
    for i, pin in enumerate(pins):
        GPIO.output(pin, (value >> i) & 1)

def load_weights(neuron, weight_byte):
    GPIO.output(MODE, 0)       # load mode
    GPIO.output(TARGET, 0)     # target = weights
    set_byte(SEL, neuron)      # select neuron
    set_byte(UI_IN, weight_byte)
    pulse_clock()

def load_bias(neuron, bias_value):
    GPIO.output(MODE, 0)       # load mode
    GPIO.output(TARGET, 1)     # target = bias
    set_byte(SEL, neuron)      # select neuron
    # encode signed bias: bit 3 = sign, bits 3:0 = magnitude
    encoded = (bias_value & 0x0F) | ((1 if bias_value < 0 else 0) << 4)
    set_byte(UI_IN, encoded)
    pulse_clock()

# Reset chip
GPIO.output(RST_N, 0)
time.sleep(0.001)
GPIO.output(RST_N, 1)

# Load trained weights into neuron 0
weight_byte = 0b11001101   # example trained weights
load_weights(0, weight_byte)
load_bias(0, -2)            # example trained bias

print("Weights loaded. Ready for inference.")
```

### Step 3 — Run inference continuously

```python
import RPi.GPIO as GPIO

FIRE_PIN = 15   # uo_out[0] = neuron 0 fire signal
GPIO.setup(FIRE_PIN, GPIO.IN)

def read_sensor():
    # Replace this with your actual sensor reading
    # Returns an integer 0-255
    return 0b10110101

def classify(sensor_byte):
    GPIO.output(MODE, 1)       # infer mode
    set_byte(UI_IN, sensor_byte)
    # Wait 8 clock cycles for result
    for _ in range(8):
        pulse_clock()
    return GPIO.input(FIRE_PIN)

# Main loop
print("Starting inference loop...")
while True:
    reading = read_sensor()
    fire = classify(reading)

    if fire:
        print(f"Input {bin(reading)}: NORMAL")
    else:
        print(f"Input {bin(reading)}: ANOMALY DETECTED")

    time.sleep(0.01)   # 100 Hz classification rate
```

### What this achieves

At 10 kHz clock speed the chip classifies 1250 sensor readings per second. At 10 MHz it classifies over 1 million per second. The Raspberry Pi is free to do other work — logging, networking, display — while the chip handles classification entirely in hardware with no CPU cycles consumed.

This is the core value proposition of a neural accelerator: offload the repetitive inference computation to dedicated silicon, freeing the host processor for higher-level tasks. The same principle is used in every phone that recognises your face, every camera that detects motion, and every hearing aid that filters background noise.

### Extending to all 16 neurons

Each of the 16 neurons can be trained to recognise a different pattern. For example:

| Neuron | Trained to detect |
|--------|------------------|
| 0 | Normal operation |
| 1 | High temperature warning |
| 2 | Vibration spike |
| 3 | Pressure drop |
| 4 | Combined fault signature |
| 5–15 | Additional conditions |

Read all 16 outputs simultaneously from `uo_out[7:0]` and `uio_out[7:0]` to get a complete 16-bit classification vector in a single inference pass.
