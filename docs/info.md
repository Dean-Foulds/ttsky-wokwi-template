# 16-Neuron Binary Neural Network

## How it works

This project implements a **Binary Neural Network (BNN) inference layer** directly in silicon — 16 independent binary perceptron neurons that each classify the same 8-bit input vector simultaneously, producing a 16-bit output in a single clock cycle.

### The mathematical model

Each neuron `n` computes a weighted sum of binary inputs and compares it to a programmable threshold:

```
y[n] = 1   if   sum( w[n][i] AND x[i] )  >=  theta[n]
y[n] = 0   otherwise

where:
  x[i]    in {0,1}  — input feature bit i  (i = 0..7)
  w[n][i] in {0,1}  — stored weight for neuron n, bit i
  theta[n] in {0..8} — 4-bit programmable threshold for neuron n
  y[n]               — fire signal (classification output)
```

Because all values are binary, multiplication reduces to a logical AND:

```
w[n][i] * x[i]  =  w[n][i] AND x[i]
```

The weighted sum counts how many input bits are both active AND considered important:

```
S[n] = sum( w[n][i] AND x[i] )   for i = 0..7,   S[n] in {0,1,...,8}
```

The neuron fires when this count meets or exceeds its threshold:

```
y[n] = ( S[n] >= theta[n] )
```

### Inside a single neuron

**Stage 1 — AND array (8 gates)**

Each input bit is multiplied by its corresponding weight bit:

```
p[i] = x[i] AND w[n][i]   for i = 0..7
```

**Stage 2 — Adder tree (popcount)**

The 8 product bits are summed using a tree of half-adders and full-adders producing a 4-bit count S[n] between 0 and 8.

```
S[n] = p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7]
```

**Stage 3 — Threshold comparator**

```
y[n] = 1  if  S[n] >= theta[n]
y[n] = 0  otherwise
```

### Why 16 neurons in parallel?

All 16 neurons share the same 8-bit input bus and compute simultaneously. Each neuron has its own independent weight register (8 bits) and threshold register (4 bits), so each can be programmed to recognise a different pattern. The result is a 16-bit output vector answering 16 different yes/no questions about the input in a single clock cycle.

### Why is this AI?

This circuit implements the McCulloch-Pitts neuron (1943) — the mathematical model that founded neural networks. Every modern AI system is built from billions of this computation. Binary Neural Networks (BNNs) are an active area of research for ultra-low-power AI inference at the edge. By binarising weights and activations to {0,1}, multiply-accumulate reduces to AND+popcount — orders of magnitude cheaper in silicon area and power than floating-point arithmetic.

---

## Pin mapping

| Pin | Direction | Function |
|-----|-----------|----------|
| `clk` | in | System clock |
| `rst_n` | in | Active-low reset |
| `ui_in[7:0]` | in | Input features (infer) or load data (load) |
| `uio_in[0]` | in | Mode: 0=load, 1=infer |
| `uio_in[1]` | in | Target: 0=weights, 1=thresholds |
| `uio_in[5:2]` | in | Neuron select 0–15 |
| `uo_out[7:0]` | out | Fire signals neurons 0–7 |
| `uio_out[7:0]` | out | Fire signals neurons 8–15 |

---

## How to test

### Step 1 — Reset

Assert rst_n low then high. All weights clear to 0, thresholds reset to 4.

### Step 2 — Load weights

For each neuron n (0–15):
1. Set uio_in[0]=0 (load mode), uio_in[1]=0 (weights)
2. Set uio_in[5:2]=n (select neuron)
3. Set ui_in[7:0] = weight pattern
4. Pulse clock

### Step 3 — Load thresholds

For each neuron n (0–15):
1. Set uio_in[0]=0 (load mode), uio_in[1]=1 (thresholds)
2. Set uio_in[5:2]=n (select neuron)
3. Set ui_in[3:0] = threshold value (0–8)
4. Pulse clock

### Step 4 — Inference

1. Set uio_in[0]=1 (infer mode)
2. Set ui_in[7:0] = input feature vector
3. Read uo_out[7:0] and uio_out[7:0] for 16-bit result

### Example

```
weights[0]    = 0b11111111  (all features equally weighted)
thresholds[0] = 5           (fire if 5 or more features active)

Input 0b11111100 → S=6, 6>=5 → uo_out[0]=1  (fires)
Input 0b11110000 → S=4, 4<5  → uo_out[0]=0  (silent)
```

### Training weights in Python

```python
from perceptron_trainer import train_perceptron, generate_load_instructions
import numpy as np

X = np.random.randint(0, 2, (200, 8))
y = (X.sum(axis=1) > 4).astype(int)

weights, threshold, _ = train_perceptron(X, y, epochs=100)
generate_load_instructions(weights, threshold)
```

---

## External hardware

No external hardware required. A microcontroller (Raspberry Pi Pico, Arduino) can load trained weights programmatically via the ui_in and uio_in pins.
