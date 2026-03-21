# 4-Stage Pipelined Binary Perceptron

## How it works

This project implements a **binary perceptron** — the fundamental building block
of all neural networks — as a 4-stage pipelined digital circuit on a Tiny Tapeout
tile. It classifies an 8-bit input vector every clock cycle after an initial 4-cycle
warmup latency.

### The Perceptron Model

The perceptron computes a weighted sum of binary inputs and compares it to a
threshold to produce a binary classification:

    y = 1  if  S(wi . xi) >= theta
    y = 0  otherwise

Where:
- xi are the 8 input feature bits (0 or 1)
- wi are the 8 stored binary weights (0 or 1)
- theta is the 4-bit programmable threshold
- y is the classification output (fire signal)

Since all values are binary, multiplication reduces to a logical AND:

    wi . xi = wi AND xi

The weighted sum counts how many input bits are both present AND
considered important by their weight:

    S = S(wi AND xi)  for i = 0..7,  S in {0,1,...,8}

### Pipeline Architecture

The circuit is divided into 4 stages separated by D flip-flop pipeline registers.
A new classification result is produced every clock cycle after the initial 4-cycle latency.

#### Stage 1 - Input latch
8 D flip-flops capture the input vector x[7:0] on the rising clock edge.

#### Stage 2 - AND array
8 AND gates compute the element-wise product of latched inputs and stored weights:

    p[i] = x[i] AND w[i]  for i = 0..7

#### Stage 3 - Adder tree
A 3-level Wallace tree of half-adders sums the 8 product bits into a 4-bit count.
Each half-adder computes:

    sum   = A XOR B
    carry = A AND B

#### Stage 4 - Threshold comparator
The 4-bit sum is compared against the stored threshold theta.
If S >= theta the fire signal is asserted.

## How to test

### Step 1 - Load weights
Set ui_in[7:0] to desired weight pattern. Clock once to latch into weight registers.
Example: 11111111 means all features are equally important.

### Step 2 - Load threshold
Set ui_in[3:0] to desired threshold value. Clock once to latch.
Example: 00000100 sets theta = 4, meaning at least 4 features must match.

### Step 3 - Run inference
Set ui_in[7:0] to the input feature vector.
After 4 clock cycles, uo_out[7] reflects the classification:
- fire = 1 means input pattern matches (S >= theta)
- fire = 0 means input pattern does not match (S < theta)

### Example

With weights = 11111111 and threshold theta = 4:
- Input 11111111 gives S = 8, 8 >= 4, fire = 1
- Input 11100000 gives S = 3, 3 < 4,  fire = 0
- Input 11110000 gives S = 4, 4 >= 4, fire = 1

## External hardware

No external hardware required.
