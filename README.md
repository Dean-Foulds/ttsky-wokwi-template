![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg)

# 8-Neuron Binary Neural Network

An 8-neuron binary neural network inference layer implemented in silicon.
Each neuron has 8 programmable binary weights and a signed 5-bit bias.

- [Read the project documentation](docs/info.md)

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that makes it easier and cheaper
than ever to get your digital designs manufactured on a real chip.

Visit https://tinytapeout.com to learn more.

## Resources

- [FAQ](https://tinytapeout.com/faq/)
- [Digital design lessons](https://tinytapeout.com/digital_design/)
- [Join the community](https://tinytapeout.com/discord)
- [Submit your design](https://app.tinytapeout.com/)


iverilog -g2012 -o sim_tb tt_um_dean_foulds_ai_accelerator.v tb_tt_um_dean_foulds.v
vvp sim_tb