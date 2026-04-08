# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


def compute_feat(ui_in):
    """Feature expansion matching the hardware (8 input bits → 8 feature bits)."""
    f = [0] * 8
    f[0] = (ui_in >> 0) & 1
    f[1] = (ui_in >> 1) & 1
    f[2] = (ui_in >> 2) & 1
    f[3] = (ui_in >> 3) & 1
    f[4] = ((ui_in >> 4) & 1) ^ ((ui_in >> 5) & 1)
    f[5] = ((ui_in >> 6) & 1) ^ ((ui_in >> 7) & 1)
    f[6] = ((ui_in >> 0) & 1) & ((ui_in >> 7) & 1)
    f[7] = ((ui_in >> 2) & 1) ^ ((ui_in >> 6) & 1)
    return f


def expected_fire(weight, bias_signed, ui_in):
    """Return 1 if the neuron fires, 0 otherwise.

    bias_signed: signed integer, e.g. -3 or 0. Valid hardware range is -8..7
    (4-bit signed stored as sign-extended 5-bit in the design).
    """
    feat = compute_feat(ui_in)
    popcount = sum(1 for i in range(8) if feat[i] == ((weight >> i) & 1))
    return int((popcount + bias_signed) >= 0)


@cocotb.test()
async def test_ai_accelerator(dut):
    dut._log.info("Start")

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # ------------------------------------------------------------------ helpers

    async def reset():
        dut.rst_n.value = 0
        dut.ena.value = 1
        dut.ui_in.value = 0
        dut.uio_in.value = 0
        await ClockCycles(dut.clk, 5)
        dut.rst_n.value = 1
        await ClockCycles(dut.clk, 2)

    async def drive(uio, ui):
        """Drive inputs, then wait one clock cycle for them to be registered."""
        dut.uio_in.value = uio
        dut.ui_in.value = ui
        await ClockCycles(dut.clk, 1)

    async def infer(input_vec):
        """Run one 8-cycle inference and return the fire outputs as an integer.

        Uses 9 clock cycles total: 8 for the BNN computation + 1 extra so that
        the non-blocking assignments from cycle 8 have settled before we sample.
        """
        dut.uio_in.value = 0b00000001  # mode=1 (infer)
        dut.ui_in.value = input_vec
        await ClockCycles(dut.clk, 9)
        return dut.uo_out.value.integer

    # ------------------------------------------------------------------ Test 1: reset
    await reset()
    assert dut.uo_out.value == 0, f"Expected uo_out=0 after reset, got {dut.uo_out.value}"
    dut._log.info("PASS test_reset")

    # ------------------------------------------------------------------ Test 2: neuron 0 fires for all-ones input
    # weight[0]=0xFF, bias[0]=0 ; input=0xFF
    # feat(0xFF) bits 0-3=1, bit4=0, bit5=0, bit6=1, bit7=0 → popcount=5; 5+0≥0 → fire
    await reset()
    await drive(0b00000000, 0xFF)   # mode=0, target=0, sel=0 → load weight[0]=0xFF
    await drive(0b00000010, 0x00)   # mode=0, target=1, sel=0 → load bias[0]=0

    result = await infer(0xFF)
    exp = expected_fire(0xFF, 0, 0xFF)
    got = result & 0x01
    assert got == exp, f"Test 2: fire[0]={got}, expected {exp}"
    dut._log.info(f"PASS test_neuron0_fire: fire[0]={got}")

    # ------------------------------------------------------------------ Test 3: neuron 1 fires for matching pattern
    # weight[1]=0b10101010, bias[1]=-3 ; input=0b10101010
    # popcount=6 (computed from feature-expanded input); 6-3=3≥0 → fire
    await reset()
    await drive(0b00000100, 0b10101010)  # mode=0, target=0, sel=1 → load weight[1]
    # bias=-3: lower nibble 4'b1101=0xD → {ui_in[3],ui_in[3:0]}={1,1101}=5'b11101=-3
    await drive(0b00000110, 0x0D)        # mode=0, target=1, sel=1 → load bias[1]=-3

    result = await infer(0b10101010)
    exp = expected_fire(0b10101010, -3, 0b10101010)
    got = (result >> 1) & 0x01
    assert got == exp, f"Test 3: fire[1]={got}, expected {exp}"
    dut._log.info(f"PASS test_neuron1_match: fire[1]={got}")

    # ------------------------------------------------------------------ Test 4: neuron 1 does NOT fire for anti-pattern
    # Same weights; input=0b01010101 (bit-inverted)
    # popcount=2; 2-3=-1<0 → no fire
    await reset()
    await drive(0b00000100, 0b10101010)
    await drive(0b00000110, 0x0D)

    result = await infer(0b01010101)
    exp = expected_fire(0b10101010, -3, 0b01010101)
    got = (result >> 1) & 0x01
    assert got == exp, f"Test 4: fire[1]={got}, expected {exp}"
    dut._log.info(f"PASS test_neuron1_antimatch: fire[1]={got}")

    dut._log.info("All tests passed!")
