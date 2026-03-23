`timescale 1ns/1ps
`default_nettype none

module tb_ai_accelerator;

    reg  [7:0] ui_in;
    wire [7:0] uo_out;
    reg  [7:0] uio_in;
    wire [7:0] uio_out;
    wire [7:0] uio_oe;
    reg        ena;
    reg        clk;
    reg        rst_n;

    tt_um_dean_foulds_ai_accelerator dut (
        .ui_in(ui_in), .uo_out(uo_out),
        .uio_in(uio_in), .uio_out(uio_out), .uio_oe(uio_oe),
        .ena(ena), .clk(clk), .rst_n(rst_n)
    );

    always #5 clk = ~clk;

    task pulse_clock;
        @(posedge clk); #1;
    endtask

    task load_weights;
        input [3:0] neuron;
        input [7:0] wdata;
        begin
            uio_in[0] = 0; uio_in[1] = 0;
            uio_in[5:2] = neuron;
            ui_in = wdata;
            pulse_clock;
            $display("  Loaded weights[%0d] = %08b", neuron, wdata);
        end
    endtask

    task load_bias;
        input [3:0] neuron;
        input signed [4:0] bdata;
        begin
            uio_in[0] = 0; uio_in[1] = 1;
            uio_in[5:2] = neuron;
            ui_in = {3'b0, bdata[4], bdata[3:0]};
            pulse_clock;
            $display("  Loaded bias[%0d]    = %0d", neuron, $signed(bdata));
        end
    endtask

    task infer;
        input [7:0] input_vec;
        output [15:0] result;
        integer k;
        begin
            uio_in[0] = 1;
            ui_in = input_vec;
            // pulse 16 times to guarantee at least one complete 8-cycle window
            for (k = 0; k < 16; k = k + 1)
                pulse_clock;
            result = {uio_out, uo_out};
        end
    endtask

    integer j;
    reg [15:0] result;

    initial begin
        $dumpfile("sim.vcd");
        $dumpvars(0, tb_ai_accelerator);

        clk = 0; rst_n = 0; ena = 1;
        ui_in = 0; uio_in = 0;
        #20; rst_n = 1; #10;

        $display("\n=== TEST 1: Reset ===");
        $display("After reset: fire_reg = %016b (expect all 0)", {uio_out, uo_out});

        $display("\n=== TEST 2: Load weights and bias ===");
        load_weights(4'd0, 8'b11111111);
        load_bias(4'd0, 5'sd0);

        $display("\n=== TEST 3: All ones input ===");
        infer(8'b11111111, result);
        $display("Input=11111111 w=11111111 bias=0 → fire=%0b (expect 1)", result[0]);

        $display("\n=== TEST 4: Pattern match ===");
        load_weights(4'd1, 8'b10101010);
        load_bias(4'd1, -5'sd3);
        infer(8'b10101010, result);
        $display("Input=10101010 w=10101010 bias=-3 → fire=%0b (expect 1)", result[1]);
        infer(8'b01010101, result);
        $display("Input=01010101 w=10101010 bias=-3 → fire=%0b (expect 0)", result[1]);

        $display("\n=== TEST 5: Multiple neurons ===");
        load_weights(4'd2, 8'b11110000);
        load_bias(4'd2, -5'sd3);
        load_weights(4'd3, 8'b00001111);
        load_bias(4'd3, -5'sd3);
        infer(8'b11110000, result);
        $display("Input=11110000: n2=%0b n3=%0b (expect 1 0)", result[2], result[3]);
        infer(8'b00001111, result);
        $display("Input=00001111: n2=%0b n3=%0b (expect 0 1)", result[2], result[3]);

        $display("\n=== ALL TESTS COMPLETE ===\n");
        $finish;
    end

endmodule
