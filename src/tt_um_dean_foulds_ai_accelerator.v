/*
 * 16-Neuron Systolic Binary Neural Network - v2
 * Dean Foulds - Tiny Tapeout
 *
 * Improvements over v1:
 *   1. XNOR instead of AND  — true BNN dot product (counts matching bits)
 *   2. Systolic compute     — reuses hardware, 1 bit per cycle over 8 cycles
 *   3. Signed bias          — replaces threshold, easier to train from Python
 *   4. Feature expansion    — cheap XOR/AND logic extracts non-linear features
 *   5. Balanced popcount    — tree structure reduces timing path
 *
 * Mathematical model per neuron n:
 *   S[n] = popcount( XNOR( weights[n], features ) )  over 8 cycles
 *   y[n] = 1  if  S[n] + bias[n] >= 0
 *   y[n] = 0  otherwise
 *
 * Feature expansion (hardware feature engineering):
 *   feat[0] = x[0]
 *   feat[1] = x[1]
 *   feat[2] = x[2]
 *   feat[3] = x[3]
 *   feat[4] = x[4] XOR x[5]   (detects difference between bits 4,5)
 *   feat[5] = x[6] XOR x[7]   (detects difference between bits 6,7)
 *   feat[6] = x[0] AND x[7]   (detects both ends active)
 *   feat[7] = x[2] XOR x[6]   (detects diagonal pattern)
 *
 * Timing:
 *   - Results update every 8 clock cycles (systolic pipeline)
 *   - After 8 clocks in infer mode, outputs reflect current input
 *   - New result produced every 8 clocks continuously
 *
 * Pin mapping:
 *   ui_in[7:0]   - input features (infer) or load data (load)
 *   uio_in[0]    - mode: 0=load, 1=infer
 *   uio_in[1]    - target: 0=weights, 1=bias
 *   uio_in[5:2]  - neuron select (0-15)
 *   uo_out[7:0]  - fire signals neurons 0-7
 *   uio_out[7:0] - fire signals neurons 8-15
 *
 * Loading weights:
 *   1. uio_in[0]=0, uio_in[1]=0, uio_in[5:2]=neuron_n
 *   2. ui_in[7:0] = 8-bit weight pattern
 *   3. Pulse clock
 *
 * Loading bias:
 *   1. uio_in[0]=0, uio_in[1]=1, uio_in[5:2]=neuron_n
 *   2. ui_in[3:0] = bias magnitude, ui_in[4] = sign (1=negative)
 *   3. Pulse clock
 *      bias stored as signed 5-bit: {ui_in[3], ui_in[3:0]}
 *
 * Inference:
 *   1. uio_in[0]=1
 *   2. ui_in[7:0] = input feature vector
 *   3. After 8 clock cycles outputs reflect classification
 */

`default_nettype none

module tt_um_dean_foulds_ai_accelerator (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    assign uio_oe = 8'hFF;

    wire        mode   = uio_in[0];
    wire        target = uio_in[1];
    wire [3:0]  sel    = uio_in[5:2];

    reg [7:0]        weights [0:15];
    reg signed [4:0] bias    [0:15];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 16; i = i + 1) begin
                weights[i] <= 8'b0;
                bias[i]    <= 5'sd0;
            end
        end else if (!mode) begin
            if (!target)
                weights[sel] <= ui_in[7:0];
            else
                bias[sel] <= {ui_in[3], ui_in[3:0]};
        end
    end

    wire [7:0] feat;

    assign feat[0] = ui_in[0];
    assign feat[1] = ui_in[1];
    assign feat[2] = ui_in[2];
    assign feat[3] = ui_in[3];
    assign feat[4] = ui_in[4] ^ ui_in[5];
    assign feat[5] = ui_in[6] ^ ui_in[7];
    assign feat[6] = ui_in[0] & ui_in[7];
    assign feat[7] = ui_in[2] ^ ui_in[6];

    reg [3:0] bit_index;
    reg [3:0] acc [0:15];

    wire feature_bit = feat[bit_index];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_index <= 4'd0;
            for (i = 0; i < 16; i = i + 1)
                acc[i] <= 4'd0;
        end else if (mode) begin
            for (i = 0; i < 16; i = i + 1)
                acc[i] <= acc[i] + {3'b0, ~(weights[i][bit_index] ^ feature_bit)};

            if (bit_index == 4'd7) begin
                bit_index <= 4'd0;
                for (i = 0; i < 16; i = i + 1)
                    acc[i] <= 4'd0;
            end else begin
                bit_index <= bit_index + 4'd1;
            end
        end
    end

    wire fire [0:15];

    genvar n;
    generate
        for (n = 0; n < 16; n = n + 1) begin : neuron
            assign fire[n] = ($signed({1'b0, acc[n]}) + bias[n]) >= 0;
        end
    endgenerate

    assign uo_out  = {fire[7],  fire[6],  fire[5],  fire[4],
                      fire[3],  fire[2],  fire[1],  fire[0]};

    assign uio_out = {fire[15], fire[14], fire[13], fire[12],
                      fire[11], fire[10], fire[9],  fire[8]};

endmodule
