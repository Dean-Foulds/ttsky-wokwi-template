/*
 * 16-Neuron Binary Neural Network
 * Dean Foulds - Tiny Tapeout
 *
 * Architecture:
 *   - 16 binary perceptron neurons arranged in a single inference layer
 *   - Each neuron has 8 binary weights and a 4-bit threshold
 *   - All 16 neurons process the same 8-bit input simultaneously
 *   - Produces a 16-bit output vector (1 bit per neuron fire/no-fire)
 *
 * Mathematical model per neuron n:
 *   S[n] = sum(w[n][i] AND x[i]) for i = 0..7
 *   y[n] = 1 if S[n] >= theta[n], else 0
 *
 * Pin mapping:
 *   ui_in[7:0]   - 8-bit input feature vector (inference mode)
 *                  or weight/threshold load data (load mode)
 *   uio_in[0]    - mode: 0=load, 1=infer
 *   uio_in[1]    - target: 0=weights, 1=thresholds
 *   uio_in[4:2]  - neuron select (0-15) for loading
 *   uo_out[7:0]  - fire signals for neurons 0-7
 *   uio_out[7:0] - fire signals for neurons 8-15
 *
 * Loading weights:
 *   1. Set uio_in[0]=0 (load mode), uio_in[1]=0 (weights)
 *   2. Set uio_in[4:2] to neuron number (0-7 via this port)
 *   3. Set ui_in[7:0] to 8-bit weight pattern
 *   4. Pulse clock - weights latched into selected neuron
 *
 * Loading thresholds:
 *   1. Set uio_in[0]=0 (load mode), uio_in[1]=1 (thresholds)
 *   2. Set uio_in[4:2] to neuron number
 *   3. Set ui_in[3:0] to 4-bit threshold value
 *   4. Pulse clock - threshold latched into selected neuron
 *
 * Inference:
 *   1. Set uio_in[0]=1 (infer mode)
 *   2. Set ui_in[7:0] to input feature vector
 *   3. After 1 clock cycle, uo_out and uio_out show fire signals
 */

`default_nettype none

module tt_um_dean_foulds_perceptron (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (1=output)
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    // ─── IO direction ───────────────────────────────────────────────
    // uio[7:0]: upper 8 neuron outputs (neurons 8-15)
    assign uio_oe = 8'b11111111;  // all bidirectional pins = output

    // ─── Control signals ────────────────────────────────────────────
    wire mode       = uio_in[0];  // 0=load, 1=infer
    wire target     = uio_in[1];  // 0=weights, 1=thresholds
    wire [3:0] sel  = uio_in[5:2]; // neuron select 0-15

    // ─── Weight registers (16 neurons × 8 bits) ─────────────────────
    reg [7:0] weights [0:15];
    reg [3:0] thresholds [0:15];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 16; i = i + 1) begin
                weights[i]    <= 8'b0;
                thresholds[i] <= 4'd4;  // default threshold = 4
            end
        end else if (!mode) begin
            // Load mode
            if (!target) begin
                weights[sel] <= ui_in[7:0];
            end else begin
                thresholds[sel] <= ui_in[3:0];
            end
        end
    end

    // ─── Inference: compute weighted sum for each neuron ────────────
    // Each neuron computes: sum = popcount(weights[n] AND ui_in)
    // Then fires if sum >= threshold[n]

    wire [7:0] products [0:15];
    wire [3:0] sums     [0:15];
    wire       fire     [0:15];

    genvar n;
    generate
        for (n = 0; n < 16; n = n + 1) begin : neuron
            // AND array: element-wise multiply weights by inputs
            assign products[n] = weights[n] & ui_in;

            // Adder tree: count active products (popcount)
            assign sums[n] =
                products[n][0] +
                products[n][1] +
                products[n][2] +
                products[n][3] +
                products[n][4] +
                products[n][5] +
                products[n][6] +
                products[n][7];

            // Threshold comparator: fire if sum >= threshold
            assign fire[n] = (sums[n] >= thresholds[n]) ? 1'b1 : 1'b0;
        end
    endgenerate

    // ─── Output assignment ──────────────────────────────────────────
    assign uo_out  = {fire[7],  fire[6],  fire[5],  fire[4],
                      fire[3],  fire[2],  fire[1],  fire[0]};

    assign uio_out = {fire[15], fire[14], fire[13], fire[12],
                      fire[11], fire[10], fire[9],  fire[8]};

endmodule
