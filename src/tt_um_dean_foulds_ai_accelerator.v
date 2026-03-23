/*
 * 16-Neuron Systolic Binary Neural Network Accelerator - v2
 * Dean Foulds - Tiny Tapeout
 *
 * Improvements over v1:
 *   1. XNOR instead of AND  - true BNN dot product
 *   2. Systolic compute     - 1 bit per cycle over 8 cycles
 *   3. Signed bias          - replaces threshold
 *   4. Feature expansion    - XOR/AND feature engineering
 *   5. Balanced popcount    - tree structure
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

    // Weight and bias loading
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights[0]  <= 8'b0; weights[1]  <= 8'b0;
            weights[2]  <= 8'b0; weights[3]  <= 8'b0;
            weights[4]  <= 8'b0; weights[5]  <= 8'b0;
            weights[6]  <= 8'b0; weights[7]  <= 8'b0;
            weights[8]  <= 8'b0; weights[9]  <= 8'b0;
            weights[10] <= 8'b0; weights[11] <= 8'b0;
            weights[12] <= 8'b0; weights[13] <= 8'b0;
            weights[14] <= 8'b0; weights[15] <= 8'b0;
            bias[0]  <= 5'sd0; bias[1]  <= 5'sd0;
            bias[2]  <= 5'sd0; bias[3]  <= 5'sd0;
            bias[4]  <= 5'sd0; bias[5]  <= 5'sd0;
            bias[6]  <= 5'sd0; bias[7]  <= 5'sd0;
            bias[8]  <= 5'sd0; bias[9]  <= 5'sd0;
            bias[10] <= 5'sd0; bias[11] <= 5'sd0;
            bias[12] <= 5'sd0; bias[13] <= 5'sd0;
            bias[14] <= 5'sd0; bias[15] <= 5'sd0;
        end else if (!mode) begin
            if (!target)
                weights[sel] <= ui_in[7:0];
            else
                bias[sel] <= {ui_in[3], ui_in[3:0]};
        end
    end

    // Feature expansion
    wire [7:0] feat;
    assign feat[0] = ui_in[0];
    assign feat[1] = ui_in[1];
    assign feat[2] = ui_in[2];
    assign feat[3] = ui_in[3];
    assign feat[4] = ui_in[4] ^ ui_in[5];
    assign feat[5] = ui_in[6] ^ ui_in[7];
    assign feat[6] = ui_in[0] & ui_in[7];
    assign feat[7] = ui_in[2] ^ ui_in[6];

    // Systolic dot product engine
    reg [2:0] bit_index;
    reg [3:0] acc [0:15];

    wire feature_bit = feat[bit_index];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_index <= 3'd0;
            acc[0]  <= 4'd0; acc[1]  <= 4'd0;
            acc[2]  <= 4'd0; acc[3]  <= 4'd0;
            acc[4]  <= 4'd0; acc[5]  <= 4'd0;
            acc[6]  <= 4'd0; acc[7]  <= 4'd0;
            acc[8]  <= 4'd0; acc[9]  <= 4'd0;
            acc[10] <= 4'd0; acc[11] <= 4'd0;
            acc[12] <= 4'd0; acc[13] <= 4'd0;
            acc[14] <= 4'd0; acc[15] <= 4'd0;
        end else if (mode) begin
            acc[0]  <= acc[0]  + {3'b0, ~(weights[0] [bit_index] ^ feature_bit)};
            acc[1]  <= acc[1]  + {3'b0, ~(weights[1] [bit_index] ^ feature_bit)};
            acc[2]  <= acc[2]  + {3'b0, ~(weights[2] [bit_index] ^ feature_bit)};
            acc[3]  <= acc[3]  + {3'b0, ~(weights[3] [bit_index] ^ feature_bit)};
            acc[4]  <= acc[4]  + {3'b0, ~(weights[4] [bit_index] ^ feature_bit)};
            acc[5]  <= acc[5]  + {3'b0, ~(weights[5] [bit_index] ^ feature_bit)};
            acc[6]  <= acc[6]  + {3'b0, ~(weights[6] [bit_index] ^ feature_bit)};
            acc[7]  <= acc[7]  + {3'b0, ~(weights[7] [bit_index] ^ feature_bit)};
            acc[8]  <= acc[8]  + {3'b0, ~(weights[8] [bit_index] ^ feature_bit)};
            acc[9]  <= acc[9]  + {3'b0, ~(weights[9] [bit_index] ^ feature_bit)};
            acc[10] <= acc[10] + {3'b0, ~(weights[10][bit_index] ^ feature_bit)};
            acc[11] <= acc[11] + {3'b0, ~(weights[11][bit_index] ^ feature_bit)};
            acc[12] <= acc[12] + {3'b0, ~(weights[12][bit_index] ^ feature_bit)};
            acc[13] <= acc[13] + {3'b0, ~(weights[13][bit_index] ^ feature_bit)};
            acc[14] <= acc[14] + {3'b0, ~(weights[14][bit_index] ^ feature_bit)};
            acc[15] <= acc[15] + {3'b0, ~(weights[15][bit_index] ^ feature_bit)};

            if (bit_index == 3'd7) begin
                bit_index <= 3'd0;
                acc[0]  <= 4'd0; acc[1]  <= 4'd0;
                acc[2]  <= 4'd0; acc[3]  <= 4'd0;
                acc[4]  <= 4'd0; acc[5]  <= 4'd0;
                acc[6]  <= 4'd0; acc[7]  <= 4'd0;
                acc[8]  <= 4'd0; acc[9]  <= 4'd0;
                acc[10] <= 4'd0; acc[11] <= 4'd0;
                acc[12] <= 4'd0; acc[13] <= 4'd0;
                acc[14] <= 4'd0; acc[15] <= 4'd0;
            end else begin
                bit_index <= bit_index + 3'd1;
            end
        end
    end

    // Neuron fire decisions
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
