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

    // uio pins are control inputs only; no bidirectional output needed
    assign uio_oe  = 8'h00;
    assign uio_out = 8'h00;

    wire        mode   = uio_in[0];       // 0=load  1=infer
    wire        target = uio_in[1];       // 0=weights  1=bias
    wire [2:0]  sel    = uio_in[4:2];     // neuron select 0-7

    // ------------------------------------------------------------------ weight / bias registers
    reg [7:0]        weights [0:7];
    reg signed [4:0] bias    [0:7];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights[0] <= 8'b0; weights[1] <= 8'b0;
            weights[2] <= 8'b0; weights[3] <= 8'b0;
            weights[4] <= 8'b0; weights[5] <= 8'b0;
            weights[6] <= 8'b0; weights[7] <= 8'b0;
            bias[0] <= 5'sd0; bias[1] <= 5'sd0;
            bias[2] <= 5'sd0; bias[3] <= 5'sd0;
            bias[4] <= 5'sd0; bias[5] <= 5'sd0;
            bias[6] <= 5'sd0; bias[7] <= 5'sd0;
        end else if (!mode) begin
            if (!target)
                weights[sel] <= ui_in[7:0];
            else
                bias[sel] <= {ui_in[3], ui_in[3:0]};
        end
    end

    // ------------------------------------------------------------------ feature expansion
    // uio_in[7:5] XOR into feat[7:5]; when 0 (normal operation) behaviour is unchanged
    wire [7:0] feat;
    assign feat[0] = ui_in[0];
    assign feat[1] = ui_in[1];
    assign feat[2] = ui_in[2];
    assign feat[3] = ui_in[3];
    assign feat[4] = ui_in[4] ^ ui_in[5];
    assign feat[5] = ui_in[6] ^ ui_in[7] ^ uio_in[5];
    assign feat[6] = (ui_in[0] & ui_in[7]) ^ uio_in[6];
    assign feat[7] = (ui_in[2] ^ ui_in[6]) ^ uio_in[7];

    // ------------------------------------------------------------------ systolic BNN compute
    reg [2:0] bit_index;
    reg [3:0] acc [0:7];

    wire feature_bit = feat[bit_index];

    wire xnor0 = ~(weights[0][bit_index] ^ feature_bit);
    wire xnor1 = ~(weights[1][bit_index] ^ feature_bit);
    wire xnor2 = ~(weights[2][bit_index] ^ feature_bit);
    wire xnor3 = ~(weights[3][bit_index] ^ feature_bit);
    wire xnor4 = ~(weights[4][bit_index] ^ feature_bit);
    wire xnor5 = ~(weights[5][bit_index] ^ feature_bit);
    wire xnor6 = ~(weights[6][bit_index] ^ feature_bit);
    wire xnor7 = ~(weights[7][bit_index] ^ feature_bit);

    wire [3:0] acc_next0 = acc[0] + {3'b0, xnor0};
    wire [3:0] acc_next1 = acc[1] + {3'b0, xnor1};
    wire [3:0] acc_next2 = acc[2] + {3'b0, xnor2};
    wire [3:0] acc_next3 = acc[3] + {3'b0, xnor3};
    wire [3:0] acc_next4 = acc[4] + {3'b0, xnor4};
    wire [3:0] acc_next5 = acc[5] + {3'b0, xnor5};
    wire [3:0] acc_next6 = acc[6] + {3'b0, xnor6};
    wire [3:0] acc_next7 = acc[7] + {3'b0, xnor7};

    reg [7:0] fire_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_index <= 3'd0;
            fire_reg  <= 8'b0;
            acc[0] <= 4'd0; acc[1] <= 4'd0;
            acc[2] <= 4'd0; acc[3] <= 4'd0;
            acc[4] <= 4'd0; acc[5] <= 4'd0;
            acc[6] <= 4'd0; acc[7] <= 4'd0;
        end else if (mode) begin
            if (bit_index == 3'd7) begin
                bit_index <= 3'd0;
                fire_reg[0] <= ($signed({1'b0, acc_next0}) + bias[0]) >= 0;
                fire_reg[1] <= ($signed({1'b0, acc_next1}) + bias[1]) >= 0;
                fire_reg[2] <= ($signed({1'b0, acc_next2}) + bias[2]) >= 0;
                fire_reg[3] <= ($signed({1'b0, acc_next3}) + bias[3]) >= 0;
                fire_reg[4] <= ($signed({1'b0, acc_next4}) + bias[4]) >= 0;
                fire_reg[5] <= ($signed({1'b0, acc_next5}) + bias[5]) >= 0;
                fire_reg[6] <= ($signed({1'b0, acc_next6}) + bias[6]) >= 0;
                fire_reg[7] <= ($signed({1'b0, acc_next7}) + bias[7]) >= 0;
                acc[0] <= 4'd0; acc[1] <= 4'd0;
                acc[2] <= 4'd0; acc[3] <= 4'd0;
                acc[4] <= 4'd0; acc[5] <= 4'd0;
                acc[6] <= 4'd0; acc[7] <= 4'd0;
            end else begin
                bit_index <= bit_index + 3'd1;
                acc[0] <= acc_next0; acc[1] <= acc_next1;
                acc[2] <= acc_next2; acc[3] <= acc_next3;
                acc[4] <= acc_next4; acc[5] <= acc_next5;
                acc[6] <= acc_next6; acc[7] <= acc_next7;
            end
        end
    end

    assign uo_out = ena ? fire_reg : 8'b0;

endmodule
