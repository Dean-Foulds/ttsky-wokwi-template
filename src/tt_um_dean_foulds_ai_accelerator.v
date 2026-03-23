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

    wire [7:0] feat;
    assign feat[0] = ui_in[0];
    assign feat[1] = ui_in[1];
    assign feat[2] = ui_in[2];
    assign feat[3] = ui_in[3];
    assign feat[4] = ui_in[4] ^ ui_in[5];
    assign feat[5] = ui_in[6] ^ ui_in[7];
    assign feat[6] = ui_in[0] & ui_in[7];
    assign feat[7] = ui_in[2] ^ ui_in[6];

    reg [2:0] bit_index;
    reg [3:0] acc [0:15];

    wire feature_bit = feat[bit_index];

    wire xnor0  = ~(weights[0] [bit_index] ^ feature_bit);
    wire xnor1  = ~(weights[1] [bit_index] ^ feature_bit);
    wire xnor2  = ~(weights[2] [bit_index] ^ feature_bit);
    wire xnor3  = ~(weights[3] [bit_index] ^ feature_bit);
    wire xnor4  = ~(weights[4] [bit_index] ^ feature_bit);
    wire xnor5  = ~(weights[5] [bit_index] ^ feature_bit);
    wire xnor6  = ~(weights[6] [bit_index] ^ feature_bit);
    wire xnor7  = ~(weights[7] [bit_index] ^ feature_bit);
    wire xnor8  = ~(weights[8] [bit_index] ^ feature_bit);
    wire xnor9  = ~(weights[9] [bit_index] ^ feature_bit);
    wire xnor10 = ~(weights[10][bit_index] ^ feature_bit);
    wire xnor11 = ~(weights[11][bit_index] ^ feature_bit);
    wire xnor12 = ~(weights[12][bit_index] ^ feature_bit);
    wire xnor13 = ~(weights[13][bit_index] ^ feature_bit);
    wire xnor14 = ~(weights[14][bit_index] ^ feature_bit);
    wire xnor15 = ~(weights[15][bit_index] ^ feature_bit);

    wire [3:0] acc_next0  = acc[0]  + {3'b0, xnor0};
    wire [3:0] acc_next1  = acc[1]  + {3'b0, xnor1};
    wire [3:0] acc_next2  = acc[2]  + {3'b0, xnor2};
    wire [3:0] acc_next3  = acc[3]  + {3'b0, xnor3};
    wire [3:0] acc_next4  = acc[4]  + {3'b0, xnor4};
    wire [3:0] acc_next5  = acc[5]  + {3'b0, xnor5};
    wire [3:0] acc_next6  = acc[6]  + {3'b0, xnor6};
    wire [3:0] acc_next7  = acc[7]  + {3'b0, xnor7};
    wire [3:0] acc_next8  = acc[8]  + {3'b0, xnor8};
    wire [3:0] acc_next9  = acc[9]  + {3'b0, xnor9};
    wire [3:0] acc_next10 = acc[10] + {3'b0, xnor10};
    wire [3:0] acc_next11 = acc[11] + {3'b0, xnor11};
    wire [3:0] acc_next12 = acc[12] + {3'b0, xnor12};
    wire [3:0] acc_next13 = acc[13] + {3'b0, xnor13};
    wire [3:0] acc_next14 = acc[14] + {3'b0, xnor14};
    wire [3:0] acc_next15 = acc[15] + {3'b0, xnor15};

    reg [15:0] fire_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_index <= 3'd0;
            fire_reg  <= 16'b0;
            acc[0]  <= 4'd0; acc[1]  <= 4'd0;
            acc[2]  <= 4'd0; acc[3]  <= 4'd0;
            acc[4]  <= 4'd0; acc[5]  <= 4'd0;
            acc[6]  <= 4'd0; acc[7]  <= 4'd0;
            acc[8]  <= 4'd0; acc[9]  <= 4'd0;
            acc[10] <= 4'd0; acc[11] <= 4'd0;
            acc[12] <= 4'd0; acc[13] <= 4'd0;
            acc[14] <= 4'd0; acc[15] <= 4'd0;
        end else if (mode) begin
            if (bit_index == 3'd7) begin
                bit_index <= 3'd0;
                fire_reg[0]  <= ($signed({1'b0, acc_next0})  + bias[0])  >= 0;
                fire_reg[1]  <= ($signed({1'b0, acc_next1})  + bias[1])  >= 0;
                fire_reg[2]  <= ($signed({1'b0, acc_next2})  + bias[2])  >= 0;
                fire_reg[3]  <= ($signed({1'b0, acc_next3})  + bias[3])  >= 0;
                fire_reg[4]  <= ($signed({1'b0, acc_next4})  + bias[4])  >= 0;
                fire_reg[5]  <= ($signed({1'b0, acc_next5})  + bias[5])  >= 0;
                fire_reg[6]  <= ($signed({1'b0, acc_next6})  + bias[6])  >= 0;
                fire_reg[7]  <= ($signed({1'b0, acc_next7})  + bias[7])  >= 0;
                fire_reg[8]  <= ($signed({1'b0, acc_next8})  + bias[8])  >= 0;
                fire_reg[9]  <= ($signed({1'b0, acc_next9})  + bias[9])  >= 0;
                fire_reg[10] <= ($signed({1'b0, acc_next10}) + bias[10]) >= 0;
                fire_reg[11] <= ($signed({1'b0, acc_next11}) + bias[11]) >= 0;
                fire_reg[12] <= ($signed({1'b0, acc_next12}) + bias[12]) >= 0;
                fire_reg[13] <= ($signed({1'b0, acc_next13}) + bias[13]) >= 0;
                fire_reg[14] <= ($signed({1'b0, acc_next14}) + bias[14]) >= 0;
                fire_reg[15] <= ($signed({1'b0, acc_next15}) + bias[15]) >= 0;
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
                acc[0]  <= acc_next0;  acc[1]  <= acc_next1;
                acc[2]  <= acc_next2;  acc[3]  <= acc_next3;
                acc[4]  <= acc_next4;  acc[5]  <= acc_next5;
                acc[6]  <= acc_next6;  acc[7]  <= acc_next7;
                acc[8]  <= acc_next8;  acc[9]  <= acc_next9;
                acc[10] <= acc_next10; acc[11] <= acc_next11;
                acc[12] <= acc_next12; acc[13] <= acc_next13;
                acc[14] <= acc_next14; acc[15] <= acc_next15;
            end
        end
    end

    assign uo_out  = fire_reg[7:0];
    assign uio_out = fire_reg[15:8];

endmodule
