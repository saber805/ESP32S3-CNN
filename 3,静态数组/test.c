#include <stdio.h>

#define DATA_CHL 3
#define DATA_HIGH 150
#define KERNEL_NUM 30
#define KERNEL_SIZE_W 3
#define KERNEL_SIZE_H 3

typedef float Num_types;

Num_types ReLu(Num_types x) {
    return (x > 0) ? x : 0;
}

void Conv0(Num_types raw_data[DATA_CHL][DATA_HIGH], 
           Num_types conv_0_weight[KERNEL_NUM][DATA_CHL][KERNEL_SIZE_W][KERNEL_SIZE_H], 
           Num_types conv_0_bias[KERNEL_NUM],
           int stride,
           Num_types output[KERNEL_NUM][50]) // 50 是通过 (DATA_HIGH - KERNEL_SIZE_H) / stride + 1 计算得到的
{
    // 遍历每一个卷积核
    for (int k = 0; k < KERNEL_NUM; k++) {
        int temp = 0; // 记录循环次数
        for (int i = 0; i <= DATA_HIGH - KERNEL_SIZE_H; i += stride) {//i是0-147 步长为3，会循环50次，
            Num_types sum = 0;
            for (int j = 0; j < KERNEL_SIZE_H; j++) {
                for (int h = 0; h < KERNEL_SIZE_W; h++) {
                    for (int c = 0; c < DATA_CHL; c++) {
                        sum += raw_data[c][i+h] * conv_0_weight[k][c][j][h];
                    }
                }
            }
            sum += conv_0_bias[k]; // 加上卷积核的bias
            output[k][temp] = ReLu(sum); // 通过RELU
            temp++;
        }
    }
}

int main() {
    // 示例输入数据 (3 通道，每个通道 150 个元素)
    Num_types raw_data[DATA_CHL][DATA_HIGH];
    for (int c = 0; c < DATA_CHL; c++) {
        for (int i = 0; i < DATA_HIGH; i++) {
            raw_data[c][i] = (Num_types)(1); // 简单填充一些数据
        }
    }

    // 示例卷积核权重和偏置
    Num_types conv_0_weight[KERNEL_NUM][DATA_CHL][KERNEL_SIZE_W][KERNEL_SIZE_H];
    Num_types conv_0_bias[KERNEL_NUM];
    for (int k = 0; k < KERNEL_NUM; k++) {
        conv_0_bias[k] = (Num_types)k; // 偏置初始化为 k
        for (int c = 0; c < DATA_CHL; c++) {
            for (int j = 0; j < KERNEL_SIZE_H; j++) {
                for (int h = 0; h < KERNEL_SIZE_W; h++) {
                    conv_0_weight[k][c][j][h] = 1.0; // 简单设置为 1.0
                }
            }
        }
    }

    // 输出数组 (每个卷积核输出 50 个值)
    Num_types output[KERNEL_NUM][50] = {0};

    // 定义步长
    int stride = 3;

    // 调用 Conv0 函数进行卷积操作
    Conv0(raw_data, conv_0_weight, conv_0_bias, stride, output);

    // 打印输出
    for (int k = 0; k < KERNEL_NUM; k++) {
        printf("Output for kernel %d:\n", k);
        for (int i = 0; i < 50; i++) {
            printf("%f ", output[k][i]);
        }
        printf("\n");
    }

    return 0;
}
