#ifndef Conv2
#include"dtype.h"

//第二层卷积核激活函数
#define INPUT_DATA_CHL 30
#define INPUT_DATA_W 50
#define KERNEL_NUM_2 15
#define KERNEL_SIZE_W 3



void Conv2(Num_types** input_data, 
                Num_types conv_2_weight[KERNEL_NUM_2][INPUT_DATA_CHL][KERNEL_SIZE_W], 
                Num_types conv_2_bias[KERNEL_NUM_2],
                int stride,
                Num_types** output2
                )
{
    //遍历所有卷积核
    for (int k = 0; k < KERNEL_NUM_2; k++)
    {
        int temp = 0;//记录循环次数
        for (int i = 0; i <= INPUT_DATA_W - KERNEL_SIZE_W - 1; i = i + stride)//0-46
        {
            Num_types sum = 0;
            //开始卷积操作
            for (int j = 0; j < INPUT_DATA_CHL; j++) //进行卷积操作
            {
                for (int h = 0; h < KERNEL_SIZE_W; h++)
                {
                    sum += input_data[j][i+h] * conv_2_weight[k][j][h];
                }
            }
            sum = sum + conv_2_bias[k];//加上卷积核的bias
            output2[k][temp] = ReLu(sum);
            temp++;
        }
    }
}

#endif