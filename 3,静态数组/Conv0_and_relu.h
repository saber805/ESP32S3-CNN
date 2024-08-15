#ifndef Conv0
#include"dtype.h"
#include"ReLU.h"

#define DATA_CHL 3
#define DATA_HIGH 150 //
#define KERNEL_NUM 30
#define KERNEL_SIZE_W 3
#define KERNEL_SIZE_H 3


void Conv0(Num_types raw_data[DATA_CHL][DATA_HIGH], 
                Num_types conv_0_weight[KERNEL_NUM][KERNEL_SIZE_H][KERNEL_SIZE_W], 
                Num_types conv_0_bias[30],
                int stride,
                Num_types output[KERNEL_NUM][50]
                )
{
    //首先遍历每一个卷积核
    for (int k = 0; k < KERNEL_NUM; k++) //conv_0_weight[k]就代表一个kernel，3*3的
    {
        int temp = 0;//记录循环次数
        for (int i = 0; i <= DATA_HIGH - KERNEL_SIZE_W; i = i + stride)//0-147 步长为3，会循环50次，
        {
            Num_types sum = 0;
            for (int j = 0; j < DATA_CHL; j++) //进行卷积操作
            {
                for (int h = 0; h < KERNEL_SIZE_W; h++)
                {
                    sum = sum + (raw_data[j][i+h] * conv_0_weight[k][j][h]);
                }
            }
            sum = sum + conv_0_bias[k];//加上卷积核的bias
            output[k][temp] = ReLu(sum);//这里直接通过RELU
            temp++;
        }//到这里是一个输入数据通过一个卷积核，输出一个50的数据,

    }
}





# endif