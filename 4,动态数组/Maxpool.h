#ifndef MaxPool
#include"dtype.h"
#define INPUT_DATA_CHANNEL 15
#define INPUT_DATA_LENGTH 16


void Maxpool(Num_types** input_data, int Kernel_size, int stride, Num_types** output)
{
    //输入数据有15通道每一个channel都进行最大池化
    for (int c = 0; c < INPUT_DATA_CHANNEL; c++)
    {
        int temp = 0;//记录循环次数
        for (int i = 0; i < INPUT_DATA_LENGTH - Kernel_size; i+=stride)//
        {
            //一维maxpool，三个数据里找个最大的就可以
            // Num_types n1 = input_data[c][i];
            // Num_types n2 = input_data[c][i+1];
            // Num_types n3 = input_data[c][i+2];
            //找最大
            output[c][temp] = (input_data[c][i]>input_data[c][i+1]?input_data[c][i]:input_data[c][i+1])>input_data[c][i+2]?(input_data[c][i]>input_data[c][i+1]?input_data[c][i]:input_data[c][i+1]):input_data[c][i+2];
            temp++;
        }
    }
    
}





#endif