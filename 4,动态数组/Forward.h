#ifndef Forward
#include"dtype.h"

#define MLP_INPUT_SIZE 75
#define MLP_OUTPUT_SIZE 13


void Forward(Num_types* INPUT_Linear, Num_types MLP_WEIGHT[MLP_OUTPUT_SIZE][MLP_INPUT_SIZE], Num_types OUTPUT_BIAS[MLP_OUTPUT_SIZE],Num_types* OUTPUT_Linear)
{
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++)
    {
        for (int j = 0; j < MLP_INPUT_SIZE; j++)
        {
            OUTPUT_Linear[i] += INPUT_Linear[j] * MLP_WEIGHT[i][j];
        }
         OUTPUT_Linear[i] += OUTPUT_BIAS[i];
    }
}

#endif