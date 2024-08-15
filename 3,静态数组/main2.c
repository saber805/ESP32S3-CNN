#include <stdio.h>
#include <stdlib.h>

int main() {
    int rows = 3;
    int cols = 4;

    // 一次性分配所有内存
    int *data = (int *)malloc(rows * cols * sizeof(int));
    int **arr = (int **)malloc(rows * sizeof(int *));

    // 将每行的起始地址指向相应的内存位置
    for (int i = 0; i < rows; i++) {
        arr[i] = data + i * cols;
    }

    // 初始化二维数组
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i][j] = i * cols + j;
        }
    }

    // 打印二维数组
    printf("2D Array:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }

    // 释放内存
    free(data);
    free(arr);

    return 0;
}
