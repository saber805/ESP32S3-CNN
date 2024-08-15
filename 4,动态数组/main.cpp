#include <Arduino.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include"dtype.h"
#include"conv_0_weight.h"
#include"conv_0_bias.h"
#include"conv_2_weight.h"
#include"conv_2_bias.h"
#include"classifier_0_weight.h"
#include"classifier_0_bias.h"
#include"Conv0_and_relu.h"
#include"Conv2_and_relu.h"
#include"Maxpool.h"
#include"Forward.h"
#include <stdlib.h>
#include <Ticker.h>
#define BUTTON_PIN 39   //连接按钮的引脚

#define ATTITUDE_INDEX_MAX 150

Adafruit_MPU6050 mpu;  //姿态动作
Ticker key_tick;
Ticker zitai_kick;

bool State = false;
bool Start_recognition = false;

float att_arr[3][ATTITUDE_INDEX_MAX] = {{0}};  //存储姿态的静态二维数组
int att_index = 0; //姿态数据index


void zitai(){
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // //Serial0.print("AccelX:");
  // Serial0.print(a.acceleration.x);
  // Serial0.print(",");
  // //Serial0.print("AccelY:");
  // Serial0.print(a.acceleration.y);
  // Serial0.print(",");
  // //Serial0.print("AccelZ:");
  // Serial0.print(a.acceleration.z);
  // Serial0.print(", ");
  //Serial0.print("GyroX:");
  // Serial0.print((int)round(g.gyro.x));
  // Serial0.print(",");
  // //Serial0.print("GyroY:");
  // Serial0.print((int)round(g.gyro.y));
  // Serial0.print(",");
  // //Serial0.print("GyroZ:");
  // Serial0.print((int)round(g.gyro.z));
  // Serial0.println("");

  //将读取到的姿态数据存入数组
  if(State == true){
    att_arr[0][att_index] = g.gyro.x;
    att_arr[1][att_index] = g.gyro.y;
    att_arr[2][att_index] = g.gyro.z;
    att_index++;
    if (att_index >= 150){
      att_index = 0;
      State = false;
      Start_recognition = true;
    }
  }
 
}


void Key_Detection(){
      // 读取按钮的电平状态
  int buttonState = digitalRead(BUTTON_PIN);
  // 按钮被按下时，电平状态为低（0），否则为高（1）
  if (buttonState == LOW) {
    Serial0.println("Button Pressed!");
    // 添加一个小延时，防止抖动
    State = true;
    delay(180);
  }
}

void setup() {
  Serial0.begin(115200);
  Wire.begin(12,13);
  // Try to initialize!
  if (!mpu.begin()) {
    Serial0.println("Failed to find MPU6050 chip");
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  key_tick.attach_ms(1, Key_Detection);
  zitai_kick.attach_ms(10,zitai);
}

void loop() {



  if (Start_recognition == true)
  {
    Serial0.print("状态存满！开始识别！\n");
    Num_types *data2 = (Num_types *)malloc(30 * 50 * sizeof(Num_types));
    Num_types **raw_data2 = (Num_types **)malloc(30 * sizeof(Num_types *));
    // 将每行的起始地址指向相应的内存位置
    for (int i = 0; i < 30; i++) {
        raw_data2[i] = data2 + i * 50;
    }
    //初始化为0
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 50; j++) {
            raw_data2[i][j] = 0;
        }
    }
    Conv0(att_arr ,conv_0_weight,conv_0_bias,3,raw_data2);

    //通过一层卷积和激活的输出
    // for (int i = 0; i < 30; i++)
    // {
    //     Serial0.printf("Output for data %d:\n", i);
    //     for (int j = 0; j < 50; j++)
    //     {
    //         Serial0.printf("%f,",raw_data2[i][j]);
    //     }
    //     Serial0.println();
    // }




    Num_types *data3 = (Num_types *)malloc(15 * 16 * sizeof(Num_types));
    Num_types **raw_data3 = (Num_types **)malloc(15 * sizeof(Num_types *));

    //检查是否分配成功
    if (data3 == NULL || *raw_data3 == NULL) {
      Serial.println("Memory allocation failed for data3.");
      return;
    }

    // 将每行的起始地址指向相应的内存位置
    for (int i = 0; i < 15; i++) {
        raw_data3[i] = data3 + i * 15;
    }
    
    //初始化为0
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 16; j++) {
            raw_data3[i][j] = 0;
        }
    }


    Conv2(raw_data2,conv_2_weight,conv_2_bias,3,raw_data3);
    //输出第二层的卷积结果
    // for (int i = 0; i < 15; i++)
    // {
    //     Serial0.printf("Output for data %d:\n", i);
    //     for (int j = 0; j < 16; j++)
    //     {
    //         Serial0.printf("%f,",raw_data3[i][j]);
    //     }
    //    Serial0.println();
    // }
    free(raw_data2);//释放掉第一层卷积的结果
    free(data2);

    Num_types *data4 = (Num_types *)malloc(15 * 5 * sizeof(Num_types));
    Num_types **raw_data4 = (Num_types **)malloc(15 * sizeof(Num_types *));

    for (int i = 0; i < 15; i++) {
        raw_data4[i] = data4 + i * 5;
    }
    
    //初始化为0
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 5; j++) {
            raw_data4[i][j] = 0;
        }
    }
    Maxpool(raw_data3,3,3,raw_data4);
    free(raw_data3);//释放掉第二层卷积的结果
    free(data3);

    free(raw_data4);//恰好，data4就是一维的

    //输出池化结果
    // for (int i = 0; i < 15; i++)
    // {
    //     Serial0.printf("Output for data %d:\n", i);
    //     for (int j = 0; j < 5; j++)
    //     {
    //         Serial0.printf("%f,",raw_data4[i][j]);
    //     }
    //    Serial0.println();
    // }

    Num_types *data5 = (Num_types *)malloc(13 * sizeof(Num_types));

    //初始化为0
    for (int i = 0; i < 13; i++) {
      data5[i] = 0;
    }

    Forward(data4, classifier_0_weight, classifier_0_bias, data5);
    free(data4);
    for (int i = 0; i < 13; i++)
    {
        Serial0.printf("%f,", data5[i]);
    }

    //找到最大值的index
    char motion_names[13][11] = {{"RightAngle"}, {"SharpAngle"}, {"Lightning"}, 
                                {"Triangle"}, {"Letter_h"}, {"letter_R"}, 
                                {"letter_W"}, {"letter_phi"}, {"Circle"}, 
                                {"UpAndDown"}, {"Horn"}, {"Wave"}, {"NoMotion"}};

    int maxIndex = 0; // 假设第一个元素为最大值
    for (int i = 1; i < 13; i++) {
        if (data5[i] > data5[maxIndex]) {
            maxIndex = i; // 更新最大值的索引
        }
    }
    free(data5);
    Serial0.println(motion_names[maxIndex]);
    
    Start_recognition = false;
  }
  
} 
