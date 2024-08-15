# ESP32S3-CNN
这个项目主要是将CNN嵌入到单片机芯片中，实现边缘计算
以下简述项目过程
## 1，数据集构建与模型训练
  数据集从这里抄的https://github.com/lyg09270/CyberryPotter_ElectromagicWand_Basic_Project  
  （看了他的项目才知道，直接将Ptorch模型转为Tensorflow框架的，然后用nnom就行了，做麻烦了——_——，还把CNN从头到尾用C写了一遍）
  同样是MPU6050，与其使用的深度学习框架与嵌入式芯片存在差异  
  关于使用pytorch的模型训练见文件夹1

## 2，模型权重的保存
  将卷积层的weight，bias，神经网络的weight，bias都保存一下
