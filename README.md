# ANN-for-mnist
Use python to write an Ann on mnist dataset from beginning

## Dataset 
http://yann.lecun.com/exdb/mnist/

## result
1. 97% accuracy after 15 epochs
2. 97.4% accuracy after litter epochs(around 70)
3. 7*7 hidden layer
4. learning_rate = 0.05
5. add infer_grad optimize　

        (raise 3% accuarcy, use 10% epochs)
6. add +-1/sqrt(layer.neuron_number) init optimize

        (based on infer_grad optimize, speed up and  raise 0.2% accuarcy, 1 epochs reach 92% accuracy)

## 设计
1. 三层网络视为两个神经元，输入层不算
2. 单个神经元模型
    
        x*w -> b -> active function -> output
3. 采用流式训练效果较好，但是耗时。batch 训练快但是效果不佳
1. 对于ＡＮＮ，网络弱化了像素的相对位置关系的影响，可以将图片转为一维向量处理。而对于ＣＮＮ，卷积核考虑到了像素的相对位置关系的影响。

   在我的实现中，全部转化为vector计算，便于反向传播。但是在每一层的*_shape变量中记录了原来设定的形状。当需要扩展为ＣＮＮ时，可以根据shape还原并卷积后在转为vector）
   
4. 批训练：多组输入，对于loss取均值后反向传播
1. 将所有vector初始化为列向量
1. weight的初始值应当用np.random.randn会更好,符合正太分布的正负值。
另外一个经验公式是对于weight的随机值最好保持在+-1/sqrt(layer.neuron_number)
1. 对于深度网络，每一层的初始值应当用pre_training进行设定，初值已经较好，bp仅作为调整作用。梯度消失不可避免，应当用设定更好的初值来解决。
1. 网络的weight应当保持在一个很小的值(起码对于ann是这样的)。如果ｂｐ算法的公式错误，会导致weight变得很大，在程序中的表现就是：
        1. exp(x)会溢出
        1. ann在mnist上的准确率到87%就不再上升
1. 对于激活函数sigmoid = 1/(1 + e ** -x)，如果ｘ过大，会导致输出始终为１。通过归一化输入来解决, mnist为0-255的灰度图，可以归一化为0-1
1. 对于输出归一化后，要在反向传播的过程中求归一化函数的导数
