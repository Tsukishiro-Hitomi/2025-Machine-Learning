# Readme:
**Naive Bayes.pdf:**  根据李航《统计学习方法》第四章总结的笔记，参考了CS229 Part4的讲义；


**Naive__Bayes.py:**  根据笔记实现了一个简单的朴素贝叶斯分类器，并用鸢尾花数据集(iris.csv)进行验证；
在test_size=0.3,random_state=42时，在测试集上的准确率为91%（40/44）；
经过验证，将本数据集x的连续特征划分为3类时分类效果最佳；
用matplotlib将数据(data.png)和预测结果(result.png)可视化，相关函数实现参考了AI回答；


**Naive_Bayes_Gaussian.pdf:**  由于鸢尾花的特征为连续特征，尝试将计算条件概率的方式改为计算高斯分布的概率，其他条件同上。第一次运行时，没有给计算的标准差加epsilon导致预测效果很差，仅有41%(18/44); 修改后预测效果达到了93%(41/44)

**sklearn_GaussianaNB.py:**  直接调用sklearn的GaussianNB模型，其他条件同上，预测效果为91%(40/44)





