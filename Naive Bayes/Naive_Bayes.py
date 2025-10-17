import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Naive_Bayes:
    def __init__(self):
        self.prior_prob={}  #先验概率
        self.condition_prob={}  #条件概率
        self.classes=[]  #Y的分类
        self.features=[]  #X的特征取值
    
    def fit(self,X,Y):
        #计算先验概率
        self.classes=np.unique(Y)   
        samples=len(Y)
        for cls in self.classes:
            self.prior_prob[cls]=np.sum(Y==cls)/samples
            self.condition_prob[cls] = {}

        self.features=[np.unique(X[:,i]) for i in range(X.shape[1])]

        #计算条件概率
        for cls in self.classes:
            cls_sample=X[Y==cls]
            cls_count=len(cls_sample)

            for i in range(X.shape[1]):
                feature_values=self.features[i]
                for feature in feature_values:
                    count=np.sum(cls_sample[:,i]==feature)
                    prob=(count+1)/(cls_count+len(feature_values))  #拉普拉斯光滑
                    self.condition_prob[cls][(i,feature)]=prob

    def predict(self,X):
        pred=[]
        for sample in X:
            probs={}

            for cls in self.classes:
                prob=np.log(self.prior_prob[cls])  #取对数相加

                for (i,value) in enumerate(sample):
                    if (i,value) not in self.condition_prob[cls]:
                        prob+=np.log(1/(len(self.features[i]))) 

                    else:    
                        prob+=np.log(self.condition_prob[cls][(i,value)])

                probs[cls]=prob

            pred.append(max(probs,key=probs.get))

        return np.array(pred)
    


def load_data_from_csv(csv_path):

    df=pd.read_csv(csv_path)
    features=df.columns[:-1]
    labels=df.columns[-1]
    
    #将连续特征值转换为离散值
    for feature in features:
        df[feature]=pd.cut(df[feature],bins=3,labels=[0,1,2])  

    X=df[features].values.tolist()
    Y=df[labels].values.tolist()

    return np.array(X), np.array(Y)

def train_test_split(X,Y,test_size=0.3,random_state=42):
    np.random.seed(random_state)

    indices=np.random.permutation(len(X))
    test_size=int(len(X)*test_size)

    test_indices=indices[:test_size]
    train_indices=indices[test_size:]

    return X[train_indices],X[test_indices],Y[train_indices],Y[test_indices]

def cal_accuracy(Y_true,Y_pred):
    count=0
    for i in range(len(Y_true)):
        if Y_pred[i]==Y_true[i]:
            count+=1
    print(f"test_y:{len(Y_true)} in total, accurate:{count} in total\n")
    return count/len(Y_true)

def visualize_data(X,Y,classes):
    plt.figure(figsize=(15,10))

    #类别条形图
    plt.subplot(2,3,1)
    unique,counts=np.unique(Y,return_counts=True)
    plt.bar(unique,counts,color='b')
    plt.title('labels')
    plt.xlabel('label frequency')

    #特征直方图
    for i in range(X.shape[1]):
        plt.subplot(2, 3, i+2)
        for idx, cls in enumerate(classes):
            plt.hist(X[Y==cls, i], bins=15, alpha=0.6, 
                     label=classes[idx],
                     edgecolor='white', linewidth=0.5)
        
        plt.title(f'feature[{i}]', fontsize=12, pad=10)
        plt.xlabel('feature value', fontsize=10)
        plt.ylabel('frequency', fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(fontsize=9, loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def visualize_results(Y_true, Y_pred, classes):
    plt.figure(figsize=(12, 5))
    
    #混淆矩阵热力图
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(Y_true, Y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'class{c}' for c in classes],
                yticklabels=[f'class{c}' for c in classes])
    plt.title('Comfusion Matrix')
    plt.xlabel('class pred')
    plt.ylabel('class true')
    
    # 预测结果与真实结果对比条形图
    plt.subplot(1, 2, 2)
    x = np.arange(len(classes))
    true_counts = np.array([np.sum(Y_true==cls) for cls in classes])
    pred_counts = np.array([np.sum(Y_pred==cls) for cls in classes])
    width = 0.35
    plt.bar(x - width/2, true_counts, width, label='true')
    plt.bar(x + width/2, pred_counts, width, label='pred')
    plt.title('Prediction Result')
    plt.xlabel('class')
    plt.ylabel('samples')
    plt.xticks(x, [f'class{c}' for c in classes])
    plt.legend()
    
    plt.tight_layout()
    plt.show()



def main():
    X,Y=load_data_from_csv(csv_path='iris.csv')
    train_X,test_X,train_Y,test_Y=train_test_split(X,Y)
    classes=np.unique(Y)
    visualize_data(X,Y,classes)

    nb=Naive_Bayes()
    nb.fit(train_X,train_Y)
    pred_y=nb.predict(test_X)

    accuracy=cal_accuracy(test_Y,pred_y)
    print(f"Accuracy:{accuracy:.2f}")
    visualize_results(test_Y,pred_y,classes)

if __name__=='__main__':
    main()




