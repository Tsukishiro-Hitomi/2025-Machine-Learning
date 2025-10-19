import numpy as np
from Naive_Bayes import load_data_from_csv,train_test_split,cal_accuracy

class GaussianNB:
    def __init__(self):
        self.pri_prob={}
        self.classes=[]
        self.mean={}
        self.stdev={}

    def cal_mean(self,X):
        return np.mean(X)
    
    def cal_stdev(self,X):
        avr=self.cal_mean(X)
        return np.sqrt(np.sum(np.power(X-avr,2))/len(X))
    
    def Gaussian_prob(self,x,mean,stdev):
        return 1/((np.sqrt(2*np.pi))*stdev)*np.exp(pow(x-mean,2)/(-2*pow(stdev,2)))
    
    def fit(self,X,Y):
        self.classes=np.unique(Y)
        for cls in self.classes:
            self.pri_prob[cls]=np.sum(Y==cls)/len(Y)

            self.mean[cls]={}
            self.stdev[cls]={}

            for i in range(X.shape[1]):
                X_cls=X[Y==cls][:,i]
                self.mean[cls][i]=self.cal_mean(X_cls)
                self.stdev[cls][i]=self.cal_stdev(X_cls)
    
    def predict(self,X):
        pred=[]
        for x in X:
            probs={}
            for cls in self.classes:
                prob=np.log(self.pri_prob[cls])
                for i in range(X.shape[1]):
                    prob+=np.log(self.Gaussian_prob(x[i],self.mean[cls][i],self.stdev[cls][i]))
                probs[cls]=prob
            pred.append(max(probs,key=probs.get))
        return np.array(pred)
                
def main():
    X,Y=load_data_from_csv(csv_path='iris.csv')
    train_X,test_X,train_Y,test_Y=train_test_split(X,Y)

    nb=GaussianNB()
    nb.fit(train_X,train_Y)
    pred_y=nb.predict(test_X)

    accuracy=cal_accuracy(test_Y,pred_y)
    print(f"Accuracy:{accuracy:.2f}")

if __name__=='__main__':
    main()
