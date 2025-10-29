import processEmail as pe
import emailFeatures as ef
from sklearn import svm
import scipy.io as scio



print('Start training SVC...')
data = scio.loadmat('data/spamTrain.mat')
X = data['X']
y = data['y'].flatten()

c = 0.1
clf = svm.SVC(C = c, kernel='linear')
clf.fit(X, y)

file_names = ['data/emailSample2.txt', 'data/spamSample1.txt', 'data/spamSample2.txt']

for file_name in file_names:

    file_contents = open(file_name, 'r').read()
    word_indices = pe.process_email(file_contents)

    features = ef.email_features(word_indices)

    pred = clf.predict(features.reshape(1, -1))[0]

    result = 'Spam' if int(pred) == 1 else 'Email'
    print(f'predict ended: the result of {file_name} is {result}')