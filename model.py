# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.svm import SVC

dataset = pd.read_csv(r'C:\Users\casper\Desktop\price.csv')
dataset['bed_room'].fillna(0, inplace=True)
dataset['area'].fillna(dataset['area'].mean(), inplace=True)
X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['bed_room'] = X['bed_room'].apply(lambda x : convert_to_int(x))
y = dataset.iloc[:, -1]

SVM_model = SVC()
#Fitting model with trainig data
SVM_model.fit(X, y)
# Saving model to disk
pickle.dump(SVM_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2200, 5]]))
