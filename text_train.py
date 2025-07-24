
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def Data_Preprocessing():
    data = pd.read_csv(r"dyslexia_dataset.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    
    #data['Attendance'] = le.fit_transform(data['Attendance'])
    
    data['Confidence'] = le.fit_transform(data['Confidence'])
    
    data['Participation'] = le.fit_transform(data['Participation'])

    data['Health_Issues'] = le.fit_transform(data['Health_Issues'])
    
    data['Distraction'] = le.fit_transform(data['Distraction'])
    
    data['Pronunciation_Issues'] = le.fit_transform(data['Pronunciation_Issues'])
    
    data['Reading_Fluency'] = le.fit_transform(data['Reading_Fluency'])
    
    data['Writing_Legibility'] = le.fit_transform(data['Writing_Legibility'])
    
    data['Math_Struggles'] = le.fit_transform(data['Math_Struggles'])
    
    data['Memory_Issues'] = le.fit_transform(data['Memory_Issues'])
    
  

    """Feature Selection => Manual"""
    x = data.drop(['Dyslexia_Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Dyslexia_Label']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

def Model_Training():
    data = pd.read_csv(r"dyslexia_dataset.csv")
    data.head()

    data = data.dropna()


    """One Hot Encoding"""

    le = LabelEncoder()
    
     
    data['Confidence'] = le.fit_transform(data['Confidence'])
    
    data['Participation'] = le.fit_transform(data['Participation'])

    data['Health_Issues'] = le.fit_transform(data['Health_Issues'])
    
    data['Distraction'] = le.fit_transform(data['Distraction'])
    
    data['Pronunciation_Issues'] = le.fit_transform(data['Pronunciation_Issues'])
    
    data['Reading_Fluency'] = le.fit_transform(data['Reading_Fluency'])
    
    data['Writing_Legibility'] = le.fit_transform(data['Writing_Legibility'])
    
    data['Math_Struggles'] = le.fit_transform(data['Math_Struggles'])
    
    data['Memory_Issues'] = le.fit_transform(data['Memory_Issues'])
    
  

    """Feature Selection => Manual"""
    x = data.drop(['Attendance','Dyslexia_Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Dyslexia_Label']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=9)


   
#############svm###############
    
    from sklearn.svm import SVC
   
    svcclassifier = SVC(kernel='linear',random_state=9)
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)
    
    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    print("Confusion Matrix :")
    cm = confusion_matrix(y_test,y_pred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix

    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    
  
    from joblib import dump
    dump (svcclassifier,"model.joblib")
    print("Model saved as model.joblib")

Model_Training()