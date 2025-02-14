import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gender_guesser.detector as gender
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

def read_datasets():
    """ Reads users profile from CSV files """
    genuine_users = pd.read_csv("data/real_users.csv")
    fake_users = pd.read_csv("data/fake_users.csv")
    x = pd.concat([genuine_users, fake_users])
    y = np.array(len(fake_users) * [0] + len(genuine_users) * [1])
    return x, y

def predict_sex(name):
    d = gender.Detector()
    first_name = name.str.split(' ').str.get(0)
    sex = first_name.apply(d.get_gender)
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    return sex.map(sex_dict).fillna(0).astype(int)

def extract_features(x):
    lang_dict = {name: i for i, name in enumerate(np.unique(x['lang']))}
    x['lang_code'] = x['lang'].map(lang_dict).astype(int)
    x['sex_code'] = predict_sex(x['name'])
    features = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'sex_code', 'lang_code']
    return x[features]

def build_nn(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

import joblib

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = build_nn(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1, validation_split=0.15)
    
    # Save model and scaler
    model.save("saved_model.h5")
    joblib.dump(scaler, "scaler.pkl")
    
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    return y_test, y_pred, model


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    classes = ['Fake', 'Genuine']
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.show()

print("Reading datasets...")
x, y = read_datasets()
print(x.columns)
print(x.describe())

print("Extracting features...")
x = extract_features(x)
print(x.describe())

print("Training model...")
y_test, y_pred, model = train(x, y)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)
plot_roc_curve(y_test, y_pred)
