import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Cardiovascular_Disease.csv')


# cleaning
age_data = df['age']//365
df['age'] = age_data

data_filter = df[(df['height'].between(125, 200)) & 
                (df['weight'].between(40, 120)) & 
                (df['ap_hi'].between(100, 200)) & 
                (df['ap_lo'].between(50, 90))
    ]
df = data_filter


## Model
def cardio():
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
        solver = 'lbfgs',
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_scale, Y_train)

    Y_pred = model.predict(X_test_scale)

    cr = classification_report(Y_test, Y_pred, output_dict=True)
    cm = confusion_matrix(Y_test, Y_pred)
    
    return features, scaler, model, Y_pred, cr, cm


def svm_cardio():
    return 'cardio_model'