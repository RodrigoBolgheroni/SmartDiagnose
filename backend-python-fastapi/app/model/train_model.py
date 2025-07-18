import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras import Sequential, layers,regularizers
from keras.callbacks import EarlyStopping
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.utils.preprocess import preprocessamento as preprocess

def avaliar_modelo(y_true, y_pred, le):
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    classes_presentes = np.unique(np.concatenate((y_true, y_pred)))
    target_names = le.classes_[classes_presentes]

    print("Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))


def treinar_modelo(caminho_csv: str, salvar_modelo: bool = True):
    df = preprocess(caminho_csv)

    Y = df['doencas']
    X = df.drop(['doencas'], axis=1)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    Y_onehot = to_categorical(Y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_onehot, test_size=0.2, random_state=42)

    model = Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],),
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(Y_onehot.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=16,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    avaliar_modelo(y_true, y_pred, le)

    if salvar_modelo:
        pasta_modelo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model'))
        if not os.path.exists(pasta_modelo):
            os.makedirs(pasta_modelo)

        caminho_modelo = os.path.join(pasta_modelo, 'model.keras')
        model.save(caminho_modelo)
        print(f"Modelo salvo em: {caminho_modelo}")

        caminho_scaler = os.path.join(pasta_modelo, 'scaler.pkl')
        joblib.dump(scaler, caminho_scaler)
        print(f"Scaler salvo em: {caminho_scaler}")

        caminho_encoder = os.path.join(pasta_modelo, 'label_encoder.pkl')
        joblib.dump(le, caminho_encoder)
        print(f"Label encoder salvo em: {caminho_encoder}")

        caminho_feature = os.path.join(pasta_modelo, 'features.pkl')
        features = list(df.drop(columns=["doencas"]))  
        joblib.dump(features, caminho_feature)
        print(f"Features salvas em: {caminho_feature}")

    return model, le, scaler, history

if __name__ == '__main__':
    caminho_csv = os.path.join(os.path.dirname(__file__), '../../dataset/data_pt.csv')
    treinar_modelo(caminho_csv)
