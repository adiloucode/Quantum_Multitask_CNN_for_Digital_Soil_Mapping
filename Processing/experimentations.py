import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import pickle
from sklearn.preprocessing import StandardScaler

def rmse(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, 1))
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2_metric(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, 1))
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

def preparation(input_name, output_name, normalized):
    X = np.load(input_name)
    y = np.load(output_name)
    y = y.reshape(y.shape[0], y.shape[1])
    num_samples = X.shape[0]
    X_reshaped = X.reshape(num_samples, -1)

    if normalized == "zscore":
        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)
        return X_scaled, y_scaled

    elif normalized == "MinMax":
        x_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y)
        return X_scaled, y_scaled

def Ph_basic_cnn(input_shape, output_dim):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_dim, activation='linear'))
    return model

def build_ph_multi_task_cnn(input_shape):
    input_layer = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = []
    for _ in range(6):
        branch = layers.Dense(32, activation='relu')(x)
        branch = layers.Dense(1, activation='linear')(branch)
        outputs.append(branch)
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[
        ['mae', rmse, r2_metric],
        ['mae', rmse, r2_metric],
        ['mae', rmse, r2_metric],
        ['mae', rmse, r2_metric],
        ['mae', rmse, r2_metric],
        ['mae', rmse, r2_metric],
    ])
    return model

def A_basic_cnn(input_shape, output_dim):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(output_dim, activation='linear'))
    return model

def build_A_multi_task_cnn(input_shape=(5, 5, 36)):
    input_layer = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='softmax')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # better for small inputs than Flatten
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = []
    for _ in range(6):
        branch = layers.Dense(64, activation='softmax')(x)
        branch = layers.Dropout(0.25)(branch)
        branch = layers.Dense(128, activation='linear')(x)
        branch = layers.Dropout(0.4)(branch)
        branch = layers.Dense(1, activation='linear')(branch)
        outputs.append(branch)

    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(
        optimizer="Adam",
        loss='mse',
        loss_weights=[1.0]*6,
        metrics=[['mae', rmse, r2_metric]]*6)
    return model

def CE_basic_cnn(input_shape, output_dim):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(output_dim, activation='linear'))
    return model

def build_ce_multi_task_cnn(input_shape=(5, 5, 36)):
    input_layer = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = []
    for _ in range(6):
        branch = layers.Dense(32, activation='relu')(x)
        branch = layers.Dense(1, activation='linear')(branch)
        outputs.append(branch)

    model = Model(inputs=input_layer, outputs=outputs)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        loss_weights=[1.0]*6,
        metrics=[['mae', rmse, r2_metric]]*6)
    return model

def fit_models(input_file, output_file, input_shape, component_name, method, norm, epochs, batch_size, k_folds):
    position = order.index(component_name)
    X, y = preparation(input_file, output_file, norm)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_histories = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]

        if method == "multitask":
            y_train = [y[train_index, i] for i in range(6)]
            y_test = [y[test_index, i] for i in range(6)]
        else:
            y_train, y_test = y[train_index], y[test_index]

        if method == "basic":
            model = Expirements_Functions[position][0](X_train.shape[1:], y.shape[1])
            model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse, r2_metric])
        elif method == "multitask":
            model = Expirements_Functions[position][1](X_train.shape[1:])

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, batch_size=batch_size, verbose=0)
        fold_histories.append(history.history)

    return fold_histories

EPOCHS = 70
BATCH_SIZE = 8
K_FOLDS = 5

order = []
Expirements_Functions = []
order.append("PH")
Expirements_Functions.append((Ph_basic_cnn, build_ph_multi_task_cnn))
order.append("A%")
Expirements_Functions.append((A_basic_cnn , build_A_multi_task_cnn))
order.append("CE")
Expirements_Functions.append((CE_basic_cnn , build_ce_multi_task_cnn))


files = [

         ("Data/aug_5x5.npy" , "Data/aug/outputA_.npy" , (5,5,36),"A%" , "multitask" , "zscore" , "aug"),
         ("Data/aug_5x5.npy" , "Data/aug/outputA_.npy" , (5,5,36),"A%" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputA_.npy" , (7,7,36),"A%" , "multitask", "zscore" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputA_.npy" , (7,7,36),"A%" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputA_.npy" , (9,9,36),"A%" , "multitask","zscore" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputA_.npy" , (9,9,36),"A%" , "multitask","MinMax" , "aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputA_.npy" , (5,5,36),"A%" , "multitask" , "zscore" , "No_aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputA_.npy" , (5,5,36),"A%" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputA_.npy" , (7,7,36),"A%" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputA_.npy" , (7,7,36),"A%" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputA_.npy" , (9,9,36),"A%" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputA_.npy" , (9,9,36),"A%" , "multitask","MinMax" , "No_aug"),

         ("Data/aug_5x5.npy" , "Data/aug/outputCE.npy" , (5,5,36),"PH" , "multitask" , "zscore" , "aug"),
         ("Data/aug_5x5.npy" , "Data/aug/outputCE.npy" , (5,5,36),"PH" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputCE.npy" , (7,7,36),"PH" , "multitask", "zscore" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputCE.npy" , (7,7,36),"PH" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputCE.npy" , (9,9,36),"PH" , "multitask","zscore" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputCE.npy" , (9,9,36),"PH" , "multitask","MinMax" , "aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputCE.npy" , (5,5,36),"PH" , "multitask" , "zscore" , "No_aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputCE.npy" , (5,5,36),"PH" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputCE.npy" , (7,7,36),"PH" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputCE.npy" , (7,7,36),"PH" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputCE.npy" , (9,9,36),"PH" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputCE.npy" , (9,9,36),"PH" , "multitask","MinMax" , "No_aug"),

         ("Data/aug_5x5.npy" , "Data/aug/outputCEC.npy" , (5,5,36),"PH" , "multitask" , "zscore" , "aug"),
         ("Data/aug_5x5.npy" , "Data/aug/outputCEC.npy" , (5,5,36),"PH" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputCEC.npy" , (7,7,36),"PH" , "multitask", "zscore" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputCEC.npy" , (7,7,36),"PH" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputCEC.npy" , (9,9,36),"PH" , "multitask","zscore" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputCEC.npy" , (9,9,36),"PH" , "multitask","MinMax" , "aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputCEC.npy" , (5,5,36),"PH" , "multitask" , "zscore" , "No_aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputCEC.npy" , (5,5,36),"PH" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputCEC.npy" , (7,7,36),"PH" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputCEC.npy" , (7,7,36),"PH" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputCEC.npy" , (9,9,36),"PH" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputCEC.npy" , (9,9,36),"PH" , "multitask","MinMax" , "No_aug"),

         ("Data/aug_5x5.npy" , "Data/aug/outputOM_.npy" , (5,5,36),"PH" , "multitask" , "zscore" , "aug"),
         ("Data/aug_5x5.npy" , "Data/aug/outputOM_.npy" , (5,5,36),"PH" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputOM_.npy" , (7,7,36),"PH" , "multitask", "zscore" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputOM_.npy" , (7,7,36),"PH" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputOM_.npy" , (9,9,36),"PH" , "multitask","zscore" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputOM_.npy" , (9,9,36),"PH" , "multitask","MinMax" , "aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputOM_.npy" , (5,5,36),"PH" , "multitask" , "zscore" , "No_aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputOM_.npy" , (5,5,36),"PH" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputOM_.npy" , (7,7,36),"PH" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputOM_.npy" , (7,7,36),"PH" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputOM_.npy" , (9,9,36),"PH" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputOM_.npy" , (9,9,36),"PH" , "multitask","MinMax" , "No_aug"),

         ("Data/aug_5x5.npy" , "Data/aug/outputL_.npy" , (5,5,36),"A%" , "multitask" , "zscore" , "aug"),
         ("Data/aug_5x5.npy" , "Data/aug/outputL_.npy" , (5,5,36),"A%" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputL_.npy" , (7,7,36),"A%" , "multitask", "zscore" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputL_.npy" , (7,7,36),"A%" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputL_.npy" , (9,9,36),"A%" , "multitask","zscore" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputL_.npy" , (9,9,36),"A%" , "multitask","MinMax" , "aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputL_.npy" , (5,5,36),"A%" , "multitask" , "zscore" , "No_aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputL_.npy" , (5,5,36),"A%" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputL_.npy" , (7,7,36),"A%" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputL_.npy" , (7,7,36),"A%" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputL_.npy" , (9,9,36),"A%" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputL_.npy" , (9,9,36),"A%" , "multitask","MinMax" , "No_aug"),

         ("Data/aug_5x5.npy" , "Data/aug/outputS_.npy" , (5,5,36),"A%" , "multitask" , "zscore" , "aug"),
         ("Data/aug_5x5.npy" , "Data/aug/outputS_.npy" , (5,5,36),"A%" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputS_.npy" , (7,7,36),"A%" , "multitask", "zscore" , "aug"),
         ("Data/aug_7x7.npy" , "Data/aug/outputS_.npy" , (7,7,36),"A%" , "multitask" , "MinMax" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputS_.npy" , (9,9,36),"A%" , "multitask","zscore" , "aug"),
         ("Data/aug_9x9.npy" , "Data/aug/outputS_.npy" , (9,9,36),"A%" , "multitask","MinMax" , "aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputS_.npy" , (5,5,36),"A%" , "multitask" , "zscore" , "No_aug"),
         ("Data/No_aug_5x5.npy" , "Data/No_aug/outputS_.npy" , (5,5,36),"A%" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputS_.npy" , (7,7,36),"A%" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_7x7.npy" , "Data/No_aug/outputS_.npy" , (7,7,36),"A%" , "multitask" , "MinMax" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputS_.npy" , (9,9,36),"A%" , "multitask","zscore" , "No_aug"),
         ("Data/No_aug_9x9.npy" , "Data/No_aug/outputS_.npy" , (9,9,36),"A%" , "multitask","MinMax" , "No_aug"),

        ] 

for i in files:
    print(f"========{i}========")
    histories = fit_models(i[0], i[1], i[2], i[3], i[4], i[5], EPOCHS, BATCH_SIZE, K_FOLDS)
    path = f'{i[0]}_{i[1]}_{i[2]}_{i[3]}_{i[4]}_{i[5]}_{i[6]}_kfold.pkl'.replace("/", "")
    with open(f'Data/results/{path}', 'wb') as f:
        pickle.dump(histories, f)
