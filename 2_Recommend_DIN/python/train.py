# -*- coding: utf-8 -*


import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Accuracy

from model import DIN
from utils import *

import os

# random.seed(2020)


# 设置log输出信息，程序运行时系统打印的信息。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置为2时，屏蔽通知信息和警告信息
# 如果电脑有多个GPU，tensorflow默认全部使用。如果想只使用部分GPU，可以设置CUDA_VISIBLE_DEVICES。在调用python程序时，可以使用
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 终端执行程序时设置使用的GPU,仅设备6可见

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    path = sys.path[0]
    file = '/../data/remap.csv'
    maxlen = 20
    embed_dim = 8
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'
    ffn_activation = 'prelu'

    learning_rate = 0.001
    batch_size = 128
    epochs = 20
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = create_amazon_electronic_dataset(path + file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test

    # ============================Build Model==========================
    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
                ffn_activation, maxlen, dnn_dropout)

    # model.summary()

    # ============================model checkpoint======================
    # 检查值，是否保存每个epoch的值
    check_path = '../save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                    verbose=1, period=5)  # save_best_only=True时，只保存验证集上性能最好的模型
    # =========================Compile============================
    # 编译创建好的模型，模型搭建完后，对网络的学习过程进行配置，否则调用fit时会抛出异常
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])  # Accuracy()
    # ===========================Fit==============================
    # EarlyStopping的patience要比ReduceLROnPlateau的patience大一些，使训练轮数增加
    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # 早停,patience能够容忍几个epoch内没有improvement
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.01, verbose=1)  # 调整学习率ReduceLROnPlateau，优化lr
    ]

    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=callbacks,  # 导入上述callbacks
        ## callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        checkpoint=checkpoint,
        validation_data=(val_X, val_y),
        batch_size=batch_size,
    )
    model.save(path + '/../DIN.model')

    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])
    # print(model.predict(test_X))
