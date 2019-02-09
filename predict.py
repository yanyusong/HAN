from attention_layer import AttentionWithContext
from keras.models import load_model
import numpy as np
from config import Model_Config


def predict(X_val_processed, model_file):
    print('开始 load compile model...')
    model = load_model(model_file, custom_objects={'AttentionWithContext': AttentionWithContext})

    loss_dict = {}
    loss_weights_dict = {}
    for i in range(Model_Config.mutil_layers_num):
        loss_dict['preds_{}'.format(i)] = 'categorical_crossentropy'
        loss_weights_dict['preds_{}'.format(i)] = 1.

    model.compile(optimizer=Model_Config.optimizer,
                  loss=loss_dict,
                  loss_weights=loss_weights_dict,
                  metrics=['accuracy'])

    print('开始 predict ...')
    Y_val_pred = model.predict(X_val_processed, verbose=1)
    preds = [np.argmax(pred, axis=1) for pred in Y_val_pred]
    return preds
