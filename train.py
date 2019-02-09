import keras
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler
from config import log_dir
import matplotlib.pyplot as plt
from config import Model_Config


# 调整学习速率
def scheduler(epoch, lr):
    # 每隔1个epoch，学习率减小为原来的1/10
    if epoch == 0:
        lr = 0.001
        print("lr changed to {}".format(0.001))
    if epoch != 0:
        lr = lr * 0.1
        print("lr changed to {}".format(lr * 0.1))
    return


reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)

tensorboard = TensorBoard(log_dir=log_dir)

checkpoint = ModelCheckpoint(Model_Config.model_checkpoint_saved_filepath, verbose=0,
                             monitor='val_loss', save_best_only=True, mode='auto')
callback_lists = [tensorboard, checkpoint]


def train(X_train_processed, X_val_processed, Y_train_list, Y_val_list, vocabulary_size, label_distribute_dict_list):
    from config import Model_Config
    from han_model import HAN_Model

    loss_dict = {}
    loss_weights_dict = {}
    class_weights_dict = {}
    Y_dict = {}
    for i in range(Model_Config.mutil_layers_num):
        loss_dict['preds_{}'.format(i)] = 'categorical_crossentropy'
        loss_weights_dict['preds_{}'.format(i)] = 1.
        Y_dict['preds_{}'.format(i)] = Y_train_list[i][:]
        class_weights_dict['preds_{}'.format(i)] = label_distribute_dict_list[Model_Config.current_trained_labels[i]]

    model = HAN_Model(vocabulary_size).build(Model_Config.mutil_layers_num, Model_Config.optimizer, loss_dict,
                                             loss_weights_dict,
                                             metrics=['accuracy'])
    # trained it
    history = model.fit(X_train_processed, Y_dict,
                        validation_data=([X_val_processed], Y_val_list),
                        epochs=Model_Config.epochs,
                        batch_size=Model_Config.batch_size,
                        class_weight=class_weights_dict,
                        callbacks=callback_lists)

    model.save(Model_Config.model_saved_filepath)
    return model, history


if __name__ == '__main__':
    from prepare import prepare

    # prepare
    X_train_processed, X_val_processed, Y_train_list, Y_val_list, vocabulary_size, label_distribute_dict_list = prepare()
    model, history = train(X_train_processed, X_val_processed, Y_train_list, Y_val_list, vocabulary_size,
                           label_distribute_dict_list)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
