from keras_model import DRML
from data_utility import ConstructSet, statistics, calc_f1_score, cal_multi_label_accuracy
from load_data import load_batch_from_sets, DataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# constorm metrics
from keras import backend as K

def multi_label_accuracy_threshold(threshold=0.5, interest_class=0):
    def multi_label_accuracy(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        accuracy = K.mean(K.equal(y_true, y_pred), axis=0)
        num_classes = y_pred.shape[1]
        if interest_class < num_classes:
            output_accuracy = accuracy[interest_class]
        elif interest_class == num_classes:
            output_accuracy = K.mean(accuracy)
        else:
            assert False
        return output_accuracy
    return multi_label_accuracy 

def f1_threshold(threshold=0.5, interest_class=0):
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            threshold_value = threshold
            y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            threshold_value = threshold
            y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        num_classes = y_pred.shape[1]
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        f1_score = 2*((precision*recall)/(precision+recall+K.epsilon()))
        if interest_class < num_classes:
            output_value = f1_score[interest_class]
        elif interest_class == num_classes:
            output_value = K.mean(f1_score)
        else:
            assert False
        return output_value
      
    return f1
                        
# generator with random batch load (train)
def generator_train_data(train_set, batch_size, img_cols, img_rows):

    while True:
        x, y = load_batch_from_sets(train_set, batch_size, img_cols, img_rows)
        yield x, y

def generator_val_data(val_set, batch_size, img_cols, img_rows):
    
    while True:
        x, y = load_batch_from_sets(val_set, batch_size, img_cols, img_rows)
        yield x, y


def training_val(train_information, val_information, num_au_labels, 
        val_ratio, num_epochs, batch_size, img_cols, img_rows):
    # load data folders
    params = {'batch_size': batch_size,
              'img_cols': img_cols,
              'img_rows': img_rows,
              'num_au_labels': num_au_labels,
              'shuffle': True}
    training_generator = DataGenerator(train_information, **params)
    validation_generator = DataGenerator(val_information, **params)
    patience = 20

    model = DRML((img_cols, img_rows, 3), num_au_labels)
    model.summary()
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', 
        metrics=[f1_threshold(0.8, 10), multi_label_accuracy_threshold(0.8, 10),
            f1_threshold(0.5, 10), multi_label_accuracy_threshold(0.5, 10)])

    model.fit_generator(
        generator=training_generator,
        epochs=num_epochs,
        verbose=1,
        validation_data=validation_generator,
        callbacks=[EarlyStopping(patience=patience),
                    ModelCheckpoint("train_model/DRML_{epoch:03d}-{val_multi_label_accuracy:.5f}.hdf5",
                        save_best_only=True)])
    
    # evaluation validation data
    evaluation_val_data(val_information, batch_size, model, img_cols, img_rows, 0.8)
    evaluation_val_data(val_information, batch_size, model, img_cols, img_rows, 0.5)


def evaluation_val_data(val_data, batch_size, model, img_cols, img_rows, threshold):
    val_y_pred = np.zeros(shape=(len(val_data), len(val_data[0][1])), dtype=np.float32)
    val_y_true = np.zeros(shape=(len(val_data), len(val_data[0][1])), dtype=np.float32)
    steps = (len(val_data) + batch_size - 1) // batch_size
    for step in range(steps):
        if step * batch_size + batch_size < len(val_data):
            data_info = val_data[step * batch_size: (step + 1) * batch_size]
        else:
            data_info = val_data[step * batch_size: len(val_data)]
        image_batch, label_batch = load_batch_from_sets(data_info, img_cols, img_rows)
        y_pred = model.predict(image_batch)
        y_true = label_batch
        if step * batch_size + batch_size < len(val_data):
            val_y_pred[step * batch_size: (step + 1) * batch_size] = y_pred
            val_y_true[step * batch_size: (step + 1) * batch_size] = y_true
        else:
            val_y_pred[step * batch_size: len(val_data)] = y_pred
            val_y_true[step * batch_size: len(val_data)] = y_true
    assert val_y_pred.shape == val_y_true.shape
    statistic_list = statistics(val_y_pred, val_y_true, threshold)
    mean_f1_score, f1_score_list = calc_f1_score(statistic_list)
    mean_accuracy, accuray_list = cal_multi_label_accuracy(val_y_true, val_y_pred, threshold)
    print('Threshold: {}, mean_f1_score: {}'.format(threshold, mean_f1_score))
    print('f1_score_list: ', f1_score_list)
    print('Threshold: {}, mean_accuracy: {}'.format(threshold, mean_accuracy))
    print('accuracy_list: ', accuray_list)



        
    

if __name__ == '__main__':
    data_dir = '/home/user/Documents/dataset/DISFA'
    num_au_labels = 10
    val_ratio = 0.1
    num_epochs = 1000
    batch_size = 128
    img_cols, img_rows = (170, 170)

    cset = ConstructSet(data_dir, num_au_labels, val_ratio)
    train_information = cset.train
    val_information = cset.val
    training_val(train_information, val_information, num_au_labels, 
        val_ratio, num_epochs, batch_size, img_cols, img_rows)    



