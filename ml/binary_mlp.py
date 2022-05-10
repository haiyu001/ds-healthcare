from typing import List, Optional, Dict
from ml.ml_utils import get_callback_list, save_confusion_matrix, save_classification_report, get_confusion_matrix_array
from utils.general_util import make_dir
import pandas as pd
import tensorflow as tf
import os


class BinaryModel(object):

    def __init__(self,
                 model_dir: str,
                 input_dimension: int,
                 class_names: List[str],
                 learning_rate: float = 0.0001,
                 dropout_rate: float = 0.4,
                 first_hidden_layer_size: int = 64,
                 second_hidden_layer_size: int = 32,
                 epochs: int = 50,
                 batch_size: int = 32,
                 threshold: int = 0.5
                 ):
        self.class_names = class_names
        self.input_dimension = input_dimension
        self.activation_1 = "relu"
        self.activation_2 = "relu"
        self.activation_3 = "sigmoid"
        self.loss = "binary_crossentropy"
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_dir = model_dir
        self.dropout_rate = dropout_rate
        self.first_hidden_layer_size = first_hidden_layer_size
        self.second_hidden_layer_size = second_hidden_layer_size

    def build_model(self) -> tf.keras.Sequential:
        mlp_model = tf.keras.Sequential()
        mlp_model.add(tf.keras.layers.Dense(self.first_hidden_layer_size, activation=self.activation_1,
                                            input_dim=self.input_dimension))
        mlp_model.add(tf.keras.layers.Dropout(self.dropout_rate))
        mlp_model.add(tf.keras.layers.Dense(self.second_hidden_layer_size, activation=self.activation_2))
        mlp_model.add(tf.keras.layers.Dropout(self.dropout_rate))
        mlp_model.add(tf.keras.layers.Dense(1, activation=self.activation_3))
        mlp_model.compile(metrics=["acc"], loss=self.loss, optimizer=self.optimizer)
        return mlp_model

    def train_model(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: pd.DataFrame,
                    y_val: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    class_weight: Optional[Dict[int, float]] = None):
        callback_list = get_callback_list(self.model_dir)
        mlp_model = self.build_model()
        mlp_model.fit(x=X_train,
                      y=y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      verbose=2,
                      validation_data=(X_val, y_val),
                      callbacks=callback_list,
                      class_weight=class_weight)  # class_weight={0: 0.75, 1:0.25}
        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        best_model = tf.keras.models.load_model(os.path.join(self.model_dir, "best_model.hdf5"))
        final_model = tf.keras.models.load_model(os.path.join(self.model_dir, "final_model.hdf5"))
        best_model_evaluation_dir = os.path.join(self.model_dir, "best_model")
        final_model_evaluation_dir = os.path.join(self.model_dir, "final_model")

        for model, model_evaluation_dir in \
                zip([best_model, final_model], [best_model_evaluation_dir, final_model_evaluation_dir]):
            make_dir(model_evaluation_dir)
            predictions = model.predict(X_test)
            predicted_scores = predictions.squeeze(axis=-1)
            predicted_labels = (predicted_scores > self.threshold).astype("int32")
            save_classification_report(y_test, predicted_labels, self.class_names, model_evaluation_dir)
            confusion_matrix_array = get_confusion_matrix_array(y_test, predicted_labels,
                                                                self.class_names, model_evaluation_dir)
            save_confusion_matrix(confusion_matrix_array, self.class_names, model_evaluation_dir)