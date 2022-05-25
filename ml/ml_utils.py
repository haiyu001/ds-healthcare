from typing import Optional, List, Any, Tuple
from sklearn.model_selection import train_test_split
from utils.general_util import save_pdf
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import logging
import os


def get_confusion_matrix_array(actual_labels: List[int],
                               predicted_labels: List[int],
                               class_names: List[str],
                               save_dir: str) -> np.ndarray:
    num_classes = len(class_names)
    level_labels = [[0] * num_classes, list(range(num_classes))]
    confusion_matrix_count_array = metrics.confusion_matrix(y_true=actual_labels, y_pred=predicted_labels)
    confusion_matrix_count_pdf = pd.DataFrame(
        data=confusion_matrix_count_array,
        columns=pd.MultiIndex(levels=[["Predicted:"], class_names], codes=level_labels),
        index=pd.MultiIndex(levels=[["Actual:"], class_names], codes=level_labels)
    )
    confusion_matrix_percentage_pdf = confusion_matrix_count_pdf.astype("float64") / \
                                      confusion_matrix_count_pdf.sum(axis=1).values[:, np.newaxis]
    save_pdf(confusion_matrix_count_pdf, os.path.join(save_dir, "confusion_matrix_count.csv"), csv_index=True)
    save_pdf(confusion_matrix_percentage_pdf, os.path.join(save_dir, "confusion_matrix_percentage.csv"), csv_index=True)
    return confusion_matrix_count_array


def save_confusion_matrix(confusion_matrix_array: np.ndarray,
                          class_names: List[str],
                          save_dir: Optional[str] = None):
    confusion_matrix_percentage_array = confusion_matrix_array.astype("float") / \
                                        confusion_matrix_array.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8.8, 6.6))
    plt.imshow(confusion_matrix_percentage_array, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = confusion_matrix_percentage_array.max() / 2.
    for i, j in itertools.product(range(confusion_matrix_percentage_array.shape[0]),
                                  range(confusion_matrix_percentage_array.shape[1])):
        plt.text(j, i, f"{confusion_matrix_percentage_array[i, j] * 100:.1f}%", horizontalalignment="center",
                 color="white" if confusion_matrix_percentage_array[i, j] > thresh else "black")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()


def save_classification_report(actual_labels: List[int],
                              predicted_labels: List[int],
                              class_names: List[str],
                              save_dir: str):
    report_txt = metrics.classification_report(actual_labels, predicted_labels, target_names=class_names)
    report_filepath = os.path.join(save_dir, "classification_report.txt")
    with open(report_filepath, "w", encoding="utf-8") as output:
        output.write(report_txt)


def get_callback_list(model_dir: str, early_stop: Optional[int] = None) -> List[Any]:
    callback_list = [
        tf.keras.callbacks.CSVLogger(os.path.join(model_dir, "epoch_results.csv"), append=True),
        tf.keras.callbacks.TensorBoard(os.path.join(model_dir, "logs")),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "final_model.hdf5"),
                                           monitor="acc", save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "best_model.hdf5"),
                                           monitor="val_acc", save_best_only=True)]
    if early_stop:
        callback_list.append(tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=early_stop))
    return callback_list


def _split_train_val_test(features_pdf: pd.DataFrame, val_size: float, test_size: float) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pdf, test_pdf = train_test_split(features_pdf, test_size=test_size, random_state=1)
    train_pdf, val_pdf = train_test_split(train_pdf, test_size=val_size, random_state=1)
    return train_pdf.copy(), val_pdf.copy(), test_pdf.copy()


def get_train_val_test(features_pdf: pd.DataFrame,
                       label_col: str = "label",
                       val_size: float = 0.1,
                       test_size: float = 0.1) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    label_pdf_list = []
    label_groups = features_pdf.groupby(by=label_col)
    for label, label_pdf in label_groups:
        label_train_pdf, label_val_pdf, label_test_pdf = _split_train_val_test(label_pdf, val_size, test_size)
        label_train_pdf["type"] = "train"
        label_val_pdf["type"] = "val"
        label_test_pdf["type"] = "test"
        label_pdf_list.extend([label_train_pdf, label_val_pdf, label_test_pdf])
    train_val_test_pdf = pd.concat(label_pdf_list).sample(frac=1.0)

    feature_cols = [i for i in features_pdf.columns if i != label_col]
    Xtrain_ytrain_Xval_yval_Xtest_ytest = []
    for t in ["train", "val", "test"]:
        type_df = train_val_test_pdf[train_val_test_pdf["type"] == t]
        logging.info(f"{t}: {type_df[feature_cols].shape}")
        Xtrain_ytrain_Xval_yval_Xtest_ytest.append(type_df[feature_cols])
        Xtrain_ytrain_Xval_yval_Xtest_ytest.append(type_df[label_col])
    return tuple(Xtrain_ytrain_Xval_yval_Xtest_ytest)
