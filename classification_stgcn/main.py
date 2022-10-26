from Actionsrecognition.ActionsEstLoader import TSSTG
import numpy as np
import warnings
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import pickle


action_model = TSSTG(device='gpu')


def plot_cm(CM, normalize=True, save_dir='', names_x=(), names_y=(), show=True):
    """
    function: plot confusion matrix
    :param CM: array cm
    :param normalize: normaize 0-1
    :param save_dir: path save
    :param names: name class
    :param show: True
    :return:
    """
    if True:
        import seaborn as sn
        array = CM / ((CM.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
        if not normalize:
            array = np.asarray(array, dtype='int')
        fmt = 'd'
        if normalize:
            fmt = '.2f'
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=2.0 if 2 < 50 else 0.8)  # for label size
        labels_x = (0 < len(names_x) < 99) and len(names_x) == 7  # apply names to ticklabels
        labels_y = (0 < len(names_y) < 99) and len(names_y) == 7  # apply names to ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array, annot=2 < 30, annot_kws={"size": 16}, cmap='Blues', fmt=fmt, square=True,
                       xticklabels=names_x if labels_x else "auto",
                       yticklabels=names_y if labels_y else "auto").set_facecolor((1, 1, 1))
        accu = sum([CM[i, i] for i in range(CM.shape[0])])/sum(sum(CM))
        fig.axes[0].set_xlabel('True', fontweight='bold', fontsize=20)
        fig.axes[0].set_ylabel('Predicted', fontweight='bold', fontsize=20)
        plt.title('Accuracy: {}%'.format(round(accu*100)), fontweight='bold', fontsize=20)
        if show:
            plt.show()
        name_save = 'confusion_matrix.png'
        if normalize:
            name_save = 'confusion_matrix_normalize.png'
        fig.savefig(Path(save_dir) / name_save, dpi=300)
        plt.close()


if __name__ == "__main__":

    path_test = '/home/duyngu/Downloads/Dataset_Human_Action/test_no_scale.pkl'
    # Load dataset
    features, labels = [], []
    with open(path_test, 'rb') as f:
        fts, lbs = pickle.load(f)
        features.append(fts)
        labels.append(lbs)
    del fts, lbs

    features = np.concatenate(features, axis=0)  # 30x34
    # get 15 frame
    features = features[:, ::2, :, :]
    features[:, :, :, :2] *= 640
    labels = np.concatenate(labels, axis=0).argmax(1)
    pbar_test = tqdm(zip(features, labels), desc=f'Evaluate', unit=f'/{len(features)}')
    truth = []
    pred = []
    for batch_vid, labels in pbar_test:
        out, score = action_model.predict(batch_vid, (640, 640))
        truth.append(labels)
        pred.append(action_model.class_names.index(out[0]))
    class_names = action_model.class_names
    CM = metrics.confusion_matrix(truth, pred).T
    precision = metrics.precision_score(truth, pred, average=None)
    recall = metrics.recall_score(truth, pred, average=None)
    accuracy = metrics.accuracy_score(truth, pred, normalize=True)
    f1_score = metrics.f1_score(truth, pred, average=None)
    print("Accuracy: ", round(accuracy, 2) * 100)
    for i in range(len(class_names)):
        print('****Precision-Recall-F1-Score of class {}****'.format(class_names[i]))
        print('Precision: ', precision[i])
        print('Recall: ', recall[i])
        print('F1-score', f1_score[i])
    with open('info_lstm.txt', 'w') as file:
        file.write('{} {} {}'.format(precision, recall, f1_score))
    plot_cm(CM, normalize=False, save_dir='', names_x=class_names,
            names_y=class_names, show=False)
    print('Finishing!.')

