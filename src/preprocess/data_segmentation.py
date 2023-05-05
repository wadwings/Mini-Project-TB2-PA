import copy

from src.preprocess.setup import config


def label_topk_not_j(data, k, j=0):
    data = copy.deepcopy(data)
    if 'label' not in data.columns:
        raise KeyError(f"provided data don't have a 'label' column")
    labels_topk = data['label'].value_counts().head(k).tail(k - j).index.tolist()
    data = data[data['label'].isin(labels_topk)]
    return data


def method_selection():
    return label_topk_not_j


def do_segmentation(data, k=10, j=0):
    method = method_selection()
    return method(data, k, j)
