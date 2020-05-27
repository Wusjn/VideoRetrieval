import pickle
import os
import re
from utils import get_longest_shot, pic_norm, get_split


if __name__ == "__main__":
    splits_dir = "./data/HMDB_splits"
    pics_dir = "./data/HMDB_main_frame"


    train_split = get_split(splits_dir, r"1")
    dev_split = get_split(splits_dir, r"2")
    search_split = get_split(splits_dir, r"2")
    corpus_split = get_split(splits_dir, r"[02]")

    train_set = {}
    train_set["xs"] = []
    train_set["ys"] = []
    train_set["path"] = []
    for label, type, name in train_split:
        base_dir = pics_dir + "/" + type + "/" + name
        pics = get_longest_shot(base_dir)
        for pic in pics:
            train_set["path"].append(pic)
            train_set["xs"].append(pic_norm(pic, (96, 72)))
            train_set["ys"].append(label)

    dev_set = {}
    dev_set["xs"] = []
    dev_set["ys"] = []
    dev_set["path"] = []
    for label, type, name in dev_split:
        base_dir = pics_dir + "/" + type + "/" + name
        pics = get_longest_shot(base_dir)
        for pic in pics:
            dev_set["path"].append(pic)
            dev_set["xs"].append(pic_norm(pic, (96, 72)))
            dev_set["ys"].append(label)

    with open("./data/HMDB_train_set.pkl", "wb") as file:
        pickle.dump(train_set, file)
    with open("./data/HMDB_dev_set.pkl", "wb") as file:
        pickle.dump(dev_set, file)
    with open("./data/HMDB_search_split.pkl", "wb") as file:
        pickle.dump(search_split, file)
    with open("./data/HMDB_corpus_split.pkl", "wb") as file:
        pickle.dump(corpus_split, file)
