import pandas as pd

def compute_auc(model, user, test, train):

    pos = set(test[test['user'] == user]['item'])

    rec = model.recommend(user)[0]
    df_test_excl = train.loc[train.user == user]
    rec_filt = filter_train_rec(rec, df_test_excl)

    n = len(rec_filt)
    p = len(pos)

    f = n - p
    tp = 0
    fp = 0

    fpr = 0

    for i, r in enumerate(rec_filt):
        if r in pos:
            tp += 1
            fpr += (fp/f)
        else:
            fp += 1

    return 1 - fpr/p

def compute_ap(model, user, test, train):

    pos = set(test[test['user'] == user]['item'])
    rec = model.recommend(user)[0]
    df_test_excl = train.loc[train.user == user]
    rec_filt = filter_train_rec(rec, df_test_excl)

    p = len(pos)
    tp = 0
    ap = 0

    for i, r in enumerate(rec_filt):
        if r in pos:
            tp += 1
            ap += tp/(i+1)
    return ap/p

def filter_train_rec(rec, user_train):
    rec_filt = pd.DataFrame({'item': rec}, )
    rec_filt = pd.merge(rec_filt, user_train, on="item", how="outer", indicator=True)
    rec_filt = rec_filt.loc[rec_filt._merge == 'left_only']['item']
    return rec_filt