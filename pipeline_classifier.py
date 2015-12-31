import pandas as pd


if __name__ == "__main__":
    uidScore = pd.read_csv("/home/lei/data/credit_prediction/uid_score.csv",sep='\t')
    print(uidScore.columns)
    score = uidScore['score'].values
    neg = [i for i in score if i > 0.5]
    print(len(neg))