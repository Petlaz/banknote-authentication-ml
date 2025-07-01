from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def build_voting_classifier():
    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = GradientBoostingClassifier(random_state=42)
    clf4 = SVC(probability=True, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('rf', clf2),
            ('gb', clf3),
            ('svc', clf4)
        ],
        voting='soft'
    )
    return voting_clf
