from sklearn.ensemble import RandomForestClassifier
from laiss_resspect_classifier.laiss_classifier import LaissRandomForest

def test_laiss_classifier():
    """Basic attribute check"""

    laiss_rf = LaissRandomForest()
    assert laiss_rf.n_estimators == 100
    assert isinstance(laiss_rf.classifier, RandomForestClassifier)

def test_classifier_with_kwargs():
    """Check that kwargs are passed to the classifier"""

    laiss_rf = LaissRandomForest(n_estimators=200, max_depth=13)
    assert laiss_rf.n_estimators == 200
    assert laiss_rf.classifier.max_depth == 13
