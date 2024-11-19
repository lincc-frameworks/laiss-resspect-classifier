from sklearn.ensemble import RandomForestClassifier

from resspect.classifiers import ResspectClassifier

class LaissRandomForest(ResspectClassifier):
    """LAISS-specific version of the sklearn RandomForestClassifier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # could specify default values for the RandomForestClassifier here

        self.n_estimators = self.kwargs.pop('n_estimators', 100)
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, **self.kwargs)
