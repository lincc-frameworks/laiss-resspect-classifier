from laiss_resspect_classifier.laiss_feature_extractor import LaissFeatureExtractor

class Elasticc2LaissFeatureExtractor(LaissFeatureExtractor):

    other_feature_names = [
        #! todo
    ]

    id_column_name = None
    label_column_name = None
    non_anomaly_classes = [] # i.e. "Normal", "Ia", ...

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_metadata_header(cls) -> list[str]:
        return [cls.id_column, cls.label_column, ] # Add other metadata columns here