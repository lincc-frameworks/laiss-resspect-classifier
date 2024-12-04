from laiss_resspect_classifier.laiss_feature_extractor import LaissFeatureExtractor

class ZtfLaissFeatureExtractor(LaissFeatureExtractor):
    other_feature_names = [
        '*momentXX',
        '*momentXY',
        '*momentYY',
        '*momentR1',
        '*momentRH',
        '*PSFFlux',
        '*ApFlux',
        '*KronFlux',
        '*KronRad',
        '*ExtNSigma',
        '*ApMag_*KronMag',
        '*ApMag_*KronMag',
        '*ApMag_*KronMag',
        '*ApMag_*KronMag',
        '*ApMag_*KronMag',
    ]

    id_column = "ztf_object_id"
    label_column = "ideal_label"
    non_anomaly_classes = ["Normal", "AGN"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_metadata_header(cls, **kwargs) -> list[str]:
        kwargs['override_primary_columns'] = [cls.id_column, "obs_num", "mjd_cutoff", cls.label_column]
        return super().get_metadata_header(**kwargs)
