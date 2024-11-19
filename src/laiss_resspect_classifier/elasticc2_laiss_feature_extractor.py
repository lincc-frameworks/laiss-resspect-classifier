from laiss_resspect_classifier.laiss_feature_extractor import LaissFeatureExtractor

class Elasticc2LaissFeatureExtractor(LaissFeatureExtractor):

    other_feature_names = [
        'hostgal_snsep',
        'hostgal_ellipticity',
        'hostgal_sqradius',
        'hostgal_mag_u',
        'hostgal_mag_g',
        'hostgal_mag_r',
        'hostgal_mag_i',
        'hostgal_mag_z',
        'hostgal_mag_y',
        'hostgal_magerr_u',
        'hostgal_magerr_g',
        'hostgal_magerr_r',
        'hostgal_magerr_i',
        'hostgal_magerr_z',
        'hostgal_magerr_y',
        # 'g-r',
        # 'r-i',
        'i-z',
        # 'z-y',
        # 'g-rErr',
        # 'r-iErr',
        'i-zErr',
        # 'z-yErr',
        '7DCD',
    ]

    yet_more_feature_names = []

    id_column = "object_id"
    label_column = "sntype"
    non_anomaly_classes = ["Normal"] # i.e. "Normal", "Ia", ...

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_metadata_header(cls) -> list[str]:
        return [cls.id_column, cls.label_column, ] # Add other metadata columns here