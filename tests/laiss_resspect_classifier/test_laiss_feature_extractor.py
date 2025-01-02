from laiss_resspect_classifier.laiss_feature_extractor import LaissFeatureExtractor

def test_laiss_feature_extractor():
    """Basic attribute check"""

    feature_extractor = LaissFeatureExtractor()
    assert feature_extractor.id_column == "ztf_object_id"
    assert feature_extractor.label_column == "ideal_label"
    assert feature_extractor.other_feature_names == []

    assert hasattr(feature_extractor, "fit")
    assert hasattr(feature_extractor, "fit_all")
