import itertools
import numpy as np

from resspect.feature_extractors.light_curve import LightCurve

class LaissFeatureExtractor(LightCurve):
    # these features were previously used with ZTF bands [r,g].
    #! But they don't have to be presumably???
    feature_names = [
        'feature_amplitude_magn_*',
        'feature_anderson_darling_normal_magn_*',
        'feature_beyond_1_std_magn_*',
        'feature_beyond_2_std_magn_*',
        'feature_cusum_magn_*',
        'feature_inter_percentile_range_2_magn_*',
        'feature_inter_percentile_range_10_magn_*',
        'feature_inter_percentile_range_25_magn_*',
        'feature_kurtosis_magn_*',
        'feature_linear_fit_slope_magn_*',
        'feature_linear_fit_slope_sigma_magn_*',
        'feature_magnitude_percentage_ratio_40_5_magn_*',
        'feature_magnitude_percentage_ratio_20_5_magn_*',
        'feature_mean_magn_*',
        'feature_median_absolute_deviation_magn_*',
        'feature_percent_amplitude_magn_*',
        'feature_median_buffer_range_percentage_10_magn_*',
        'feature_median_buffer_range_percentage_20_magn_*',
        'feature_percent_difference_magnitude_percentile_5_magn_*',
        'feature_percent_difference_magnitude_percentile_10_magn_*',
        'feature_skew_magn_*',
        'feature_standard_deviation_magn_*',
        'feature_stetson_k_magn_*',
        'feature_weighted_mean_magn_*',
        'feature_anderson_darling_normal_flux_*',
        'feature_cusum_flux_*',
        'feature_excess_variance_flux_*',
        'feature_kurtosis_flux_*',
        'feature_mean_variance_flux_*',
        'feature_skew_flux_*',
        'feature_stetson_k_flux_*'
    ]

    # these are galaxy features are used with the LSST bands [u,g,r,i,z,y]
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
        '*ApMag_*KronMag', #! Confirm with Haille that this is correct
        '*ApMag_*KronMag', #! Confirm with Haille that this is correct
        '*ApMag_*KronMag', #! Confirm with Haille that this is correct
        '*ApMag_*KronMag', #! Confirm with Haille that this is correct
        '*ApMag_*KronMag', #! Confirm with Haille that this is correct
    ]

    # these features don't seem to be related to bands
    yet_more_feature_names = [
        'i-z',
        '7DCD',
        'dist/DLR'
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_metadata_header(cls) -> list[str]:
        return ["ztf_object_id"]

    @classmethod
    def get_features(cls, filters: list) -> list[str]:
        """Produce the full list of feature names for the given filters.

        Parameters
        ----------
        filters : list[str]
            List of filters to use in the feature names. i.e. ['r', 'g']

        Returns
        -------
        list[str]
            The complete list of feature names for the given filters, plus the
            other features that don't depend on the filter.
        """
        features = []
        features.extend(cls._get_features_per_filter(cls.feature_names, filters))

        LSST_filters = ['g', 'r', 'i', 'z', 'y']
        features.extend(cls._get_features_per_filter(cls.other_feature_names, LSST_filters))

        features.extend(cls.yet_more_feature_names)

        return features

    def fit(self, band:str = None) -> np.ndarray:
        pass

    def fit_all(self) -> np.ndarray:
        pass

    @classmethod
    def _get_features_per_filter(cls, features: list, filters: list) -> list[str]:
        """Simple function to get all possible combinations of features and filters.
        Will replace the '*' in the feature name with the filter name.

        i.e. features = ['example_*'], filters = ['r', 'g']. Returns ['example_r', 'example_g']

        Parameters
        ----------
        features : list[str]
            List of features where each '*' in the name will be replaced with the filter name.
        filters : list[str]
            List of filters to replace the '*' in the feature names.

        Returns
        -------
        list[str]
            List of features with the '*' replaced by the filter name.
        """

        return [pair[0].replace('*', pair[1]) for pair in itertools.product(features, filters)]