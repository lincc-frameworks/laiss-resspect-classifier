import numpy as np
import importlib_resources
import scipy.interpolate as scinterp
from astropy.table import Table
import pandas as pd
from itertools import chain

from laiss_resspect_classifier.laiss_feature_extractor import LaissFeatureExtractor
from laiss_resspect_classifier.laiss_extractor_helper import *

class Elasticc2LaissFeatureExtractor(LaissFeatureExtractor):

    host_feature_names = [
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
    ]

    calculated_host_feature_name = [
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

    id_column = "object_id"
    label_column = "sntype"
    non_anomaly_classes = ["Normal"] # i.e. "Normal", "Ia", ...

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # as a child of LaissFeatureExtractor, it has feature_names
        self.filters = ['g', 'r']
        self.num_features = len(self.filters)*len(Elasticc2LaissFeatureExtractor.feature_names) + len(Elasticc2LaissFeatureExtractor.other_feature_names) + len(Elasticc2LaissFeatureExtractor.other_feature_names)
        
    @classmethod
    def get_metadata_header(cls) -> list[str]:
        return [cls.id_column, cls.label_column, ] # Add other metadata columns here


    def get_features(self) -> list[str]:
        return self._get_lc_features() + self._get_host_features()

    def _get_lc_features(self) -> list[str]:
        return self._get_features_per_filter(self.features, self.filters)

    def _get_host_features(self) -> list[str]:
        return Elasticc2LaissFeatureExtractor.host_feature_names + Elasticc2LaissFeatureExtractor.calculated_host_feature_name

    def all_fit(self) -> np.ndarray:

        # TODO: check first if TOM had lc features
            # if so - no need to extract them again

        laiss_features = ['None'] * self.num_features

        lightcurve = self.photometry
        min_obs_count = 4
        _, property_names, _ = create_base_features_class(MAGN_EXTRACTOR, FLUX_EXTRACTOR)
        
        lc_properties_d={}
        for band, names in property_names.items():
            detections = get_detections(lightcurve, band.lower())

            # check if there are enough observations
            if (len(detections) < min_obs_count):
                print(f"Not enough obs for {self.id}. pass!\n")
                return

            # extract lc features        
            t = np.array(detections['mjd'])
            m = np.array(detections['mag'], dtype=np.float64)
            merr = np.array(detections['magerr'], dtype=np.float64)
            flux = np.array(detections['flux'], dtype=np.float64)
            fluxerr = np.array(detections['fluxerr'], dtype=np.float64)

            magn_features = MAGN_EXTRACTOR(
                t,
                m,
                merr,
                fill_value=None,
            )
            flux_features = FLUX_EXTRACTOR(
                t,
                flux,
                fluxerr,
                fill_value=None,
            )

            for name, value in zip(names, chain(magn_features, flux_features)):
                lc_properties_d[name] = value
    
        laiss_features = lc_properties_d[self._get_lc_features()].values()

        # extract host features
        host_d = {host_feature: self.additional_info[host_feature] for host_feature in Elasticc2LaissFeatureExtractor.host_feature_names}
        host_df = pd.DataFrame(host_d, index=[0])  

        # calculate additional host features
        for f, g in zip('griz', 'rizy'):
            host_df[f'{f}-{g}'] = host_df[f'hostgal_mag_{f}'] - host_df[f'hostgal_mag_{g}']
        host_df = calc_7DCD(host_df)
        host_features = np.array(host_df.iloc[0])

        #TODO: check that this array has its features in the correct order for the classifier
        return laiss_features.extend(host_features)
