import numpy as np
import importlib_resources
import scipy.interpolate as scinterp
from astropy.table import Table
import pandas as pd

from laiss_resspect_classifier.laiss_feature_extractor import LaissFeatureExtractor

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

        laiss_features = ['None'] * self.num_features

        # TOM has lightcurve extracted features
        # TODO: figure out how those features will get here?
            # current LightCurve class doesnt have a parameter for TOM features
            # need self._get_features_per_filter(self.features, self.filters) from TOM
        

        host_d = {host_feature: self.additional_info[host_feature] for host_feature in Elasticc2LaissFeatureExtractor.host_feature_names}
        host_df = pd.DataFrame(host_d, index=[0])  

        # calculate additional host features
        for f, g in zip('griz', 'rizy'):
            host_df[f'{f}-{g}'] = host_df[f'hostgal_mag_{f}'] - host_df[f'hostgal_mag_{g}']
        host_df = self._calc_7DCD(host_df)
        host_features = np.array(host_df.iloc[0])

        return laiss_features.extend(host_features)


    def _calc_7DCD(self, host_df):
        """Calculates the color distance (7DCD) of objects in df from the
        stellar locus from Tonry et al., 2012 as in implemented in astro-ghost (Gagliano et al., 2021).

        :param df: Dataframe of PS1 objects.
        :type df: Pandas DataFrame

        :return: The same dataframe as input, with new column 7DCD.
        :rtype: Pandas DataFrame
        """
        host_df.replace(999.00, np.nan)
        host_df.replace(-999.00, np.nan)

        #read the stellar locus table from PS1
        stream = importlib_resources.files(__name__).joinpath('tonry_ps1_locus.txt')
        skt = Table.read(stream, format='ascii')

        gr = scinterp.interp1d(skt['ri'], skt['gr'], kind='cubic', fill_value='extrapolate')
        iz = scinterp.interp1d(skt['ri'], skt['iz'], kind='cubic', fill_value='extrapolate')
        zy = scinterp.interp1d(skt['ri'], skt['zy'], kind='cubic', fill_value='extrapolate')
        ri = np.arange(-0.4, 2.01, 0.001)

        gr_new = gr(ri)
        iz_new = iz(ri)
        zy_new = zy(ri)

        #adding the errors in quadrature
        host_df["g-rErr"] =  np.sqrt(host_df["hostgal_magerr_g"].astype('float')**2 + host_df["hostgal_magerr_r"].astype('float')**2)
        host_df["r-iErr"] =  np.sqrt(host_df["hostgal_magerr_r"].astype('float')**2 + host_df["hostgal_magerr_i"].astype('float')**2)
        host_df["i-zErr"] =  np.sqrt(host_df["hostgal_magerr_i"].astype('float')**2 + host_df["hostgal_magerr_z"].astype('float')**2)
        host_df['z-yErr'] =  np.sqrt(host_df['hostgal_magerr_z'].astype('float')**2 + host_df['hostgal_magerr_y'].astype('float')**2)

        host_df["7DCD"] = np.nan
        host_df.reset_index(drop=True, inplace=True)
        for i in np.arange(len(host_df["i-z"])):

            temp_7DCD_1val_gr = (host_df.loc[i,"g-r"] - gr_new)**2/host_df.loc[i, "g-rErr"]
            temp_7DCD_1val_ri = (host_df.loc[i,"r-i"] - ri)**2 /host_df.loc[i, "r-iErr"]
            temp_7DCD_1val_iz = (host_df.loc[i,"i-z"] - iz_new)**2/host_df.loc[i, "i-zErr"]
            temp_7DCD_1val_zy = (host_df.loc[i,"z-y"] - zy_new)**2/host_df.loc[i, "z-yErr"]

            temp_7DCD_1val = temp_7DCD_1val_gr + temp_7DCD_1val_ri + temp_7DCD_1val_iz + temp_7DCD_1val_zy

            host_df.loc[i,"7DCD"] = np.nanmin(np.array(temp_7DCD_1val))
        return host_df
