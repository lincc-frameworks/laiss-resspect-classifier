import numpy as np
import importlib_resources
import scipy.interpolate as scinterp
import pandas as pd
from astropy.table import Table, MaskedColumn
import light_curve as lc

# helper functions as in https://github.com/alexandergagliano/laiss_timedomainanomalies
MAGN_EXTRACTOR = lc.Extractor(
    lc.Amplitude(),
    lc.AndersonDarlingNormal(),
    lc.BeyondNStd(1.0),
    lc.BeyondNStd(2.0),
    lc.Cusum(),
    lc.EtaE(),
    lc.InterPercentileRange(0.02),
    lc.InterPercentileRange(0.1),
    lc.InterPercentileRange(0.25),
    lc.Kurtosis(),
    lc.LinearFit(),
    lc.LinearTrend(),
    lc.MagnitudePercentageRatio(0.4, 0.05),
    lc.MagnitudePercentageRatio(0.2, 0.05),
    lc.MaximumSlope(),
    lc.Mean(),
    lc.MedianAbsoluteDeviation(),
    lc.PercentAmplitude(),
    lc.PercentDifferenceMagnitudePercentile(0.05),
    lc.PercentDifferenceMagnitudePercentile(0.1),
    lc.MedianBufferRangePercentage(0.1),
    lc.MedianBufferRangePercentage(0.2),
    lc.Periodogram(
        peaks=5,
        resolution=10.0,
        max_freq_factor=2.0,
        nyquist='average',
        fast=True,
        features=(
            lc.Amplitude(),
            lc.BeyondNStd(2.0),
            lc.BeyondNStd(3.0),
            lc.StandardDeviation(),
        ),
    ),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StandardDeviation(),
    lc.StetsonK(),
    lc.WeightedMean(),
)

FLUX_EXTRACTOR = lc.Extractor(
    lc.AndersonDarlingNormal(),
    lc.Cusum(),
    lc.EtaE(),
    lc.ExcessVariance(),
    lc.Kurtosis(),
    lc.MeanVariance(),
    lc.ReducedChi2(),
    lc.Skew(),
    lc.StetsonK(),
)

def create_base_features_class(
        magn_extractor,
        flux_extractor,
        bands=('R', 'g',),
    ):
    feature_names = ([f'{name}_magn' for name in magn_extractor.names]
                     + [f'{name}_flux' for name in flux_extractor.names])

    property_names = {band: [f'feature_{name}_{band}'.lower()
                             for name in feature_names]
                      for band in bands}

    features_count = len(feature_names)

    return feature_names, property_names, features_count

def remove_simultaneous_alerts(table):
    """Remove alert duplicates"""
    dt = np.diff(table['mjd'], append=np.inf)
    return table[dt != 0]

def get_detections(photometry, band):
    """Extract clean light curve in given band from locus photometry"""
    # band_lc = photometry[(photometry['ant_passband'] == band) & (~photometry['ant_mag'].isna())]
    # idx = ~MaskedColumn(band_lc['ant_mag']).mask
    band_lc = photometry[(photometry['band'] == band) & (~photometry['mag'].isna())]
    idx = ~MaskedColumn(band_lc['mag']).mask
    detections = remove_simultaneous_alerts(band_lc[idx])
    return detections

def calc_7DCD(host_df):
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

