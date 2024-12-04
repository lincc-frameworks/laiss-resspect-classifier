import pandas as pd
from resspect.fit_lightcurves import fit
from resspect.tom_client import TomClient
import os
import numpy as np
from laiss_resspect_classifier.elasticc2_laiss_feature_extractor import Elasticc2LaissFeatureExtractor

additional_info = [
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

url = os.environ.get('TOM_URL', "https://desc-tom-2.lbl.gov")

username = os.environ.get('TOM_USERNAME', 'USER')
passwordfile = os.environ.get('TOM_PASSWORDFILE', 'FILEPATH')

def get_phot(obj_df):

    tom = TomClient(url = url, username = username, passwordfile = passwordfile)

    # get all of the photometry at once
    ids = obj_df['diaobject_id'].tolist()
    res = tom.post('db/runsqlquery/',
                          json={ 'query': 'SELECT diaobject_id, filtername, midpointtai, psflux, psfluxerr'  
                                ' FROM elasticc2_ppdbdiaforcedsource' 
                              ' WHERE diaobject_id IN (%s) ORDER BY diaobject_id, filtername, midpointtai;' % (', '.join(str(id) for id in ids)),
                                'subdict': {} } )
    all_phot = res.json()['rows']
    all_phot_df = pd.DataFrame(all_phot)
    # if you need mag from the arbitrary flux-
    all_phot_df['mag'] = -2.5*np.log10(all_phot_df['psflux']) + 27.5
    all_phot_df['magerr'] = 2.5/np.log(10) * all_phot_df['psfluxerr']/all_phot_df['psflux']

    #! Need to send Rob a message to ask that these features be included when querying for hot super nova
    host_res = tom.post('db/runsqlquery/',
                          json={ 'query': 'SELECT diaobject_id, hostgal_mag_u, hostgal_mag_g, hostgal_mag_r, hostgal_mag_i, hostgal_mag_z, hostgal_mag_Y, hostgal_magerr_u, hostgal_magerr_g, hostgal_magerr_r, hostgal_magerr_i, hostgal_magerr_z, hostgal_magerr_Y, hostgal_snsep, hostgal_ellipticity, hostgal_sqradius'
                                ' FROM elasticc2_ppdbdiaobject'
                              ' WHERE diaobject_id IN (%s) ORDER BY diaobject_id;' % (', '.join(str(id) for id in ids)),
                                'subdict': {} } )
    all_host = host_res.json()['rows']


    # format into a list of dicts
    data = []
    for idx, obj in obj_df.iterrows():
        phot = all_phot_df[all_phot_df['diaobject_id'] == obj['diaobject_id']]

        phot_d = {}
        phot_d['objectid'] = int(obj['diaobject_id'])
        phot_d['sncode'] = int(obj['gentype'])
        phot_d['redshift'] = obj['zcmb']
        phot_d['ra'] = obj['ra']
        phot_d['dec'] = obj['dec']
        phot_d['photometry'] = phot[['filtername', 'midpointtai', 'psflux', 'psfluxerr', 'mag', 'magerr']].to_dict(orient='list')

        phot_d['photometry']['band'] = phot_d['photometry']['filtername']
        phot_d['photometry']['mjd'] = phot_d['photometry']['midpointtai']
        phot_d['photometry']['flux'] = phot_d['photometry']['psflux']
        phot_d['photometry']['fluxerr'] = phot_d['photometry']['psfluxerr']
        phot_d['photometry']['mag'] = phot_d['photometry']['mag']
        phot_d['photometry']['magerr'] = phot_d['photometry']['magerr']
        del phot_d['photometry']['filtername']
        del phot_d['photometry']['midpointtai']
        del phot_d['photometry']['psflux']
        del phot_d['photometry']['psfluxerr']
        phot_d = {**phot_d, **all_host[idx]}
        del phot_d['diaobject_id']
        data.append(phot_d)

    return data

def validate_objects(objects_to_test):
    fe = Elasticc2LaissFeatureExtractor()
    good_objs = []

    for t_obj in objects_to_test:
        fe.photometry= pd.DataFrame(t_obj['photometry'])
        fe.id = t_obj['objectid']

        fe.additional_info = {}
        for info in additional_info:
            fe.additional_info[info] = t_obj[info]

        res = fe.fit_all()
        if res:
            good_objs.append(t_obj)

    return good_objs

def main():


    #MAKE INITIAL TRAINING SET 
    objs = []

    tom = TomClient(url = url, username = username, passwordfile = passwordfile)

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61300 and peakmjd<61309 and gentype=10 limit 20;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61300 and peakmjd<61309 and gentype=21 limit 10;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61300 and peakmjd<61309 and gentype=31 limit 10;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    training_objs = get_phot(pd.DataFrame(objs))

    good_objs = validate_objects(training_objs)

    outdir = 'TOM_days_storage'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fit(
        good_objs,
        output_features_file = outdir+'/TOM_training_features',
        feature_extractor = 'laiss_resspect_classifier.elasticc2_laiss_feature_extractor.Elasticc2LaissFeatureExtractor',
        filters='ZTF',
        additional_info=additional_info
    )
    data = pd.read_csv('TOM_days_storage/TOM_training_features',index_col=False)
    data['orig_sample'] = 'train'
    data["type"] = np.where(data["sncode"] == 10, 'Ia', 'other')
    data.to_csv('TOM_days_storage/TOM_training_features',index=False)

    #MAKE TEST SET 
    objs = []

    tom = TomClient(url = url, username = username, passwordfile = passwordfile)

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61310 and peakmjd<61339 and gentype=10 limit 1000;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61310 and peakmjd<61339 and gentype=21 limit 500;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61310 and peakmjd<61339 and gentype=31 limit 500;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    test_objs = get_phot(pd.DataFrame(objs))
    outdir = 'TOM_days_storage'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    good_test_objs = validate_objects(test_objs)

    fit(
        good_test_objs,
        output_features_file = outdir+'/TOM_test_features',
        feature_extractor = 'laiss_resspect_classifier.elasticc2_laiss_feature_extractor.Elasticc2LaissFeatureExtractor',
        filters='ZTF',
        additional_info=additional_info
    )
    data = pd.read_csv('TOM_days_storage/TOM_test_features',index_col=False)
    data['orig_sample'] = 'test'
    data["type"] = np.where(data["sncode"] == 10, 'Ia', 'other')
    data.to_csv('TOM_days_storage/TOM_test_features',index=False)

    #MAKE VALIDATION SET 
    objs = []

    tom = TomClient(url = url, username = username, passwordfile = passwordfile)

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61340 and gentype=10 limit 1000;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61340 and gentype=21 limit 500;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    res = tom.post('db/runsqlquery/',
                            json={ 'query': 'SELECT diaobject_id, gentype, zcmb, peakmjd,' 
                                ' peakmag_g, ra, dec FROM elasticc2_diaobjecttruth WHERE peakmjd>61340 and gentype=31 limit 500;', 
                                'subdict': {}} )
    objs.extend(res.json()['rows'])

    val_objs = get_phot(pd.DataFrame(objs))
    outdir = 'TOM_days_storage'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    good_val_objs = validate_objects(val_objs)

    fit(
        good_val_objs,
        output_features_file = outdir+'/TOM_validation_features',
        feature_extractor = 'laiss_resspect_classifier.elasticc2_laiss_feature_extractor.Elasticc2LaissFeatureExtractor',
        filters='ZTF',
        additional_info=additional_info
    )
    data = pd.read_csv('TOM_days_storage/TOM_validation_features',index_col=False)
    data['orig_sample'] = 'validation'
    data["type"] = np.where(data["sncode"] == 10, 'Ia', 'other')
    data.to_csv('TOM_days_storage/TOM_validation_features',index=False)



if __name__ == '__main__':
    main()