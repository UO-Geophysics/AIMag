
import sys
sys.path.append('/home/sdybing/neic-mlaapde')

from mlaapde.access import MLAAPDE_Access
from mlaapde import UTC
import os
import h5py
import numpy as np

def main():

	log('script starting')

	log('building mlpa')
	mlpa = MLAAPDE_Access(data_dir = '/data/hank/mlaapde_v1b/data', random_seed = 616)

	log('defining variables')
	train_nsamp = False
	valid_nsamp = False
	valid_phases = ['P', 'Pn', 'Pg']
	return_labels = ['source_magnitude', 'source_magnitude_type', 'snr_db', 'phase_id']
	trim_pre_sec = 60
	trim_post_sec = 60
	min_snr_db = False
	max_snr_db = False
	cast_dtype = np.float32
	decimate_factor = 2
	log_progress_fraction = 100

	kwargs = {'valid_phases':valid_phases, 'labels':return_labels, 
				'trim_pre_sec':trim_pre_sec, 'trim_post_sec':trim_post_sec, 
				'min_snr_db':min_snr_db, 'max_snr_db':max_snr_db,
				'cast_dtype':cast_dtype, 'decimate_factor':decimate_factor,
				'log_progress_fraction':log_progress_fraction}

	pt0 = UTC('2013-8-1') # Start of training window
	pt1 = UTC('2018-10-1') # End of training window/start of validation window
	pt2 = UTC('2020-1-1') # End of validation window
	#pt0 = UTC('2013-8-1') # Start of training window
	#pt1 = UTC('2013-10-1') # End of training window/start of validation window
	#pt2 = UTC('2014-1-1') # End of validation window


	##### TRAINING LOAD #####
	log('loading training data from mlaapde...')
	train_samps, train_cat = mlpa.sample_catalog(time1 = pt0, time2 = pt1, nsamp = train_nsamp, split = [1,0], **kwargs)
	train_pids = train_samps['training']
	train_waves = mlpa.get_waves(train_pids, **kwargs)
	train_labels = mlpa.get_labels(train_pids, train_cat, labels=return_labels)

	log('writing training data to hdf5...')
	hdf5_save_dir = '/data/sdybing/allwaveforms/actuallydecimated/'
	hdf5_save_fname = hdf5_save_dir + 'training_data.hdf5'
	if os.path.isdir(hdf5_save_dir):
	    pass
	else:
	    os.makedirs(hdf5_save_dir)
	with h5py.File(hdf5_save_fname, 'w') as h5file:
		log('  saving waves')
		h5file.create_dataset('waves', (train_waves.shape), data=train_waves, chunks=True)
		log('  saving mags')
		h5file.create_dataset('magnitude', (len(train_pids),), data=train_labels['source_magnitude'], chunks=True)
		log('  saving mag types')
		h5file.create_dataset('magnitude_type', (len(train_pids),), data=train_labels['source_magnitude_type'], chunks=True)
		log('  saving snr_db')
		h5file.create_dataset('snr_db', (len(train_pids),), data=train_labels['snr_db'], chunks=True)
		log('  saving phase_ids')
		h5file.create_dataset('phase_id', (len(train_pids),), data=train_pids, chunks=True)
	log(f'  hdf5 saved {hdf5_save_fname}')

	log('  clearing variables')
	del train_samps, train_cat, train_waves, train_labels

	##### VALIDATION LOAD #####
	log('loading validation data from mlaapde...')
	valid_samps, valid_cat = mlpa.sample_catalog(time1 = pt1, time2 = pt2, nsamp = valid_nsamp, split = [0,1], **kwargs)
	valid_pids = valid_samps['validation']
	valid_waves = mlpa.get_waves(valid_pids, **kwargs)
	valid_labels = mlpa.get_labels(valid_pids, valid_cat, labels=return_labels)

	log('writing validation data to hdf5...')
	hdf5_save_dir = '/data/sdybing/allwaveforms/actuallydecimated/'
	hdf5_save_fname = hdf5_save_dir + 'validation_data.hdf5'
	if os.path.isdir(hdf5_save_dir):
	    pass
	else:
	    os.makedirs(hdf5_save_dir)
	with h5py.File(hdf5_save_fname, 'w') as h5file:
		log('  saving waves')
		h5file.create_dataset('waves', (valid_waves.shape), data=valid_waves, chunks=True)
		log('  saving mags')
		h5file.create_dataset('magnitude', (len(valid_pids),), data=valid_labels['source_magnitude'], chunks=True)
		log('  saving mag types')
		h5file.create_dataset('magnitude_type', (len(valid_pids),), data=valid_labels['source_magnitude_type'], chunks=True)
		log('  saving snr_db')
		h5file.create_dataset('snr_db', (len(valid_pids),), data=valid_labels['snr_db'], chunks=True)
		log('  saving phase_ids')
		h5file.create_dataset('phase_id', (len(valid_pids),), data=valid_pids, chunks=True)
	log(f'hdf5 saved {hdf5_save_fname}')
	del valid_samps, valid_cat, valid_waves, valid_labels



def log(text):
	print(f"{UTC().isoformat().split('.')[0]}: {text}")

if __name__=='__main__':
	main()
