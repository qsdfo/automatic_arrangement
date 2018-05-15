from LOP_database.midi.write_midi import write_midi
import LOP.Scripts.config as config
import avoid_tracks
import os
import glob
import re
import LOP.Database.build_data_aux as build_data_aux
import LOP.Database.build_data_aux_no_piano as build_data_aux_no_piano


def build_split_matrices(folder_paths, quantization, temporal_granularity):
	file_counter = 0
	train_only_files={}
	train_and_valid_files={}

	for folder_path in folder_paths:
		#############
		# Read file
		folder_path = folder_path.rstrip()
		print(folder_path)
		if not os.path.isdir(folder_path):
			continue

		# Is there an original piano score or do we have to create it ?
		num_music_file = max(len(glob.glob(folder_path + '/*.mid')), len(glob.glob(folder_path + '/*.xml')))
		if num_music_file == 2:
			is_piano = True
		elif num_music_file == 1:
			is_piano = False
		else:
			raise Exception("CAVAVAVAMAVAL")

		# Get pr, warped and duration
		if is_piano:
			new_pr_piano, _, _, _, new_name_piano, _, _, _, _, _, duration\
				= build_data_aux.process_folder(folder_path, quantization, temporal_granularity, gapopen=3, gapextend=1)
		else:
			new_pr_piano, _, _, _, new_name_piano, _, _, _, _, _, duration\
				= build_data_aux_no_piano.process_folder_NP(folder_path, quantization, temporal_granularity)

		if new_pr_piano is None:
			print("FAIL !")
			continue

		split_name = re.split("/", new_name_piano)
		folder_name = "Piano_files_for_embeddings/" + split_name[-3]
		file_name = split_name[-1] + '.mid'
		if not os.path.isdir(folder_name):
			os.makedirs(folder_name)
		write_midi(new_pr_piano, 1000,  folder_name + '/' + file_name, tempo=80)
	return

if __name__ == '__main__':
	temporal_granularity='event_level'
	quantization=8
	pretraining_bool=True
	# Database have to be built jointly so that the ranges match
	DATABASE_PATH = config.database_root()
	DATABASE_PATH_PRETRAINING = config.database_pretraining_root()
	
	DATABASE_NAMES = [
		# DATABASE_PATH + "/bouliane", 
		# DATABASE_PATH + "/hand_picked_Spotify", 
		# DATABASE_PATH + "/liszt_classical_archives", 
		# DATABASE_PATH + "/imslp"
		# DATABASE_PATH_PRETRAINING + "/OpenMusicScores",
		# DATABASE_PATH_PRETRAINING + "/Kunstderfuge", 
		# DATABASE_PATH_PRETRAINING + "/Musicalion", 
		# DATABASE_PATH_PRETRAINING + "/Mutopia"
	]
	
	DATABASE_NAMES_PRETRAINING = [
		DATABASE_PATH_PRETRAINING + "/OpenMusicScores",
		DATABASE_PATH_PRETRAINING + "/Kunstderfuge", 
		DATABASE_PATH_PRETRAINING + "/Musicalion", 
		DATABASE_PATH_PRETRAINING + "/Mutopia"
	]

	# Create a list of paths
	def build_filepaths_list(path):
		folder_paths = []
		for file_name in os.listdir(path):
			if file_name != '.DS_Store':
				this_path = os.path.join(path, file_name)
				folder_paths.append(this_path)
		return folder_paths

	folder_paths = []
	for path in DATABASE_NAMES:
		folder_paths += build_filepaths_list(path)

	folder_paths_pretraining = []
	if pretraining_bool:
		for path in DATABASE_NAMES_PRETRAINING:
			folder_paths_pretraining += build_filepaths_list(path)

	# Remove garbage tracks
	avoid_tracks_list = avoid_tracks.avoid_tracks()
	folder_paths = [e for e in folder_paths if e not in avoid_tracks_list]
	folder_paths_pretraining = [e for e in folder_paths_pretraining if e not in avoid_tracks_list]
	build_split_matrices(folder_paths + folder_paths_pretraining, quantization, temporal_granularity)