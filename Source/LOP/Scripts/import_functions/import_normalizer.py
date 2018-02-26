def import_normalizer(name):
	if name=="no_normalization":
		from LOP.Utils.Normalization.no_normalization import no_normalization as Normalizer
	elif name=="PCA":
		from LOP.Utils.Normalization.PCA import PCA as Normalizer
	elif name=="zero_mean_unit_variance":
		from LOP.Utils.Normalization.zero_mean_unit_variance import zero_mean_unit_variance as Normalizer
	else:
		raise Exception("Not a Normalizer name")
	# from LOP.Models.Real_time.Baseline.random import Random as Model
	# from LOP.Models.Real_time.Baseline.mlp import MLP as Model
	# from LOP.Models.Real_time.Baseline.mlp_K import MLP_K as Model
	# from LOP.Models.Real_time.LSTM_static_bias import LSTM_static_bias as Model
	# from LOP.Models.Future_past_piano.Conv_recurrent.conv_recurrent_embedding_0 import Conv_recurrent_embedding_0 as Model
	return Normalizer