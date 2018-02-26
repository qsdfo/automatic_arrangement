def import_model(name):
	if name=="LSTM_plugged_base":
		from LOP.Models.Real_time.LSTM_plugged_base import LSTM_plugged_base as Model
	elif name=="Odnade_mlp":
		from LOP.Models.NADE.odnade_mlp import Odnade_mlp as Model
	else:
		raise Exception("Not a model name")
	# from LOP.Models.Real_time.Baseline.random import Random as Model
	# from LOP.Models.Real_time.Baseline.mlp import MLP as Model
	# from LOP.Models.Real_time.Baseline.mlp_K import MLP_K as Model
	# from LOP.Models.Real_time.LSTM_static_bias import LSTM_static_bias as Model
	# from LOP.Models.Future_past_piano.Conv_recurrent.conv_recurrent_embedding_0 import Conv_recurrent_embedding_0 as Model
	return Model
