def import_training_strategy(name):
	if name=="full_A":
		from LOP.Scripts.training_strategies.A.full_A import TS_full_A as Training_strategy
	elif name=="trA_teB":
		from LOP.Scripts.training_strategies.AB.trA_teB import TS_trA_teB as Training_strategy
	elif name=="trAB_teA":
		from LOP.Scripts.training_strategies.AB.trAB_teA import TS_trAB_teA as Training_strategy
	elif name=="trB__A_teA":
		from LOP.Scripts.training_strategies.AB.trB__A_teA import TS_trB__A_teA as Training_strategy
	elif name=="trC__B__A_teA":
		from LOP.Scripts.training_strategies.ABC.trC__B__A_teA import TS_trC__B__A_teA as Training_strategy
	return Training_strategy