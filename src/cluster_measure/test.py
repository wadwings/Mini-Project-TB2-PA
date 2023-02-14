from ..distance_compute import config
config.set_config(config.speciesType.human, config.chainType.alpha)
print(config.get_columns())