from ..distance_compute import config
config.setConfig(config.speciesType.human, config.chainType.alpha)
print(config.getColumns())