
# The name of the experiment.
# All results from this experiment are cached in the same DB collection,
# so it is important to use a unique Experiment name when you change
# the input.
# TODO: Perhaps the experiment name should be derived by hashing the data-set?
EXPERIMENT_NAME = "experiment"

CACHESIZE = 400 # For ec2 small
JOBS_PER_FILE = 5
##KFOLD = True
#KFOLD = False

# Classification models
# TODO: Add GBC
#MODELS_TO_USE = ["SVC", "RandomForestClassifier"]
MODELS_TO_USE = ["RandomForestClassifier"]
## Regression models
#MODELS_TO_USE = ["SVR", "GradientBoostedRegressor", "RandomForestRegressor"]
