
# The name of the experiment.
# All results from this experiment are cached in the same DB collection,
# so it is important to use a unique Experiment name when you change
# the input.
# TODO: Perhaps the experiment name should be derived by hashing the data-set?
EXPERIMENT_NAME = "experiment"

# Convert matrices to dense. This is slow, so you should only do it if your models require it.
CONVERT_TO_DENSE = False
#CONVERT_TO_DENSE = True

CACHESIZE = 400 # For ec2 small
JOBS_PER_FILE = 25
#JOBS_PER_FILE = 5
##KFOLD = True
#KFOLD = False

# Classification models
# TODO: Add GBC
#MODELS_TO_USE = ["MultinomialNB", "SGDClassifier", "BayesianRidge", "Perceptron", "SVC", "RandomForestClassifier", "KNeighborsClassifier", "LabelSpreading", "GradientBoostingClassifier"]
#MODELS_TO_USE = ["RandomForestClassifier"]
MODELS_TO_USE = ["Perceptron"]
#MODELS_TO_USE = ["GradientBoostingClassifier"]
## Regression models
#MODELS_TO_USE = ["SVR", "GradientBoostedRegressor", "RandomForestRegressor"]
