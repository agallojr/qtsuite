"""
basic test
"""

#pylint: disable = wrong-import-position, invalid-name

import os
import sys
import pandas as pd

sys.path.append("./materials/models")
sys.path.append("./materials")
from materials.models import fm4m

os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_df  = pd.read_csv("./materials/data/bace/train.csv")
test_df   = pd.read_csv("./materials/data/bace/test.csv")

input_set = "smiles"
output = "Class"
task_name = "bace"

xtrain = list(train_df[input_set].values)
ytrain = list(train_df[output].values)

xtest = list(test_df[input_set].values)
ytest = list(test_df[output].values)

model_type = "SELFIES-TED"
x_batch, x_batch_test = fm4m.get_representation(xtrain, xtest,
    model_type = model_type, return_tensor = False)
# Replace model_list and task parameters depending on your requirement
result = fm4m.multi_modal(model_list=["SELFIES-TED","MHG-GED","SMI-TED"],
    x_train=xtrain, y_train=ytrain, x_test=xtest, y_test=ytest,
    downstream_model="DefaultClassifier")

print(result)
