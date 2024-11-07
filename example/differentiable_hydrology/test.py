import sys
import os

# os.chdir('../generic_diffModel/example/differentiable_hydrology/')
sys.path.append('../../') # Add the root directory of deltaMod
# from core.utils import load_config
from example import load_config 
from hydroDL2.models.hbv.hbv import HBVMulTDET as hbv
from deltaMod.models.neural_networks import init_nn_model
from deltaMod.core.data.dataset_loading import get_dataset_dict # Eventually a hydroData import
from deltaMod.models.differentiable_model import DeltaModel as dHBV



CONFIG_PATH = '../conf/dhbv_config.yaml'

# Load configuration dictionary of model parameters and options
config = load_config(CONFIG_PATH)

#

# Setup a dataset dictionary of NN and physics model inputs.
dataset = get_dataset_dict(config, train=True)
# dataset_sample = 

# Initialize physical model and neural network
phy_model = hbv(config['dpl_model'])
nn = init_nn_model(phy_model, config['dpl_model'])

# Create the differentiable model dHBV: 
# a torch.nn.Module that describes how nn is linked to the physical model.
dpl_model = dHBV(phy_model, nn)

# Now dpl_model can be run or trained as any torch.nn.Module model in a standard training loop.

# For example, to forward:
output = dpl_model.forward(dataset)
