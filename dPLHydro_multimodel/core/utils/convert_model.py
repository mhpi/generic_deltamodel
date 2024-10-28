"""
Use saved model state dict to load states into model with different architecture.
"""
import torch
from models.differentiable_model import dPLHydroModel


save_path = '/home/lglonzarich/projects/hydro_multimodel/dPLHydro_multimodel/runs/conus_3200_merit/debugging/train_1980_1995/3_forcing/no_ensemble/LSTM_E50_R365_B100_H64_n4_111111/HBV_waterLoss_/NseLossBatchFlow_/dynamic_para/parBETA_parBETAET_/'
state_path = save_path + 'model_states_50.pt'
state_dict = torch.load(state_path)


new_state_dict = {}

new_state_dict['NN_model.linearIn.weight'] = state_dict['lstminv1.linearIn.weight']
new_state_dict['NN_model.linearIn.bias'] = state_dict['lstminv1.linearIn.bias']
new_state_dict['NN_model.lstm.w_ih'] = state_dict['lstminv1.lstm.w_ih']
new_state_dict['NN_model.lstm.w_hh'] = state_dict['lstminv1.lstm.w_hh']
new_state_dict['NN_model.lstm.b_ih'] = state_dict['lstminv1.lstm.b_ih']
new_state_dict['NN_model.lstm.b_hh'] = state_dict['lstminv1.lstm.b_hh']
new_state_dict['NN_model.linearOut.weight'] = state_dict['lstminv1.linearOut.weight']
new_state_dict['NN_model.linearOut.bias'] = state_dict['lstminv1.linearOut.bias']

new_state_dict['ANN_model.i2h.weight'] = state_dict['Ann.i2h.weight']
new_state_dict['ANN_model.i2h.bias'] = state_dict['Ann.i2h.bias']
new_state_dict['ANN_model.h2h1.weight'] = state_dict['Ann.h2h1.weight']
new_state_dict['ANN_model.h2h1.bias'] = state_dict['Ann.h2h1.bias']

new_state_dict['ANN_model.h2h2.weight'] = state_dict['Ann.h2h2.weight']
new_state_dict['ANN_model.h2h2.bias'] = state_dict['Ann.h2h2.bias']
new_state_dict['ANN_model.h2h3.weight'] = state_dict['Ann.h2h3.weight']
new_state_dict['ANN_model.h2h3.bias'] = state_dict['Ann.h2h3.bias']

new_state_dict['ANN_model.h2h4.weight'] = state_dict['Ann.h2h4.weight']
new_state_dict['ANN_model.h2h4.bias'] = state_dict['Ann.h2h4.bias']
new_state_dict['ANN_model.h2h5.weight'] = state_dict['Ann.h2h5.weight']
new_state_dict['ANN_model.h2h5.bias'] = state_dict['Ann.h2h5.bias']

new_state_dict['ANN_model.h2h6.weight'] = state_dict['Ann.h2h6.weight']
new_state_dict['ANN_model.h2h6.bias'] = state_dict['Ann.h2h6.bias']
new_state_dict['ANN_model.h2o.weight'] = state_dict['Ann.h2o.weight']
new_state_dict['ANN_model.h2o.bias'] = state_dict['Ann.h2o.bias']


model = dPLHydroModel(self.config, 'HBV_waterLoss').to(self.config['device'])
model.load_state_dict(new_state_dict)

torch.save(new_state_dict, save_path + 'HBV_waterLoss_states_Ep50.pt')
torch.save(model, save_path + 'HBV_waterLoss_model_Ep50.pt')
