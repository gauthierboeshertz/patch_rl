from collections import defaultdict
from progress.bar import Bar
import einops 
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate
from hydra.utils import get_original_cwd
from src.datasets import SequenceImageTransitionDataset
from src.action_quantizer import ActionVQVAE
from stable_baselines3.common.utils import set_random_seed
set_random_seed(0)
import logging
logger = logging.getLogger(__name__)

def train_epoch(model,optimizer, dataloader,device):
    model.train()

    total_loss = 0
    epoch_loss_dict = defaultdict(float)
    for _, batch in enumerate(dataloader):
        
        optimizer.zero_grad()
        action = batch[1].to(device)
        action = einops.rearrange(action, 'b t (n a) -> (b t n) a', a=2)
        loss, loss_dict = model.compute_loss(action)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        for k in loss_dict:
            epoch_loss_dict[k] += float(loss_dict[k])
        total_loss += float(loss)
        
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= len(dataloader)
    return total_loss/len(dataloader), epoch_loss_dict

def val_epoch(model,dataloader,device):
    model.eval()
    
    epoch_loss_dict = defaultdict(float)
    total_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            action = batch[1].to(device)
            action = einops.rearrange(action, 'b t (n a) -> (b t n) a', a=2)
            loss,  loss_dict = model.compute_loss(action)
            for k in loss_dict:
                epoch_loss_dict[k] += float(loss_dict[k])
            total_loss += float(loss)
            
    for k in epoch_loss_dict:
        epoch_loss_dict[k] /= len(dataloader)

    return  total_loss/len(dataloader), epoch_loss_dict
                


def train(model, optimizer, train_dataloader, val_dataloader, num_epochs,device,scheduler):
    info_bar = Bar('Training', max=num_epochs)
    min_val_loss = 100000
    
    for epoch in range(num_epochs):
        train_loss, train_loss_dict = train_epoch(model,optimizer, train_dataloader,device)
        val_loss, val_loss_dict = val_epoch(model,val_dataloader,device)   
        
        scheduler.step()
        
        short_epoch_info = "Epoch: {},  train Loss: {}, Val Loss: {}".format(epoch,train_loss,val_loss )   
        epoch_info = f"Epoch: {epoch}, TRAIN: "
        for k in train_loss_dict:
            epoch_info += f"{k}: {train_loss_dict[k]:.5f}, "
        epoch_info += "VAL: "
        for k in val_loss_dict:
            epoch_info += f"{k}: {val_loss_dict[k]:.5f}, "
        #epoch_info = f"Epoch: {epoch},TRAIN : DYN Loss: {train_dyn_loss} VAE LOSS: {train_vae_loss}  INV LOSS: {train_inv_loss}||   VAL : DYN Loss: {val_dyn_loss} VAE LOSS: {val_vae_loss} INV LOSS: {val_inv_loss}"
        #print(epoch_info)
        logger.info(epoch_info)
        torch.save(model.state_dict(), f"last_act_quant.pt")
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"best_act_quant.pt")

        Bar.suffix = short_epoch_info
        info_bar.next()
    info_bar.finish()
    return model


@hydra.main(config_path="configs", config_name="train_action_quantizer")
def main(config):
    
    print("Training with config: {}".format(config))
    
    data_path = "{}/data/visual_{}transitions_{}_{}_{}_{}{}{}.npz".format(get_original_cwd(),config["env"]["num_transitions"],config["env"]["num_sprites"],("all_sprite_mover"if config["env"]["all_sprite_mover"] else "one_sprite_mover" if config["env"]["one_sprite_mover"] else  "discrete_all_sprite_mover" if config["env"]["discrete_all_sprite_mover"] else "select_move"),config["env"]["random_init_places"],config["env"]["num_action_repeat"],("instantmove" if config["env"]["instant_move"] else ""),("no_targets" if config["env"]["dont_show_targets"] else ""))
    dataset = SequenceImageTransitionDataset(data_path=data_path,onehot_action=False,sequence_length=3)#config["env"]["discrete_all_sprite_mover"])
        
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=config["train_loop"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=128,shuffle=False,num_workers=config["train_loop"]["num_workers"])
    
    train_dataloader = DataLoader(train_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=True,num_workers=config["train_loop"]["num_workers"])
    val_dataloader = DataLoader(val_dataset,batch_size=config["train_loop"]["batch_size"],shuffle=False,num_workers=config["train_loop"]["num_workers"])
    
    model = ActionVQVAE(2,64,0.25,0.99,)
    model.to(config["device"])
    optimizer = torch.optim.AdamW( list(model.parameters()) , lr=config["train_loop"]["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["train_loop"]["scheduler_milestones"], gamma=0.2)
    train(model, optimizer,train_dataloader,val_dataloader,config["train_loop"]["num_epochs"],config["device"],scheduler)
    print("Training Done")
    
    
if __name__ == "__main__":

    main()