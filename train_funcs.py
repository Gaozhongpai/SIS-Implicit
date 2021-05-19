import os
import torch
from tqdm import tqdm
import numpy as np

def train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                 device, model, optim, loss_fn, 
                                 bsize, start_epoch, n_epochs, eval_freq, scheduler=None,
                                 writer=None, save_recons=True, shapedata=None,
                                 metadata_dir=None, samples_dir=None, checkpoint_path=None, 
                                 io=None):
    # if not shapedata.normalization:
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)
    
    total_steps = start_epoch*len(dataloader_train)
    for epoch in range(start_epoch, n_epochs):
        model.train()
          
        tloss = []
        tloss_lap = []
        for b, tx in enumerate(tqdm(dataloader_train)):

            optim.zero_grad()
            coords, bcoords, trilist, first_idx, index_sub = dataloader_train.dataset.random_submesh()

            verts_init = []
            for name in dataloader_train.dataset.name:
                verts_init.append(tx[:, index_sub[name]])
            verts_init = torch.cat(verts_init, dim=1)

            tx, verts_init, coords, bcoords, trilist, first_idx = \
                tx.to(device), verts_init.to(device), coords.to(device), \
                bcoords.to(device), trilist.to(device), first_idx.to(device)

            cur_bsize = tx.shape[0]
            tx_hat = model(verts_init, coords, bcoords, trilist, first_idx)

            # mesh_ind = [0, 1]
            # msh = tx_hat[mesh_ind,:tx_hat.shape[1],:].detach().cpu().numpy()
            # shapedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch)),
            #                                      msh, mesh_ind)

            tx = tx * shapedata_std + shapedata_mean
            loss_l1 = loss_fn(tx, tx_hat)       
            loss_lap = dataloader_train.dataset.lap(tx, tx_hat)/20
            loss = loss_l1 + loss_lap    

            loss.backward()
            optim.step()
            
            tloss.append(cur_bsize * loss_l1.item())
            tloss_lap.append(cur_bsize * loss_lap.item())
            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss',loss_l1.item(),total_steps)
                writer.add_scalar('loss/loss/data_loss_lap',loss_lap.item(),total_steps)
                writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'],total_steps)
            total_steps += 1

        # validate
        model.eval()
        vloss = []
        vloss_lap = []
        with torch.no_grad():
            for b, tx in enumerate(tqdm(dataloader_val)):

                coords, bcoords, trilist, first_idx, index = dataloader_val.dataset.random_submesh()

                verts_init = []
                for name in dataloader_val.dataset.name:
                    verts_init.append(tx[:, index[name]])
                verts_init = torch.cat(verts_init, dim=1)

                tx, verts_init, coords, bcoords, trilist, first_idx = \
                    tx.to(device), verts_init.to(device), coords.to(device), \
                    bcoords.to(device), trilist.to(device), first_idx.to(device)
                cur_bsize = tx.shape[0]

                tx_hat = model(verts_init, coords, bcoords, trilist, first_idx)    
                
                tx = tx * shapedata_std + shapedata_mean           
                loss_l1 = loss_fn(tx, tx_hat)       
                loss_lap = dataloader_val.dataset.lap(tx, tx_hat)/20
                loss = loss_l1 + loss_lap    
                
                vloss.append(cur_bsize * loss_l1.item())
                vloss_lap.append(cur_bsize * loss_lap.item())

        # if scheduler:
        #    scheduler.step()
            
        epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        epoch_tloss_lap = sum(tloss_lap) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss_lap',epoch_tloss_lap,epoch)
        if len(dataloader_val.dataset) > 0:
            epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
            epoch_vloss_lap = sum(vloss_lap) / float(len(dataloader_val.dataset))
            writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
            writer.add_scalar('avg_epoch_valid_loss_lap', epoch_vloss_lap,epoch)
            io.cprint('epoch {0} | tr {1} tr_lap {2} | val {3} val_lap {4} | lr {5}'.format(epoch,
                epoch_tloss,epoch_tloss_lap,epoch_vloss,epoch_vloss_lap,optim.param_groups[1]['lr']))
        else:
            io.cprint('epoch {0} | tr {1} '.format(epoch,epoch_tloss))

        shape_dict = model.module.state_dict()
        shape_dict = {k: v for k, v in shape_dict.items() if 'D.' not in k and 'U.' not in k}
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': shape_dict,  #model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': shape_dict,  #model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        #model = model.to(device)
        if save_recons:
            with torch.no_grad():
                if epoch == 0:
                    mesh_ind = [0]
                    msh = tx[mesh_ind[0]:1,:tx_hat.shape[1],:].detach().cpu().numpy()
                    shapedata.save_meshes(os.path.join(samples_dir,'input_epoch_{0}'.format(epoch)),
                                                     msh, mesh_ind, False)
                mesh_ind = [0,1]
                msh = tx_hat[mesh_ind[0]:1,:tx_hat.shape[1],:].detach().cpu().numpy()
                shapedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch)),
                                                 msh, mesh_ind, False)

    print('~FIN~')


