import torch.optim as optim 
import torch
import numpy as np 

def train(model,tensor_train_w,train_label,args,all_indices,device):
  

  epochs = args.epochs
  learning_rate = args.learning_rate
  beta1=0.99
  beta2 = 0.999
  kld_arr,recon_arr = [],[]
  model.to(device)

  optimizer = optim.Adam(model.parameters(), learning_rate, betas=(beta1, beta2))
  for epoch in range(epochs):

      loss_u_epoch = 0.0 ## NL loss
      loss_KLD = 0.0  ## KL loss
      loss_epoch = 0.0 ## Loss per batch #
      
      model.train()
      zx_l = []
      label_l = []
      
      for batch_ndx in all_indices:
        input_w = torch.tensor(tensor_train_w[batch_ndx]).float().to(device)
        labels = train_label[batch_ndx]
        label_l.extend(labels)
        recon_v, zx,(loss, loss_u, xkl_loss) = model(input_w,compute_loss=True)
        zx_l.extend(zx.data.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()             # backpror.step()        
        loss_epoch += loss.item()
        loss_u_epoch += loss_u.item()
        loss_KLD += xkl_loss.item()
        current_model = model 
      kld_arr.append(loss_KLD)
      recon_arr.append(loss_u_epoch)

      if epoch % 10 == 0:
          #  print('Epoch -> {}'.format(epoch))
          print('Epoch -> {} , loss -> {}'.format(epoch,loss_epoch))
          print('recon_loss==> {} || KLD==> {}'.format(loss_u_epoch, loss_KLD))
          # plot_fig(np.array(zx_l),label_l,model.decoder_phi_bn(model.centres).data.cpu().numpy(),10.0,'No')
  return current_model

def test(model,all_indices,tensor_train_w,args,keywords_as_labels,num_topic,train_label,device):
  num_coordinate = args.num_coordinate
  model.eval()
  x_list = []
  labels_list = []
  doc_ids = []
  zx_phi_list=[]
  with torch.no_grad():
      for batch_ndx in all_indices:
          input_w = torch.tensor(tensor_train_w[batch_ndx]).float().to(device)
          labels = train_label[batch_ndx] 
          labels_list.extend(labels)
          
          z, recon_v, zx,zphi, zx_phi = model(input_w,compute_loss=False)
          zx = zx.view(-1, num_coordinate).data.detach().cpu().numpy()
          zphi = zphi.data.detach().cpu().numpy()
          zx_phi = zx_phi.view(-1, num_topic).data.detach().cpu().numpy()
          zx_phi_list.extend(zx_phi)
          x_list.extend(zx)
          doc_ids = np.append(doc_ids,batch_ndx.numpy().astype(int))
     
      x_list = np.asarray(x_list)
      labels_list = np.asarray(labels_list)
      pseudo_idx = np.argwhere(np.isin(labels_list,keywords_as_labels)).ravel()
      query_center = x_list[pseudo_idx] 
    
      beta = model.decoder.weight.data.cpu().numpy().T#
      ir_query_center = torch.zeros(1,2)
      
  return x_list,labels_list,zphi,doc_ids,beta,query_center,ir_query_center