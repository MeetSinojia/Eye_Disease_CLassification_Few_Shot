import numpy as np
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt 
from tqdm import tqdm_notebook
from tqdm import tnrange

from utilities import MEAN, STD


# create samples
def extract_sample(n_way, n_support, n_query, datax, datay):

  sample = []

  K = np.random.choice(np.unique(datay), n_way, replace=False) # returns a numpy array with a size equal to n_way

  for cls in K: # cls = data class

    datax_cls = datax[datay == cls] # get the images corresponding to our class

    perm = np.random.permutation(datax_cls) # randomly shuffle the images of this class

    sample_cls = perm[:(n_support+n_query)] #  select a sample, which is == (no. of support images + no. of query images)

    sample.append(sample_cls) # add sample images to list, we end up with [[images of cls_1],[images of cls_2],..., [images of class cls_n_way]]

  sample = np.array(sample) # convert list to numpy array

  sample = torch.from_numpy(sample) # convert to a tensor

  sample = sample.type(torch.float32) / 255.0 # convert to float and scale the image data to [0, 1]
  sample = sample.permute(4, 2, 3, 0, 1) 
  sample = (sample - MEAN[:,None, None, None, None]) / STD[:, None, None, None, None]

  sample = sample.permute(3, 4, 1, 2, 0)

 
  sample = sample.permute(0,1,4,2,3)
  
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'class_labels': K
      })


def display_sample(sample):
  sample_4D = sample.view(sample.shape[0]*sample.shape[1],*sample.shape[2:])  

  out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
  plt.figure(figsize=(16,7))


  out = out.permute(1, 2, 0) # [img_height, img_width, channels]
  out = out * STD[None, None, :] + MEAN[None, None, :] # remember you are only multiplying by the channels dimension

  plt.imshow(out)



def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, epoch, train_episode):
 
  model.train()

  running_loss = 0.0
  running_acc = 0.0
  L2_regulization = 0

  for episode in tnrange(train_episode, desc=f"Epoch {epoch+1} train: "):

    sample = extract_sample(n_way, n_support, n_query, train_x, train_y)

    optimizer.zero_grad()

    loss, output = model.set_forward_loss(sample)

    L2_regularization = torch.tensor(0., requires_grad=True)  
    for param in model.parameters():
      L2_regularization = L2_regularization + torch.norm(param, 2)


    loss = loss + (0.01 * L2_regularization)

    running_loss += output['loss']
    running_acc += output['acc']
    
    loss.backward()
    optimizer.step()

  avg_loss = running_loss / train_episode
  avg_acc = running_acc / train_episode

  return avg_loss, avg_acc

def validate(model, validation_x, validation_y, n_way, n_support, n_query, epoch, validation_episode):
  running_loss = 0.0
  running_acc = 0.0
 
  model.eval()

  with torch.no_grad(): # we are not computing any gradients, this uses less memory and speeds up computations

    for episode in tnrange(validation_episode, desc=f"Epoch {epoch+1} validation: "):
      sample = extract_sample(n_way, n_support, n_query, validation_x, validation_y)
      loss, output = model.set_forward_loss(sample)
      running_loss += output['loss']
      running_acc += output['acc']
      
    avg_loss = running_loss / validation_episode
    avg_acc = running_acc / validation_episode
  
  return avg_loss, avg_acc

def test_model_on_one_task(model, n_way, n_support, n_query, test_episodes, x_test, y_test):

  running_loss = 0.0
  running_acc = 0.0

  model.eval()

  print(f"Test loss and accuracy every 100 episodes: ")
  with torch.no_grad(): # we are not computing any gradients, this uses less memory and speeds up computations

    for episode in range(test_episodes):
      sample = extract_sample(n_way, n_support, n_query, x_test, y_test)
      loss, output = model.set_forward_loss(sample)
      running_loss += output['loss']
      running_acc += output['acc']

      if (episode % 100 == 0):
        print(f"Episode: {episode} ---> Loss: {output['loss']:.3f}, Accuracy: {output['acc']:.2f}")
      
    avg_loss = running_loss / test_episodes
    avg_acc = running_acc / test_episodes
  
  return avg_loss, avg_acc


def run_training_and_evaluation(model, 
                                train_x, 
                                train_y, 
                                validation_x, 
                                validation_y, 
                                n_way, 
                                n_support, 
                                n_query, 
                                train_episode, 
                                validation_episode,
                                optimizer,
                                max_epoch,
                                filename                                
                                ):


  best_validation_loss = float('inf')

  train_loss_list, train_accuracy_list = [], []
  validation_loss_list, validation_accuracy_list = [], []
  scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

  print(f"Start training: ")

  for epoch in range(max_epoch):  

    train_loss, train_accuracy = train(model, optimizer, train_x, train_y, n_way, n_support, n_query, epoch, train_episode)
    validation_loss, validation_accuracy = validate(model, validation_x, validation_y, n_way, n_support, n_query, epoch, validation_episode)

    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      torch.save(model.state_dict(), filename)

    print(f"\nEpoch: {epoch + 1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}%")
    print(f"\t Val. Loss: {validation_loss:.3f} | Val. Acc: {validation_accuracy*100:.2f}%")


    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
    validation_loss_list.append(validation_loss)
    validation_accuracy_list.append(validation_accuracy)

    scheduler.step()

  return train_loss_list, train_accuracy_list, validation_loss_list, validation_accuracy_list 


# Prediction on a sample of data
def predict(model, sample, device="cpu"):
  model.to(device)
  l, output = model.set_forward_loss(sample)

  return output