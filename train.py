import datetime  # <1>
import torch
import numpy as np
from sklearn.metrics import accuracy_score

device = 'cuda'

iter = 0

def train(model, optimizer, optimizer_arch, loss_fn, train_loader, epoch, writer):
    model.train()  # explicit training mode
    loss_train = 0.0

    for id, input in enumerate(train_loader):
        print(f'{id} / {len(train_loader)}', end='\r')
        # LOADING THE DATA IN A BATCH
        data, target = input
        # data = data_cpu.requires_grad_()
        # MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data, target = data.to(device), target.to(device)
        data.requires_grad_()
        # FORWARD PASS
        output = model(data)
        #        loss = loss_fn(output, target.unsqueeze(1))
        loss = loss_fn(output, target[:, None])

        # BACKWARD AND OPTIMIZE FOR NETWORK PARAMETERS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # BACKWARD AND OPTIMIZE SPECIFICALLY FOR ARCHITECTURE PARAMETERS
        reward = model.sal_foward(data)
        optimizer_arch.zero_grad()
        reward.backward()
        optimizer_arch.step()

        # PREDICTIONS
        pred = np.round(output.detach().cpu().numpy())
        target = np.round(target.detach().cpu().numpy())
        loss_train += loss.item()
        
        if id % 10 == 0:
            # PRINT BATCH OUTPUTS
            train_acc = accuracy_score(target.tolist(), pred.tolist())
            train_loss = loss_train / len(train_loader)

            writer.add_scalar('Loss/train', loss.item(), len(train_loader) * (epoch - 1) + id)
            writer.add_scalar('Accuracy/train', train_acc, len(train_loader) * (epoch - 1) + id)
            print('{} TRAINING: Epoch {}, Loss {}, Accuracy {}'.format(id, epoch, loss, train_acc))

        torch.cuda.empty_cache() 
    return train_loss, train_acc


def test(model, test_loader):
    # model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []

    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for id, input in enumerate(test_loader):
            # LOAD THE DATA IN A BATCH
            data, target = input

            # moving the tensors to the configured device
            data, target = data.to(device), target.to(device)

            # Create model on data
            output = model(data)

            # PREDICTIONS
            pred = np.round(output.detach().cpu().numpy())
            target = target.float()
            y_true.extend(target.tolist())
            y_pred.extend(pred.tolist())
            val_acc = accuracy_score(y_true, y_pred)

    print("TESTING: Accuracy on test set is", val_acc)
    return val_acc
