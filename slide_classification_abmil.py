import os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split,Dataset, DataLoader
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from sklearn.neighbors import KNeighborsClassifier
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.classification import MulticlassAUROC

from collections import Counter



def get_pat_list (pat_list, ref_label, bag_dir):
    feat_path, feat_label = [], []
    pat_list_ = pat_list
    for pat in pat_list_:
        if pat[:4] == 'TCGA':
            bag_path = glob.glob(f"{bag_dir}{pat}*/*")
        else: 
            bag_path = glob.glob(f"{bag_dir}{pat}*")
        label = ref_label[ref_label['id'] == pat]['label'].to_numpy()
        for _ in range (len(bag_path)):
            if label != None :
                feat_label.append (label)
        feat_path.append (bag_path)
    return np.concatenate(feat_path), np.concatenate(feat_label)



class Dataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.data_paths)
    def __getitem__(self, idx):
        if self.data_paths[idx][0] == '/':
            path = self.data_paths[idx]
        else:
            path = self.data_paths[idx][0]
        label = self.labels[idx]
        if path.split('.')[1] =='npy':
            data = np.load(path,allow_pickle=True)
            data = torch.from_numpy(data)
        else:
           data = torch.load(path) 
        data = data.reshape(-1, args.feat_dim)
        return data, label, path
    


class ABMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ABMIL, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        H = self.tanh(self.fc1(x))  # Hidden representations
        A = self.softmax(self.fc2(H).squeeze(-1))  # Attention scores
        M = torch.sum(A.unsqueeze(-1) * x, dim=1)  # Weighted sum of instances

        return M, A




class ABMIL_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ABMIL_classifier, self).__init__()
        self.abmil = ABMIL (input_size, input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)  
    def forward(self, x):
        x, weight = self.abmil(x)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x




def train(classifier, num_epochs, train_loader,  val_loader, run_name, label_dir, optimizer, criterion):
    
    best = 1.0 
    os.makedirs(f'{label_dir}/ckpt/', exist_ok=True)
    best_model_path = f'{label_dir}/ckpt/{run_name}.pth'
    print ('Start Training')
    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            print (run_name)
        running_loss = 0.0
        tcorrect = 0
        ttotal = 0

        
        classifier.train()
        for inputs, targets, fn in train_loader:  
            inputs = inputs.to(torch.float32).cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            outputs = classifier(inputs)
            
            loss = criterion(outputs, targets).cuda()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
            _, predicted = torch.max(outputs.data, 1)
            ttotal += targets.size(0)
            tcorrect += (predicted == targets).sum().item()

        train_accuracy = 100 * tcorrect / ttotal

        
        # Validation loop
        classifier.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, fn in val_loader:
                inputs = inputs.to(torch.float32).cuda()
                targets = targets.cuda()

                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # Print average training and validation loss for each epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        
        
        if avg_val_loss < best:
            print (run_name)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% --> Saving best model !!!")
            best = avg_val_loss
            torch.save(classifier, best_model_path)
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print('Finished Training')
    return best_model_path


def main(args):
    main_dir = f'{args.main_dir}{args.dataset}'
    
    bag_dir = f"{main_dir}/{args.bag_folder}/"
    bag_list = glob.glob(f"{bag_dir}*")
    print (f"{bag_dir} --> # bag: {len(bag_list)}")
    
    if len(args.dataset.split('/')) == 2:
        dataset = args.dataset.split('/')[0]
    else: 
        dataset = args.dataset

    label_dir =  f'{args.main_dir}/{dataset}'
    
         
    ref_dir = f"{label_dir}/{dataset}_{args.task}.csv"
    label = pd.read_csv(ref_dir)
    ref_label = label.loc[:, ['P_ID', args.task]]
    
    ref_label.columns =['id','label']

    tr_te_split =  label.loc[:, ['P_ID', f'fold_{args.splits}']]
    tr_te_split.columns =['folder_name', 'split']
    val_patient = list(tr_te_split[tr_te_split['split']=='val']['folder_name'].to_numpy())[:10]
    train_patient = list(tr_te_split[tr_te_split['split']=='train']['folder_name'].to_numpy())[:30]
    test_patient = list(tr_te_split[tr_te_split['split']=='test']['folder_name'].to_numpy())[:10]
    patient_list = bag_list
    patient_list = [os.path.split(x)[1].split('-.npy')[0] for x in patient_list]


    new_train_patient,new_val_patient, new_test_patient = [], [], []
    for patient in patient_list:
        #print (pat_)
        patient_ = patient[:12]
        if patient_ in train_patient:
            new_train_patient.append (patient_)
        elif patient_ in val_patient:
            new_val_patient.append (patient_)
        elif patient_ in test_patient:
            new_test_patient.append (patient_)

   

    X_train, y_train = get_pat_list (train_patient, ref_label, bag_dir)
    X_test, y_test = get_pat_list (test_patient, ref_label, bag_dir)
    X_val, y_val = get_pat_list (val_patient, ref_label, bag_dir)

    print ('Training data: ' ,len(X_train), len(y_train))
    print ('Testing data: ',len(X_test), len(y_test))
    print ('Validation data: ',len(X_val), len(y_val))

    over_sampler = RandomOverSampler(random_state=42)
    X_train, y_train = over_sampler.fit_resample(np.reshape(X_train, (-1,1)), y_train)
   
    uni_label =  np.unique(ref_label['label'])
    num_classes = len(uni_label)

    

    classifier = ABMIL_classifier(input_size=args.feat_dim, hidden_size=512, num_classes=num_classes)
    train_data = Dataset(X_train, y_train)
    val_data = Dataset(X_val, y_val)
  
    
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    classifier.cuda()
    


    if args.opt == 'sgd':
        optimizer = optim.SGD(classifier.parameters(), lr= args.lr, weight_decay=args.dr)    
    elif args.opt == 'adam':

        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.dr)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.dr)
        
    criterion = nn.CrossEntropyLoss()


    run_name = '{}_ep{}_bsz{}_lr{}_dc{}_{}_abmil_split{}_{}'.\
            format(args.bag_folder.split('/')[-1], args.num_epochs, args.batch_size, args.lr, args.dr, args.opt, args.splits, args.task)
    
    print (run_name)
    
        
    

    best_model_path = train(classifier, args.num_epochs, train_loader,  val_loader, run_name, label_dir, optimizer, criterion)

    test_data = Dataset(X_test, y_test)
    model = torch.load(best_model_path)
    model.eval() 
    correct = 0
    total = 0
    test_loader = DataLoader(test_data, batch_size=1)
    gt, raw, pred, bag_path = [], [], [], []
    for inputs, targets, fn in test_loader:
        inputs = inputs.to(torch.float32).cuda()
        targets = targets.cuda()
        gt.append(targets.detach().cpu().numpy())
        bag_path.append(fn)
        outputs = model(inputs)
        raw.append(outputs.data.detach().cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        pred.append(predicted.detach().cpu().numpy())
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    test_accuracy = 100 * correct / total
    print (best_model_path)
    print ('Testing Accuracy: ', test_accuracy)
    bag_path2 = [os.path.split(x)[1].split('.npy')[0] for x in np.concatenate(bag_path)]
    gt = np.concatenate(gt)
    pred = np.concatenate(pred)
    raw = np.concatenate(raw)

    report = classification_report(gt,pred,digits=4)
    print ('******* Slide-based Report ********')
    print (report)
    
    balanced_acc = balanced_accuracy_score(gt, pred)
    print ('Balanced Acc: ', balanced_acc)
    auroc = MulticlassAUROC(num_classes=num_classes)

    result = auroc(torch.tensor(raw), torch.tensor(gt))
    print(f'Multiclass AUROC: {result}')
    print (run_name)
    



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Args')

    ### Data Setting
    parser.add_argument('--main_dir', type=str, default='/home/Daejeon/jingwei/')
    parser.add_argument('--dataset', type=str, default='tcga_brca')
    parser.add_argument('--bag_folder', type=str, default='donov1_kd_simloss_1e-5_feat_s1_agg_patfeat')
    parser.add_argument('--splits', type=int, default=1, help='Split number from cross-validation sets.')
    parser.add_argument('--feat_dim', type=int, default=512, help='Fetaure dimension per bag.')
    parser.add_argument('--task', type=str, default='subtype')

    ### Model setting
    parser.add_argument('--arc', type=str, default='abmil', help='Select Model')
    parser.add_argument('--num_epochs', type=int, default=100, help='Epoch number.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')

    ### Loss function and optimizer setting.
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--dr', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--opt', type=str, default='sgd', help='optimizer.')
    parser.add_argument('--full', action="store_true")



    args = parser.parse_args()
    main(args)
