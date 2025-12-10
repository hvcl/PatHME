import os, sys, glob
import numpy as np
import torch, csv
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
#import wandbs
from sklearn.neighbors import KNeighborsClassifier
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
#from torchmetrics import MulticlassAccuracy
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.classification import MulticlassAUROC
#from statsmodels.stats.inter_rater import ac1

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from collections import Counter
#from models.model_hierarchical_mil import HIPT_LGP_FC

def find_label2(pat_list, ref_label):
    #print (ref_label)
    sam, labels = [], []
    for sample in pat_list:
        #print (sample)
        if len(str(sample)) != 13: 
            try:
                sample_ = sample.split('/')
                sample_name = os.path.split(sample)[1].split('-S')[0]
            except:
                sample_name = sample.split('-S')[0]#sample.split('/')[6].split('-S')[0]
            if len(str(sample_name)) != 13:
                sample_name = sample_name[:13] 
        else:
            sample_name = sample
        label = ref_label[ref_label['id']== sample_name]['label'].to_numpy()
        #print (os.path.exists (f"{main_dir}{bag_folder}/{sample}.npy"))
        #if os.path.exists (f"{main_dir}{bag_folder}/{sample}.npy") == True: #os.path.exists(sample) == True:
        sam.append(sample)
        labels.append(int(label))
    return sam, labels





def get_pat_list (pat_list, ref_label, bag_dir):
    feat_path, feat_label = [], []
    #print (pat_list[:10])
    #print (ref_label)
    pat_list_ = pat_list#np.unique([x.split('-S')[0] for x in pat_list])
    for pat in pat_list_:
        #print (pat)
        if pat[:4] == 'TCGA':# or 'tcga':
            #print ('TCGA')
            bag_path = glob.glob(f"{bag_dir}{pat}*/level0*")
            #bag_path = glob.glob(f"{bag_dir}{pat}*.npy")
        else: 
            bag_path = glob.glob(f"{bag_dir}{pat}*/level0*")
        #print (bag_path)
        label = ref_label[ref_label['id'] == pat]['label'].to_numpy()
        #print (pat, label, len(bag_path))
        # if len (bag_path) >1:
        #     print (pat, len(bag_path), label)
        for _ in range (len(bag_path)):
            if label != None :
                feat_label.append (label)
                #feat_path.append (pat)
        feat_path.append (bag_path)
    #print (np.concatenate(feat_path).shape, np.concatenate(feat_label).shape)
    return np.concatenate(feat_path), np.concatenate(feat_label)#feat_path, np.concatenate(feat_label)#



class Dataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        feature = []
        for path in self.data_paths:
            if path[0] == '/':
                path = path
            else:
                path = path[0]
            feature.append (path)
        self.datas = feature
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.data_paths)
    def __getitem__(self, idx):
        path = self.datas[idx]
        if path[-4:] == '.npy':
            data = np.load(path,allow_pickle=True)
            if len(data.shape) == 1:
                data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)
        else:
            data = torch.load(path)
        label = self.labels[idx]
        if len(data.shape) == 3:
            data = data[:,0,:]
        data = data.reshape(-1, args.feat_dim)
        #data = data.reshape(-1, 1536)
        # Apply transformations if needed
        if self.transform:
            data = self.transform(data)
        # Convert to PyTorch tensor
        if path[-4:] == '.npy':
            data = torch.from_numpy(data)
        #print (data.shape, label)
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
        #print ('before: ', x.shape)
        x, weight = self.abmil(x)  
        #print ('after abmil: ', x.shape)
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



def balanced_accuracy2(y_true, y_pred):
    # Convert predictions to binary labels
    #y_pred = torch.round(torch.sigmoid(y_pred))
    
    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP = (y_true * y_pred).sum().to(torch.float32)
    TN = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    FP = ((1 - y_true) * y_pred).sum().to(torch.float32)
    FN = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    # Calculate Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    # Calculate Balanced Accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    return balanced_accuracy


def train(classifier, num_epochs, train_loader,  val_loader, run_name, label_dir, optimizer, criterion):
    
    best = 0.0  # Initialize best validation loss
    os.makedirs(f'{label_dir}/ckpt/', exist_ok=True)
    best_model_path = f'{label_dir}/ckpt/{run_name}.pth'
    print ('Start Training')
    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            print (run_name)
        running_loss = 0.0
        tcorrect = 0
        ttotal = 0
        # Training loop
        
        classifier.train()
        for inputs, targets, fn in train_loader:  # Assuming you have a train_loader for your data
            inputs = inputs.to(torch.float32).cuda()
            #print (f'inputs: {inputs.shape}')
            targets = targets.cuda()
            
            optimizer.zero_grad()
            outputs = classifier(inputs)
            #print (f"target: {targets}, outputs: {outputs}")
            
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
                #print (f"target: {targets}, outputs: {outputs}")
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # Print average training and validation loss for each epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # wandb.log({
        #     "Train Loss": avg_train_loss, "Train Accuracy": train_accuracy,
        #     "Val Loss": avg_val_loss, "Validation Accuracy": val_accuracy
        # })
            
        if val_accuracy > best:
            print (run_name)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% --> Saving best model !!!")
            #print ("Previous best: ", best)
            best = val_accuracy
            #print ('Updated best: ', best)
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
        
        print ('dataset: ' , args.dataset)
        if len(args.dataset.split('/')) == 2:
            dataset = args.dataset.split('/')[0]
        else: 
            dataset = args.dataset

        label_dir = f'{args.main_dir}/{dataset}'
        if args.dataset == 'tcga_brca':
            ref_dir = f"{label_dir}/{dataset}_{args.task}.csv"
        else:
            ref_dir = f"{label_dir}/{dataset}_{args.task}.csv"
        label = pd.read_csv(ref_dir)
        ref_label = label.loc[:, ['P_ID', args.task]]
        print (ref_label)
        ref_label.columns =['id','label']

        #tr_te_split = pd.read_csv(f'{main_dir}/split/splits{args.splits}.csv')

        #tr_te_split.columns =['split','folder_name']
        tr_te_split =  label.loc[:, ['P_ID', f'fold_{args.splits}']]
        tr_te_split.columns =['folder_name', 'split']
        val_patient = list(tr_te_split[tr_te_split['split']=='val']['folder_name'].to_numpy())
        train_patient = list(tr_te_split[tr_te_split['split']=='train']['folder_name'].to_numpy())
        test_patient = list(tr_te_split[tr_te_split['split']=='test']['folder_name'].to_numpy())
        patient_list = bag_list# glob.glob(f'{main_dir}*')
        patient_list = [os.path.split(x)[1].split('-.npy')[0] for x in patient_list]
        #print ('pat#: ',len(patient_list), patient_list[:5])
        #print ('ori pat#: ',len(train_patient), train_patient[:5])
        #print ('training patient list: ', len(train_patient), len(val_patient), len(test_patient))

        # new_train_patient,new_val_patient, new_test_patient = [], [], []
        # for patient in patient_list:
        #     #print (pat_)
        #     patient_ = patient[:12]
        #     if patient_ in train_patient:
        #         new_train_patient.append (patient_)
        #     elif patient_ in val_patient:
        #         new_val_patient.append (patient_)
        #     elif patient_ in test_patient:
        #         new_test_patient.append (patient_)

        # print ('new train pat: ', len(new_train_patient))
        # print ('new val pat: ', len(new_val_patient))
        # print ('new test pat: ', len(new_test_patient))

        #exit()
        X_train, y_train = get_pat_list (train_patient, ref_label, bag_dir)
        X_test, y_test = get_pat_list (test_patient, ref_label, bag_dir)
        X_val, y_val = get_pat_list (val_patient, ref_label, bag_dir)
        print (X_train[:5], y_train[:5])
        print ('Training data: ' ,len(X_train), len(y_train))
        print ('Testing data: ',len(X_test), len(y_test))
        print ('Validation data: ',len(X_val), len(y_val))
        print ('Before class balancing training: ', Counter(list(y_train)))
        print ('Before class balancing testing: ', Counter(list(y_test)))

        over_sampler = RandomOverSampler(random_state=42)
        X_train, y_train = over_sampler.fit_resample(np.reshape(X_train, (-1,1)), y_train)
    
        print(f"After class balancing: {Counter(y_train)}")
        print (args.bag_folder)
        print ("Preparing Data for training")

        
        # Define your input size and number of classes
        uni_label =  np.unique(ref_label['label'])
        num_classes = len(uni_label)
        print ("Preparing model for training")
        # Create an instance of the classifier
        


        if args.arc == 'abmil':
            classifier = ABMIL_classifier(input_size=args.feat_dim, hidden_size=512, num_classes=num_classes)
            if args.L0_only:
                train_data = Dataset_L0(X_train, y_train)
                val_data = Dataset_L0(X_val, y_val)
            else:
                train_data = Dataset(X_train, y_train)
                val_data = Dataset(X_val, y_val)
        else: 
            classifier = SimpleClassifier2(input_size=args.feat_dim, hidden_size=4096, num_classes=num_classes, pat_num = 100)
            train_data = Dataset_pad(X_train, y_train, 100)
            val_data = Dataset_pad(X_val, y_val, 100)
        
        
            

        # Creating data loaders for each set
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        
        # classifier = torch.load(path)
        # #classifier.load_state_dict(state_dict)
        classifier.cuda()
        

        # Define your optimizer and loss function
        if args.opt == 'sgd':
            optimizer = optim.SGD(classifier.parameters(), lr= args.lr, weight_decay=args.dr)    
        elif args.opt == 'adam':

            optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.dr)
        elif args.opt == 'adamw':
            optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.dr)
            
        criterion = nn.CrossEntropyLoss()


        run_name = '{}_{}_ep{}_bsz{}_lr{}_dc{}_{}_abmil_split{}_{}_seed{}'.\
                format(args.dataset, args.bag_folder, args.num_epochs, args.batch_size, args.lr, args.dr, args.opt, args.splits, args.task, args.seed)

            
        
        #best_model_path = f'{label_dir}/ckpt/vit_small_5organs_clsToken_49999_new_s1_agg_patfeat2_ep500_bsz1_lr0.001_dc1e-05_sgd_abmil_split{args.splits}_{args.task}.pth'
        best_model_path = train(classifier, args.num_epochs, train_loader,  val_loader, run_name, label_dir, optimizer, criterion)
        #best_model_path = '/home/Paris/jingwei/tcga_brca/ckpt/vit_small_brca_s1_provgiga_s1_patchToken_fold1_clsToken_24999_new_s1_agg_patfeat_ep500_bsz1_lr0.01_dc0.001_sgd_abmil_split1_subtype.pth'    
        if args.L0_only:
            test_data = Dataset_L0(X_train, y_train)
        else:
            test_data = Dataset(X_test, y_test)
        model = torch.load(best_model_path)
        model.eval() 
        correct = 0
        total = 0
        test_loader = DataLoader(test_data, batch_size=1)
        gt, raw, pred, bag_path = [], [], [], []
        #test_loader = val_loader
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
        #print (raw[:10])
        report = classification_report(gt,pred,digits=4, output_dict=True)
        print ('******* Slide-based Report ********')
        print ( classification_report(gt,pred,digits=4))
        df = pd.concat([pd.DataFrame(bag_path2), pd.DataFrame(gt), pd.DataFrame(pred)], axis=1)
        df.columns = ['id', 'gt', 'pred']
        df['group_id'] = df['id'].str[:17]
        max_pred = df.groupby('group_id')['pred'].max()
        max_gt = df.groupby('group_id')['gt'].max()
        final = pd.merge(max_pred, max_gt, on='group_id', suffixes=('_pred', '_gt'))
        final = final.reset_index()
        #report = classification_report(final['gt'].to_numpy(),final['pred'].to_numpy(),digits=4)
        #print ('******* Patient-based Report ********')
        #print(report)
        balanced_acc = balanced_accuracy_score(gt, pred)
        print ('Balanced Acc: ', balanced_acc)
        auroc = MulticlassAUROC(num_classes=num_classes)

        result = auroc(torch.tensor(raw), torch.tensor(gt))
        print(f'Multiclass AUROC: {result}')
        print (run_name, args.splits)
        

        save_dir = f'{args.main_dir}/{args.dataset}/'
        #os.makedirs(save_dir, exist_ok=True)
        
        def get_unique_filepath(save_dir, dataset, run_name, suffix='_roc.csv'):
            """
            Returns a unique file path by appending _1, _2, etc. if the file exists.
            """
            os.makedirs(os.path.join(save_dir, dataset), exist_ok=True)
            
            base_path = os.path.join(save_dir, dataset, f'{run_name}{suffix}')
            if not os.path.exists(base_path):
                return base_path
            
            # If file exists, add a number
            i = 1
            while True:
                new_path = os.path.join(save_dir, dataset, f'{run_name}_{i}{suffix}')
                if not os.path.exists(new_path):
                    return new_path
                i += 1

        file_path = get_unique_filepath(save_dir, args.dataset, run_name)

    
        df = pd.DataFrame(raw, columns=[f'class_{i}' for i in range(num_classes)])
        df['gt'] = gt
        
        df.to_csv(file_path, index=False)
        print(f'Saved {run_name} to {file_path}')

        
        
        # Flatten main scores (like precision/recall/f1 for each class) into one row
        row = {
            'split': f"{args.splits}",
            'Accuracy': f"{test_accuracy:.4f}", 
            'Balanced Accuracy': f"{balanced_acc:.4f}",
            'AUROC': f"{result:.4f}",
            'filename': f"{file_path}"
        }
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    row[f"{label}_{metric_name}"] = f"{value:.4f}"

        result_dir = f"{label_dir}/result/"
        os.makedirs(result_dir, exist_ok=True)
        csv_file = f"{result_dir}/{run_name.split('_seed')[0]}_eval_results.csv"
        fieldnames = list(row.keys())

        # Create directory if needed
        os.makedirs(os.path.dirname(csv_file), exist_ok=True) if os.path.dirname(csv_file) else None

        # Write or append to CSV
        file_exists = os.path.exists(csv_file)

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or f.tell() == 0:  # Write header if new file or empty
                writer.writeheader()
            writer.writerow(row)

        #kappa_score = cohen_kappa_score(gt, pred)
        #print("Cohen's Kappa Score:", kappa_score)

        #quadratic_weighted_kappa = cohen_kappa_score(gt, pred, weights='quadratic')
        #print("Quadratic Weighted Kappa Score:", quadratic_weighted_kappa)

        #weighted_kappa = cohen_kappa_score(gt, pred, weights='linear')
        #print("Weighted Kappa Score:", weighted_kappa)
        #pred_ = pred.reshape(-1, 1)
        #auc = roc_auc_score(gt, raw, multi_class='ovr')
        #print("AUC:", auc)



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
    parser.add_argument('--L0_only', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    ### Model setting
    parser.add_argument('--arc', type=str, default='abmil', help='Select Model')
    parser.add_argument('--num_epochs', type=int, default=200, help='Epoch number.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')

    ### Loss function and optimizer setting.
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--dr', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--opt', type=str, default='sgd', help='optimizer.')

    
    import random
    args = parser.parse_args()
    random.seed(42)
    seed =  42#random.sample(range(0, 10000), 5) ##[4506,3657]# # [1679]#

    for split in range (1,6):

        args.splits = split
        print(f"Running with split: {args.splits}")
        
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        main(args)
