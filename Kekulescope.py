from __future__ import division, print_function
#args.cell_line = sys.argv[1]
#args.seed = int(sys.argv[2])
#args.task=sys.argv[3]
#net=sys.argv[4]
#lr=float(sys.argv[5])
#args.step_size_lr_decay = int(sys.argv[6])
#args.drop_factor_lr = float(sys.argv[7])
#args.batch_size = int(sys.argv[8])
#data_augmentation=int(sys.argv[9])
#args.args.nb_epochs_training_per_cycle=int(sys.argv[10])
#args.nb_epochs_training=int(sys.argv[11])
#seed = args.seed

from argparse import ArgumentParser
cell_lines = ["KB","SK-OV-3","CCRF-CEM","LoVo","HCT-15","A2780","DU-145","PC-3","A549","K562","L1210","HL-60","MDA-MB-231","PC-3","HeLa","MCF7"]
nets = ["alexnet", "vgg19_bn", "resnet152", "densenet201",  "resnet152_extended", "alexnet_extended", "densenet201_extended"]
argparser = ArgumentParser()
args_group = argparser.add_argument_group(title='Running args')
args_group.add_argument('-cell_line', type=str, help='Cancer cell line to train the model on', required=True, choices=cell_lines)
args_group.add_argument('-seed', type=int, help='Seed used to split the data into training, validation and test sets', required=False, default=1)
args_group.add_argument('-args.task', type=str, help='Set this argument to train in order to train the network, or to predict to load a pretrained model', required=False, default="train",choices=['train','predict'])
args_group.add_argument('-architecture', type=str, help='ConvNet to be used', required=False, choices=nets, default="vgg19_bn")
args_group.add_argument('-lr', type=float, help='Maximum learning rate value. Please note that optimal values vary across architectures', required=False, default=0.01)
args_group.add_argument('-step_size_lr_decay', type=int, help='Step size to decrease the learning rate by a given factor (parameter drop_factor_lr)', required=False, default=25)
args_group.add_argument('-drop_factor_lr', type=float, help='The learning rate is reduced by the factor indicated in this argument', required=False, default=0.6 )
args_group.add_argument('-args.batch_size', type=int, help='Batch size', required=False, default=16)
args_group.add_argument('-data_augmentation', type=int, help='Whether data augmentation should be applied to the validation and training sets (1: yes; 0: no)', required=False, default=1,choices=[0,1] )
args_group.add_argument('-args.nb_epochs_training_per_cycle', type=int, help='Number of epochs for each learning rate annealing cycle', required=False, default=200)
args_group.add_argument('-args.nb_epochs_training_training', type=int, help='Number of epochs to be considered for training', required=False, default=600)
args_group.add_argument('-args.nb_epochs_training_args.epochs_early_stopping', type=int, help='Number of epochs for early stopping', required=False, default=250)
args = argparser.parse_args()

#args.cell_line = args.cell_line
#args.seed = args.seed
#seed = args.seed
#args.task=args.args.task
#net=args.architecture
#lr=args.lr
#args.step_size_lr_decay = args.step_size_lr_decay
#args.drop_factor_lr = args.drop_factor_lr
#args.batch_size = args.args.batch_size
#data_augmentation=args.data_augmentation
#args.args.nb_epochs_training_per_cycle=args.args.nb_epochs_training_per_cycle
#args.nb_epochs_training=args.args.nb_epochs_training_training
#args.epochs_early_stop=args.epochs_args.epochs_early_stop
import torch
from profilehooks import profile
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, data_augmentations
import os, glob, time
import copy
import joblib, sys
import numpy as np
import scipy
from scipy import stats
from scipy import spatial
import os,sys, os.path
from collections import defaultdict
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.rdBase
from rdkit import DataStructs
from rdkit.DataStructs import BitVectToText
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
import IPython
#IPython.core.display.set_matplotlib_formats('svg')
from IPython.core.display import SVG
from torch.autograd import Variable
import multiprocessing



split_size = [0.7, 0.15, 0.15]

## if calculated, stop
outsum="results/preds_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(net,args.cell_line, args.seed,args.lr,args.drop_factor_lr,args.step_size_lr_decay,args.batch_size, args.data_augmentation)
print(outsum)
if not os.path.isfile(outsum):

    suppl = Chem.SDMolSupplier('datasets_cells/'+args.cell_line+'/'+args.cell_line+'.sdf')
    mols = [x for x in suppl if x is not None]
    my_smiles=[Chem.MolToSmiles(submol) for submol in mols]
    chembl_ids=[m.GetProp("ChEMBL_ID") for m in mols]
    #activities =[float(m.GetProp("pIC50")) for m in mols]
    activities = np.load("./datasets_cells/"+args.cell_line+"/"+args.cell_line+"_bios.npy")
    if(len(my_smiles) != len(activities)):
        raise "The number of compounds does not correspond to the number of bioactivities"
    
    train_actives=0
    train_inactives=0
    chembl_ids = np.asarray(chembl_ids)
    activities =  np.asarray(activities)
    my_smiles = np.asarray(my_smiles)
    
    base_indices = np.arange(0,len(activities))
    np.random.seed(seed)
    np.random.shuffle(base_indices)
    np.random.seed(seed)
    np.random.shuffle(base_indices)
    
    nb_test = int(len(base_indices) * split_size[2])
    nb_val = int(len(base_indices) * split_size[1])
    test_indices = base_indices[1:nb_test]
    val_indices = base_indices[(nb_test+1):(nb_test+nb_val)]
    train_indices = base_indices[(nb_test+nb_val+1):len(base_indices)]
    
    # divide training into: true training and validation
    activities_train = activities[train_indices]
    activities_test = activities[test_indices]
    activities_val = activities[val_indices]
    
    chembl_ids_train = chembl_ids[train_indices]
    chembl_ids_test = chembl_ids[test_indices]
    chembl_ids_val = chembl_ids[val_indices]
    
    mols_train = [m for i,m in enumerate(mols) if i in train_indices]
    mols_val = [m for i,m in enumerate(mols) if i in val_indices]
    mols_test = [m for i,m in enumerate(mols) if i in test_indices]
    my_smiles_train = my_smiles[train_indices]
    my_smiles_test = my_smiles[test_indices]
    my_smiles_val = my_smiles[val_indices]
    #------------
    
    os.system("mkdir -p images")
    os.system("mkdir -p ./images/{}".format(args.cell_line))
    os.system("mkdir -p ./images/{}/{}".format(args.cell_line, args.seed))
    os.system("mkdir -p ./images/{}/{}/train".format(args.cell_line, args.seed))
    os.system("mkdir -p ./images/{}/{}/test".format(args.cell_line, args.seed))
    os.system("mkdir -p ./images/{}/{}/val".format(args.cell_line, args.seed))
    os.system("mkdir -p ./images/{}/{}/train/images".format(args.cell_line, args.seed))
    os.system("mkdir -p ./images/{}/{}/test/images".format(args.cell_line, args.seed))
    os.system("mkdir -p ./images/{}/{}/val/images".format(args.cell_line, args.seed))
    
    #-----------------------------------------------
    # generate images
    #-----------------------------------------------
    svgs = glob.glob( "./images/{}/{}/train/images/*svg".format(args.cell_line, args.seed) )
    pngs = glob.glob( "./images/{}/{}/train/images/*png".format(args.cell_line, args.seed) )
    if len(svgs) == 0 and len(pngs) == 0:
        for i,mm in enumerate(my_smiles_train):
            mol_now=[Chem.MolFromSmiles(my_smiles_train[i])]
            koko=Chem.Draw.MolsToGridImage([x for x in mol_now], molsPerRow=1,useSVG=True)
            orig_stdout = sys.stdout
            f = open('./images/{}/{}/train/images/{}.svg'.format(args.cell_line,args.seed,chembl_ids_train[i]), 'w')
            sys.stdout = f
            print(koko.data)
            sys.stdout = orig_stdout
            f.close()
    else:
        print("SVGs ready")
    
    svgs = glob.glob( "./images/{}/{}/val/images/*svg".format(args.cell_line, args.seed) )
    pngs = glob.glob( "./images/{}/{}/val/images/*png".format(args.cell_line, args.seed) )
    if len(svgs) == 0 and len(pngs) == 0:
        for i,mm in enumerate(my_smiles_val):
            mol_now=[Chem.MolFromSmiles(my_smiles_val[i])]
            koko=Chem.Draw.MolsToGridImage([x for x in mol_now], molsPerRow=1,useSVG=True)
            orig_stdout = sys.stdout
            f = open('./images/{}/{}/val/images/{}.svg'.format(args.cell_line,args.seed,chembl_ids_val[i]), 'w')
            sys.stdout = f
            print(koko.data)
            sys.stdout = orig_stdout
            f.close()
    else:
        print("SVGs ready")
    
    
    svgs = glob.glob( "./images/{}/{}/test/images/*svg".format(args.cell_line, args.seed) )
    pngs = glob.glob( "./images/{}/{}/test/images/*png".format(args.cell_line, args.seed) )
    if len(svgs) == 0 and len(pngs) == 0:
        for i,mm in enumerate(my_smiles_test):
            mol_now=[Chem.MolFromSmiles(my_smiles_test[i])]
            koko=Chem.Draw.MolsToGridImage([x for x in mol_now], molsPerRow=1,useSVG=True)
            orig_stdout = sys.stdout
            f = open('./images/{}/{}/test/images/{}.svg'.format(args.cell_line,args.seed,chembl_ids_test[i]), 'w')
            sys.stdout = f
            print(koko.data)
            sys.stdout = orig_stdout
            f.close()
    else:
        print("SVGs ready")
    #-----------------------------------------------
    
    #-----------------------------------------------
    # convert images to png
    #-----------------------------------------------
    
    pngs = glob.glob( "./images/{}/{}/train/images/*png".format(args.cell_line, args.seed) )
    if len(pngs) == 0:
        basedir=os.getcwd()
        os.chdir("./images/{}/{}/train/images".format(args.cell_line, args.seed))
        cmd = "AA=($( find . -name '*.svg'  ));for i in ${AA[*]}; do convert -density 800 ${i} -resize 300x ${i}.png ; done"
        os.system(cmd)
        cmd="rm -rf *.svg"
        os.system(cmd)
        os.chdir(basedir)
    
    pngs = glob.glob( "./images/{}/{}/test/images/*png".format(args.cell_line, args.seed) )
    if len(pngs) == 0:
        basedir=os.getcwd()
        os.chdir("./images/{}/{}/test/images".format(args.cell_line, args.seed))
        cmd = "AA=($( find . -name '*.svg'  ));for i in ${AA[*]}; do convert -density 800 ${i} -resize 300x ${i}.png ; done"
        os.system(cmd)
        cmd="rm -rf *.svg"
        os.system(cmd)
        os.chdir(basedir)
    
    pngs = glob.glob( "./images/{}/{}/val/images/*png".format(args.cell_line, args.seed) )
    if len(pngs) == 0:
        basedir=os.getcwd()
        os.chdir("./images/{}/{}/val/images".format(args.cell_line, args.seed))
        cmd = "AA=($( find . -name '*.svg'  ));for i in ${AA[*]}; do convert -density 800 ${i} -resize 300x ${i}.png ; done"
        os.system(cmd)
        cmd="rm -rf *.svg"
        os.system(cmd)
        os.chdir(basedir)
    
    del svgs
    del pngs

    #-----------------------------------------------
    # data augmentation
    #-----------------------------------------------
    
    if args.data_augmentation == 1:
        transform = {
                'train': args.data_augmentations.Compose([
                    args.data_augmentations.Resize(224),
                    args.data_augmentations.RandomHorizontalFlip(),
                    args.data_augmentations.RandomVerticalFlip(),
                    args.data_augmentations.RandomRotation(degrees=90),
                    args.data_augmentations.ToTensor(),
                    args.data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                'val': args.data_augmentations.Compose([
                    args.data_augmentations.Resize(224),
                    args.data_augmentations.RandomHorizontalFlip(),
                    args.data_augmentations.RandomVerticalFlip(),
                    args.data_augmentations.RandomRotation(degrees=90),
                    args.data_augmentations.ToTensor(),
                    args.data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                'test': args.data_augmentations.Compose([
                    args.data_augmentations.Resize(224),
                    args.data_augmentations.ToTensor(),
                    args.data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                }
    else:
        transform = {
                'train': args.data_augmentations.Compose([
                    args.data_augmentations.Resize(224),
                    args.data_augmentations.ToTensor(),
                    args.data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                'val': args.data_augmentations.Compose([
                    args.data_augmentations.Resize(224),
                    args.data_augmentations.ToTensor(),
                    args.data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                'test': args.data_augmentations.Compose([
                    args.data_augmentations.Resize(224),
                    args.data_augmentations.ToTensor(),
                    args.data_augmentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                }
    
    
    #------------------------------------
    # Data loaders
    #------------------------------------
    from load_images import *
    
    data_dir="./images/{}/{}/".format(args.cell_line, args.seed)
    
    paths_labels_train=[]
    for i,x in enumerate(activities_train):
        path_now = './images/{}/{}/train/images/{}.svg.png'.format(args.cell_line,args.seed,chembl_ids_train[i])
        now = (path_now , x)
        paths_labels_train.append(now)
    
    paths_labels_val=[]
    for i,x in enumerate(activities_val):
        path_now = './images/{}/{}/val/images/{}.svg.png'.format(args.cell_line,args.seed,chembl_ids_val[i])
        now = (path_now , x)
        paths_labels_val.append(now)
    
    paths_labels_test=[]
    for i,x in enumerate(activities_test):
        path_now = './images/{}/{}/test/images/{}.svg.png'.format(args.cell_line,args.seed,chembl_ids_test[i])
        now = (path_now , x)
        paths_labels_test.append(now)


    workers=multiprocessing.cpu_count()
    shuffle=False
    
    ## use the custom functions to load the data
    trainloader = torch.utils.data.DataLoader(
                ImageFilelist(paths_labels= paths_labels_train,
                transform=transforms['train']),
                batch_size=args.batch_size, shuffle=shuffle,
                num_workers=workers) 
    
    valloader = torch.utils.data.DataLoader(
                ImageFilelist(paths_labels= paths_labels_val,
                data_augmentation=transform['val']),
                batch_size=args.batch_size, shuffle=shuffle,
                num_workers=workers) 
    
    testloader = torch.utils.data.DataLoader(
                ImageFilelist(paths_labels= paths_labels_test,
                data_augmentation=transform['test']),
                batch_size=args.batch_size, shuffle=shuffle,
                num_workers=workers) 
    
    dataloaders = {'train': trainloader, 'val':valloader, 'test':testloader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    #print(torch.cuda.device_count())
    
    #-----------------------------------------------
    # Training the model
    #-----------------------------------------------
    
    use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() and x else torch.FloatTensor)
    use_gpu()
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        start_time = time.time()
        # use the lines below to use more than 1 GPU
        #model = torch.nn.DataParallel(model, device_ids=[0, 1 , 2, 3])
        model.cuda()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1000.0
        early = 0
    
        for epoch in range(num_epochs):
            time_epoch = time.time()

            # cyclical learning rate
            if epoch % args.args.nb_epochs_training_per_cycle == 0:
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size_lr_decay, gamma=args.drop_factor_lr)

            print('Epoch {}/{} {}'.format(epoch, num_epochs - 1, early))
            print('-' * 10)
            if early >= args.epochs_early_stop:
                model.load_state_dict(best_model_wts)
                return model
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                epoch_losses=0.0
                deno=0.0
                if phase == 'train':
                    scheduler.step()
                    model.train()  
                else:
                    model.eval()
    
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    labels = labels.type(torch.FloatTensor)
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        aa = time.time()
                        outputs = model(inputs)
                        preds=outputs.squeeze(1)
                        preds = preds.type(torch.FloatTensor)
                        loss = criterion(preds, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            aa = time.time()
                            loss.backward()
                            optimizer.step()

                    del inputs; del outputs
                    epoch_losses += loss.data[0] * len(preds)
                    deno +=len(preds)
                    del preds

                epoch_loss = epoch_losses / deno
                print('{} Loss: {:.4f} {}'.format(phase, epoch_loss, deno))
                del deno

                #torch.cuda.empty_cache()
    
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early=0
                if phase == 'val' and epoch_loss > best_loss:
                    early+=1
                
                # stop if there is no convergence....
                #if phase == 'val' and best_loss > 2 and epoch >= 50:
                #    model.load_state_dict(best_model_wts)
                #    return model

                # now predict for test set
                if phase == 'val':
                    pred=[]
                    obs=[]
                    for inputs, labels in dataloaders['test']:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                        outputs = model(inputs)
                        for i in range(len(labels)):
                            pred.append(outputs.data[i])
                            obs.append(labels.data[i])
                        del labels, outputs, inputs
                    pred=np.asarray(pred)
                    obs=np.asarray(obs)
                    rms = sqrt(mean_squared_error( pred,obs)) 
                    r2=scipy.stats.pearsonr(pred,obs)
                    print('test Loss: {:.4f} {}'.format(rms, r2[0]))
                    del pred, obs, rms, r2
    
            print('Epoch complete in {:.0f}m {:.0f}s'.format( (time.time() - time_epoch) // 60, (time.time() - time_epoch) % 60))

    
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))
    
        model.load_state_dict(best_model_wts)
        return model
    
    #-----------------------------------------------
    # Architectures
    #-----------------------------------------------
    if net not in nets:
        raise "The selected architecture is not available"

    if net == "alexnet": 
        model_ft = models.alexnet(pretrained=True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 1)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr)
        model_ft = model_ft.to(device)
    
    if net == "alexnet_extended": 
        model_ft = models.alexnet(pretrained=True)
        modules=[]
        modules.append( nn.Linear(in_features=9216, out_features=4096, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=100, out_features=1, bias=True) )
        classi = nn.Sequential(*modules)
        model_ft.classifier = classi
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr)
        model_ft = model_ft.to(device)
    
    if net == "densenet201": 
        model_ft = models.densenet201(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 1)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr)#, momentum=0.95) #, nesterov=True)
        model_ft = model_ft.to(device)

    if net == "densenet201_extended": 
        model_ft = models.densenet201(pretrained=True)
        modules=[]
        modules.append( nn.Linear(in_features=1920, out_features=4096, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=100, out_features=1, bias=True) )
        classi = nn.Sequential(*modules)
        model_ft.classifier = classi
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr)
        model_ft = model_ft.to(device)

    if net == "vgg19_bn": 
        model_ft = models.vgg19_bn(pretrained=True)
        modules=[]
        modules.append( nn.Linear(in_features=25088, out_features=4096, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=100, out_features=1, bias=True) )
        classi = nn.Sequential(*modules)
        model_ft.classifier = classi
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr)#, momentum=0.95) #, nesterov=True)
        model_ft = model_ft.to(device)
    
    if net == "resnet152": 
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 1)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr)#, momentum=0.95) #, nesterov=True)
        model_ft = model_ft.to(device)

    if net == "resnet152_extended": 
        model_ft = models.resnet152(pretrained=True)
        modules=[]
        modules.append( nn.Linear(in_features=2048, out_features=4096, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=4096, out_features=1000, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=1000, out_features=200, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=200, out_features=100, bias=True) )
        modules.append( nn.ReLU(inplace=True) )
        modules.append( nn.Dropout(p=0.5) )
        modules.append( nn.Linear(in_features=100, out_features=1, bias=True) )
        classi = nn.Sequential(*modules)
        model_ft.fc = classi
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr)
        model_ft = model_ft.to(device)
    
    criterion = torch.nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size_lr_decay, gamma=args.drop_factor_lr)
    
    
    #-----------------------------------------------
    # Training
    #-----------------------------------------------
    if args.task=="train":
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=args.nb_epochs_training)
        try:
            torch.save(model_ft, 'models/model_{}_{}_{}_{}.pt'.format(net,args.cell_line, args.seed,args.lr))
        except:
            pass
    else:
        try:
            model_ft = torch.load('models/model_{}_{}_{}_{}.pt'.format(net,args.cell_line, args.seed,args.lr))
        except:
            pass
    
    #-----------------------------------------------
    # Predictions test set
    #-----------------------------------------------
    model_ft.eval()
    pred=[]
    obs=[]
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        for i in range(len(labels)):
            pred.append(outputs.data[i])
            obs.append(labels.data[i])
    pred=np.asarray(pred)
    obs=np.asarray(obs)
    rms = sqrt(mean_squared_error( pred,obs)) ##target.flatten() - y_pred.flatten() )
    print("RMSE test: {}".format(rms))
    r2=scipy.stats.pearsonr(pred,obs)
    print("Pearson R2 test: ", scipy.stats.pearsonr(pred,obs))
    
    os.system("mkdir -p results")
    ff = open("results/summary_{}_{}_{}.txt".format(net,args.cell_line, args.seed),'a')
    ff.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(rms,r2[0],args.lr,args.drop_factor_lr,args.step_size_lr_decay, args.batch_size, args.data_augmentation))
    ff.close()
    
    ff = open("results/preds_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(net,args.cell_line, args.seed, args.lr,args.drop_factor_lr,args.step_size_lr_decay,args.batch_size, args.data_augmentation),'a')
    for i,x in enumerate(pred):
        ff.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(obs[i], pred[i], args.lr,args.drop_factor_lr,args.step_size_lr_decay,args.batch_size, args.data_augmentation))
    ff.close()
