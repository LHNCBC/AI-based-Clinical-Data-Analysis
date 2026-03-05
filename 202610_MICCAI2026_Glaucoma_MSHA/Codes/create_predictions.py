#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
import copy
import sys
from sklearn import metrics
from timm.models import create_model
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs

from timm.data import ImageDataset, create_loader, resolve_data_config, IterableImageCsv
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#import ml_metrics as metrics
#from utils import convert_regression_predictions
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
#from sklearn.metrics import sensitive_score, specificity_score, confusion_matrix

from sklearn.metrics import f1_score
from sklearn.metrics import auc


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data_path', metavar='DIR',
                    help='path to dataset')
#parser.add_argument('--set_type', '-st',metavar='DATASET SET TYPE',
#                    help='dataset split type')
parser.add_argument('--output_dir', metavar='DIR', default='./performance_results/',
                    help='path to output files')
parser.add_argument('--csv_path', '-csv', metavar='NAME', default='',
                    help='dataset type (default: Enter csv file path with images path and labels)')
parser.add_argument('--experiment_name', default='',
                    help='add extension for the file name to save')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--model_type', '-m_t', metavar='MODEL_TYPE', default='classification',
                    help='type of task (default: classification)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--has_labels', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--enable_mc', action='store_true',
                    help='use monte carlo dropout')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')

def pick10Thres(tpr, fpr, thresholds):
    pickX = []
    pickY = []
    pickThres = []
    i = 0
    while i in range(len(tpr) - 1):
        if i == 0:
           i += len(tpr)//10
           continue
        pickX.append(tpr[i])
        pickY.append(fpr[i])
        pickThres.append(thresholds[i])
        i += len(tpr)//10
    return np.asarray(pickX), np.asarray(pickY), np.asarray(pickThres)


def plot_roc_curve(y_true, y_pred, scores, dst_dir, experiment_name, n_classes):
    from sklearn.metrics import roc_curve, auc
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Normal','Covid']))
    new_scores = []
    for label, pred, score in zip(y_true, y_pred, scores):
        new_scores.append(float(score[1]))
        #print(score)
    scores = new_scores
    # compute ROC curve
    fpr, tpr, thresholds  =  roc_curve(y_true, scores, pos_label = 1, drop_intermediate = True)
    pickX, pickY, pickThres = [] , [], []
    for i,j,k in zip(fpr,tpr, thresholds):
        pickX.append(i)
        pickY.append(j)
        pickThres.append(k)
    roc_auc =auc(fpr, tpr)
    print(" ROC auc score: ",round(roc_auc,4))
    
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.001, 1.01])
    plt.ylim([-0.001, 1.01])
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.title('ROC', fontsize = 18)
    plt.legend(loc="lower right", fontsize = 18)
    ## plot the thresholds
    if len(fpr) >= 10:
        pickX, pickY, pickThres = pick10Thres(fpr, tpr, thresholds)
    pickThres = np.round(pickThres, 2)
    #dst_file_path = str(dst_dir) + '/' + '{}'.format(str(fold)) + '-ROC.png'
    for i in range(len(pickX)):
        plt.plot(pickX[i], pickY[i], 'bo', linewidth = 2)
        plt.text(pickX[i], pickY[i], pickThres[i], fontsize=12)
    dst_path = os.path.join(dst_dir,'{}-roc.png'.format(experiment_name))
    plt.savefig(dst_path)
    #plt.show()

    # compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pickX, pickY, pickThres = [] , [], []
    for i,j,k in zip(precision,recall, thresholds):
        pickX.append(i)
        pickY.append(j)
        pickThres.append(k)
    roc_auc = auc(recall, precision)
    print(" PR ROC value: ",round(roc_auc,4))
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(recall, precision, color='red', lw=lw, label='Precision recall curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.001, 1.01])
    plt.ylim([-0.001, 1.01])
    plt.xlabel('Precision', fontsize = 18)
    plt.ylabel('Recall', fontsize = 18)
    plt.title('PR Curve', fontsize = 18)
    plt.legend(loc="lower right", fontsize = 18)
    ## plot the thresholds
    pickX, pickY, pickThres = pick10Thres(recall, precision, thresholds)
    pickThres = np.round(pickThres, 2)
    for i in range(len(pickX)):
        plt.plot(pickX[i], pickY[i], 'bo', linewidth = 2)
        plt.text(pickX[i], pickY[i], pickThres[i], fontsize=12)
    #dst_file_path = str(dst_dir) + '/' + '{}'.format(str(fold)) + '-PR.png'
    dst_path = os.path.join(dst_dir,'{}-pr.png'.format(experiment_name))
    plt.savefig(dst_path)
    #plt.show()


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model

def monte_carlo_it(model,
                   input_data,
                   n_it,
                   n_classes):
    """ Activate dropout and generate n_it Monte Carlo iterations. """
    model = copy.deepcopy(model)
    input_data = copy.deepcopy(input_data)
    pred_lst = []
    estimate_lst = []
    wt = np.arange(n_classes)
    with torch.no_grad():
        # Activate dropout during inference
        #for m in model.modules():
        #    if m.__class__.__name__.startswith('Dropout'):
        #        m.train()
        #wt = torch.tensor(0,torch.range(n_classes)-1).cuda()
        for mc_it in range(n_it):
            pred = torch.nn.functional.softmax(model(input_data))
            #estimate = torch.sum(pred * wt, axis=1)
            #pred_lst.append(pred) #.detach().cpu().numpy().tolist())
            #estimate_lst.append(estimate.detach().cpu().numpy().tolist())
            pred_lst.append(torch.unsqueeze(pred, 0))
    output_mean = torch.cat(pred_lst, 0).mean(0).detach().cpu().numpy().tolist()
    estimate = np.matmul(np.array(output_mean), wt)
    y_pred = np.argmax(output_mean, axis=1)
    return pred_lst, y_pred, output_mean, estimate

def bootstrap_accuracy(predictions, actuals, num_bootstrap_samples=1000):
    n = len(predictions)
    bootstrapped_accuracies = []
    
    for _ in range(num_bootstrap_samples):
        indices = np.random.choice(range(n), n, replace=True)
        bootstrapped_preds = np.array(predictions)[indices]
        bootstrapped_actuals = np.array(actuals)[indices]
        bootstrapped_accuracy = np.mean(bootstrapped_preds == bootstrapped_actuals)
        bootstrapped_accuracies.append(round(bootstrapped_accuracy,4))
    
    return np.array(bootstrapped_accuracies)



def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    if args.model_type == 'classification':
        model = create_model(
            args.model,
            num_classes=args.num_classes,
            in_chans=3,
            pretrained=args.pretrained,
            checkpoint_path=args.checkpoint)
    elif args.model_type == 'regression':
        model = create_model(
            args.model,
            num_classes=1,
            in_chans=3,
            pretrained=args.pretrained,
            checkpoint_path=args.checkpoint)
    else:
        print(" Enter valid model type")
        exit()

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)
    print(model)
    

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    if args.csv_path:
        data = pd.read_csv(args.csv_path)
        gt_data = []
        for index, row in data.iterrows():
            imagepath_ = row['IMAGE_PATH']
            if 'LABEL' in data:
                gt_ = row['LABEL']
            else:
                gt_ = sys.maxsize
            gt_data.append((imagepath_, gt_))
        data_ = IterableImageCsv(args.csv_path)
    else:
        data_ = ImageDataset(args.data)

    loader = create_loader(
        data_,
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    if args.enable_mc: 
        model = enable_dropout(model)
    else:
        model.eval()

    if args.enable_mc:
        UNCERTAINTY_LST = ['epistemic']
        mc_preds = {unc_type: [] for unc_type in UNCERTAINTY_LST}

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    pred_scores = []
    pred_probs = []
    gt = []
    y_preds = []
    y_pred = []
    y_probs = []
    with torch.no_grad():
        for batch_idx, (input, y_true) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            gt.extend(y_true.cpu().tolist())


            if args.model_type == 'classification':
                if not args.enable_mc:
                    _, preds = torch.max(labels, 1)
                    pred_scores.extend(labels.cpu().tolist())
                    pred_probs.extend(torch.nn.functional.softmax(labels).cpu().tolist()) 
                    topk = labels.topk(k)[1]
                    y_preds.extend(preds.cpu().tolist())
                    topk_ids.append(topk.cpu().numpy())
            
                # Store multiple predictions if is_mc is activated
                if args.enable_mc:
                    mc_pred = {}
                    for unc_type in UNCERTAINTY_LST:
                        pred_lst, y_pred, output_mean, estimate = monte_carlo_it(model, input, 50, args.num_classes) 
                        #pred_ = convert_to_labels(estimates)
                    y_preds.extend(y_pred)
                    pred_probs.extend(output_mean)
                   
            #elif args.model_type == 'regression':
            #    preds = convert_regression_predictions(labels, args.num_classes)
            #    y_preds.extend(preds.cpu().tolist())

            #print(y_true.cpu().tolist(),preds.cpu().tolist())    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    if args.has_labels:
        #topk_ids = np.concatenate(topk_ids, axis=0).squeeze()
        labels_ = ['Normal','Glaucoma']
        # printing the final performance
        print(classification_report(gt, y_preds))
        print(" Accuracy Score : ", round(accuracy_score(gt,y_preds),4))
        print(" Kappa Score: ",round(cohen_kappa_score(gt, y_preds),4))
        #print(" PPV Score :",precision_score(gt, y_preds, average='macro'))
        #print(confusion_matrix(gt, y_preds,labels=np.array([0,1])))
        conf_matrix = confusion_matrix(gt, y_preds)
        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        conf_sensitivity = (TP / float(TP + FN))
        conf_specificity = (TN / float(TN + FP))
        conf_ppv = (TP/float(TP+FP))
        #print(gt)
        #print(y_preds)
        gt_ = [ labels_[x] for x in gt]
        y_preds_ = [ labels_[x] for x in y_preds]
        print(" Sensitivity: ",round(conf_sensitivity,4))
        print(" Sprecificty: ",round(conf_specificity,4))
        print(" PPV : ", round(conf_ppv,4))
        bootstrapped_accuracies = bootstrap_accuracy(y_preds, gt)
        confidence_interval = np.percentile(bootstrapped_accuracies, [2.5, 97.5])
        print(f'95% Confidence Interval for Accuracy: {confidence_interval}')
        cm = confusion_matrix(gt_, y_preds_,labels=labels_)
        print(cm)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                      display_labels=labels_)
        #disp.plot(cmap=plt.cm.Blues,)

        file_name = str(args.experiment_name) + 'conf.png'
        dst_path = os.path.join(args.output_dir, file_name)
        #print(dst_path)
        #plt.savefig(dst_path)
        #plt.show()
        n_classes = 2
        plot_roc_curve(gt, y_preds, pred_probs, args.output_dir, args.experiment_name, n_classes)


    # writing outputs to file
    dst_file = str(args.model) +  '_{}.csv'.format(args.experiment_name)
    dst_file = os.path.join(dst_file)
    print(dst_file)
    with open(os.path.join(args.output_dir, dst_file), 'w') as out_file:
        filenames = loader.dataset.parser.filenames(basename=True)
        out_file.write("FILENAME,LABEL,PREDICTION,CLASS_0,CLASS_1\n")
        #for filename, y_true, y_pred, scores in zip(filenames, gt, y_preds, pred_probs):
        #    out_file.write('{0},{1},{2},{3}\n'.format(
        #        filename, y_true, y_pred,','.join([ str(v) for v in scores])))
        for filename, label, y_pred, scores in zip(filenames, gt, y_preds, pred_probs):
            #print(filename, y_pred, scores)
            out_file.write('{0},{1},{2},{3}\n'.format(
                filename, label, y_pred,','.join([ str(v) for v in scores])))
    out_file.close()
    print(os.path.join(args.output_dir, dst_file))
    exit() 

if __name__ == '__main__':
    main()
