import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import argparse
import json
import gc
import gspread
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW
from utils import modeling, model_eval


def run_classifier():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_target", type=str, default="all_teacher")
    parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
    parser.add_argument("--col", type=str, default="Stance1", help="Stance1 or Stance2")
    parser.add_argument("--train_mode", type=str, default="unified", help="unified or adhoc")
    parser.add_argument("--model_name", type=str, default="teacher", help="teacher or student")
    parser.add_argument("--dataset_name", type=str, default="all", help="mt,semeval,am,wtwt,covid or all-dataset")
    parser.add_argument("--filename", type=str)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--theta", type=float, default=0.6, help="AKD parameter")
    args = parser.parse_args()

    sheet_num = 4  # Google sheet number
    random_seeds = [1,2,4,5,9,10]
    target_word_pair = [args.input_target]
    model_select = args.model_select
    col = args.col
    train_mode = args.train_mode
    model_name = args.model_name
    dataset_name = args.dataset_name
    file = args.filename
    lr = args.lr
    batch_size = args.batch_size
    total_epoch = args.epochs
    dropout = args.dropout
    alpha = args.alpha
    theta = args.theta
    
    # create normalization dictionary for preprocessing
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1,**data2}

    # saved name of teacher predictions
    teacher = {
                'mt':'teacher_output_multi_whole',
                'semeval':'teacher_output_semeval_whole',
                'wtwt':'teacher_output_wtwt_batch',
                'am':'teacher_output_am_batch',
                'covid':'teacher_output_covid_whole',
                'all':'teacher_output_all_batch',
    }
    target_num = {'mt': 6, 'semeval': 4, 'wtwt': 4, 'am': 8, 'covid': 1, 'all': 23}
    eval_batch = {'mt': False, 'semeval': False, 'wtwt': True, 'am': True, 'covid': False, 'all': True}

    for target_index in range(len(target_word_pair)):
        best_result, best_val = [], []
        for seed in random_seeds:    
            print("current random seed: ", seed)

            if train_mode == "unified":
                filename1 = '/home/ubuntu/'+file+'/raw_train_all_dataset_onecol.csv'
                filename2 = '/home/ubuntu/'+file+'/raw_val_all_dataset_onecol.csv'
                filename3 = '/home/ubuntu/'+file+'/raw_test_all_dataset_onecol.csv'
                
                x_train,y_train,x_train_target = pp.clean_all(filename1,col,dataset_name,normalization_dict)
                x_val,y_val,x_val_target = pp.clean_all(filename2,col,dataset_name,normalization_dict)
                x_test,y_test,x_test_target = pp.clean_all(filename3,col,dataset_name,normalization_dict)
            
            elif train_mode == "adhoc":
                filename1 = '/home/ubuntu/'+dataset_name+'/raw_train_'+file+'.csv'  # E.g., load AM dataset 
                filename2 = '/home/ubuntu/'+dataset_name+'/raw_val_'+file+'.csv'
                filename3 = '/home/ubuntu/'+dataset_name+'/raw_test_'+file+'.csv'
                x_train,y_train,x_train_target = pp.clean_all(filename1,col,dataset_name,normalization_dict)
                x_val,y_val,x_val_target = pp.clean_all(filename2,col,dataset_name,normalization_dict)
                x_test,y_test,x_test_target = pp.clean_all(filename3,col,dataset_name,normalization_dict)

            if model_name == 'student':
                y_train2 = torch.load(teacher[dataset_name]+'_seed{}.pt'.format(seed))  # load teacher predictions

            num_labels = 3  # Favor, Against and None
            # print(x_train_target[0])
            x_train_all = [x_train,y_train,x_train_target]
            x_val_all = [x_val,y_val,x_val_target]
            x_test_all = [x_test,y_test,x_test_target]

            
            # set up the random seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed) 

            # prepare for model
            x_train_all,x_val_all,x_test_all = dh.data_helper_bert(x_train_all,x_val_all,x_test_all,\
                                        target_word_pair[target_index],model_select)
            # print(x_test_all[0][0])
            if model_name == 'teacher':
                x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len, trainloader, \
                  trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name)
            else:
                x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len, trainloader, \
                  trainloader_distill = dh.data_loader(x_train_all, batch_size, model_select, 'train', model_name,\
                                                       y_train2=y_train2)
            x_val_input_ids, x_val_seg_ids, x_val_atten_masks, y_val, x_val_len, valloader = \
                                        dh.data_loader(x_val_all, batch_size, model_select, 'val',model_name)                            
            x_test_input_ids, x_test_seg_ids, x_test_atten_masks, y_test, x_test_len, testloader = \
                                        dh.data_loader(x_test_all, batch_size, model_select, 'test',model_name)

            model = modeling.stance_classifier(num_labels,model_select).cuda()

            for n,p in model.named_parameters():
                if "bert.embeddings" in n:
                    p.requires_grad = False
                    
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
                ]
            
            loss_function = nn.CrossEntropyLoss(reduction='sum')
            if model_name == 'student':
                loss_function2 = nn.KLDivLoss(reduction='sum')
            
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

            sum_loss, sum_loss2 = [], []
            val_f1_average = []
            train_preds_distill,train_cls_distill = [], []
            if train_mode == "unified":
                test_f1_average = [[] for i in range(target_num[dataset_name])]
            elif train_mode == "adhoc":
                test_f1_average = [[]]
            
            for epoch in range(0, total_epoch):
                print('Epoch:', epoch)
                train_loss, train_loss2 = [], []
                model.train()
                if model_name == 'teacher':
                    for input_ids,seg_ids,atten_masks,target,length in trainloader:
                        optimizer.zero_grad()
                        output1 = model(input_ids, seg_ids, atten_masks, length)
                        loss = loss_function(output1, target)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        train_loss.append(loss.item())
                else:
                    for input_ids,seg_ids,atten_masks,target,length,target2 in trainloader:
                        optimizer.zero_grad()
                        output1 = model(input_ids, seg_ids, atten_masks, length)
                        output2 = output1

                        # 3. proposed AKD
                        output2 = torch.empty(output1.shape).fill_(0.).cuda()
                        for ind in range(len(target2)):
                            soft = max(F.softmax(target2[ind]))
                            if soft <= theta:
                                rrand = random.uniform(2,3)  # parameter b1 and b2 in paper
                            elif soft < theta+0.2 and soft > theta:  # parameter a1 and a2 are theta and theta+0.2 here 
                                rrand = random.uniform(1,2)
                            else:
                                rrand = 1
                            target2[ind] = target2[ind]/rrand
                            output2[ind] = output1[ind]/rrand
                        target2 = F.softmax(target2)
                            
                        loss = (1-alpha)*loss_function(output1, target) + \
                               alpha*loss_function2(F.log_softmax(output2), target2)
                        loss2 = alpha*loss_function2(F.log_softmax(output2), target2)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        train_loss.append(loss.item())
                        train_loss2.append(loss2.item())
                    sum_loss2.append(sum(train_loss2)/len(x_train))  
                    print(sum_loss2[epoch])
                sum_loss.append(sum(train_loss)/len(x_train))  
                print(sum_loss[epoch])

                if model_name == 'teacher':
                    # train evaluation
                    model.eval()
                    train_preds = []
                    with torch.no_grad():
                        for input_ids,seg_ids,atten_masks,target,length in trainloader_distill:
                            output1 = model(input_ids, seg_ids, atten_masks, length)
                            train_preds.append(output1)
                        preds = torch.cat(train_preds, 0)
                        train_preds_distill.append(preds)
                        print("The size of train_preds is: ", preds.size())

                # evaluation on val set 
                model.eval()
                val_preds = []
                with torch.no_grad():
                    if not eval_batch[dataset_name]:
                        pred1 = model(x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len)
                    else:
                        for input_ids,seg_ids,atten_masks,target,length in valloader:
                            pred1 = model(input_ids, seg_ids, atten_masks, length) # unified
                            val_preds.append(pred1)
                        pred1 = torch.cat(val_preds, 0)
                    acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_val)
                    val_f1_average.append(f1_average)

                # evaluation on test set
                if train_mode == "unified":
                    x_test_len_list = dh.sep_test_set(x_test_len,dataset_name)
                    y_test_list = dh.sep_test_set(y_test,dataset_name)
                    x_test_input_ids_list = dh.sep_test_set(x_test_input_ids,dataset_name)
                    x_test_seg_ids_list = dh.sep_test_set(x_test_seg_ids,dataset_name)
                    x_test_atten_masks_list = dh.sep_test_set(x_test_atten_masks,dataset_name)
                elif train_mode == "adhoc":
                    x_test_len_list = [x_test_len]
                    y_test_list = [y_test]
                    x_test_input_ids_list, x_test_seg_ids_list, x_test_atten_masks_list = \
                                    [x_test_input_ids], [x_test_seg_ids], [x_test_atten_masks]
                
                with torch.no_grad():
                    if eval_batch[dataset_name]:
                        test_preds = []
                        for input_ids,seg_ids,atten_masks,target,length in testloader:
                            pred1 = model(input_ids, seg_ids, atten_masks, length)
                            test_preds.append(pred1)
                        pred1 = torch.cat(test_preds, 0)
                        if train_mode == "unified":
                            pred1_list = dh.sep_test_set(pred1,dataset_name)
                        else:
                            pred1_list = [pred1]
                        
                    test_preds = []
                    for ind in range(len(y_test_list)):
                        if not eval_batch[dataset_name]:
                            pred1 = model(x_test_input_ids_list[ind], x_test_seg_ids_list[ind], \
                                          x_test_atten_masks_list[ind], x_test_len_list[ind])
                        else:
                            pred1 = pred1_list[ind]
                        test_preds.append(pred1)
                        acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_test_list[ind])
                        test_f1_average[ind].append(f1_average)

            # model that performs best on the dev set is evaluated on the test set
            best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
            best_result.append([f1[best_epoch] for f1 in test_f1_average])
            
            if model_name == 'teacher':
                best_preds = train_preds_distill[best_epoch]
                torch.save(best_preds, teacher[dataset_name]+'_seed{}.pt'.format(seed))

            print("******************************************")
            print("dev results with seed {} on all epochs".format(seed))
            print(val_f1_average)
            best_val.append(val_f1_average[best_epoch])
            print("******************************************")
            print("test results with seed {} on all epochs".format(seed))
            print(test_f1_average)
            print("******************************************")
            print(max(best_result))
            print(best_result)

        # save to Google sheet
        best_result_t = np.transpose(best_result).tolist()  # results on test set
        best_result_t.append(best_val)  # results on val set
        gc = gspread.service_account(filename='/home/ubuntu/service_account_google.json')
        sh = gc.open("Stance_Aug").get_worksheet(sheet_num) 
        row_num = len(sh.get_all_values())+1
        sh.update('A{0}'.format(row_num), target_word_pair[target_index])
        sh.update('B{0}:O{1}'.format(row_num,row_num+30), best_result_t)

if __name__ == "__main__":
    run_classifier()