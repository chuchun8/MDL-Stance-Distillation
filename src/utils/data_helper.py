import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer
    

# Tokenization
def convert_data_to_ids(tokenizer, target, text):
    
    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []
    for tar, sent in zip(target, text):
        encoded_dict = tokenizer.encode_plus(
                            ' '.join(tar),
                            ' '.join(sent),             # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            truncation = True,
                       )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))
    
    return input_ids, seg_ids, attention_masks, sent_len


# BERT/BERTweet tokenizer    
def data_helper_bert(x_train_all,x_val_all,x_test_all,main_task_name,model_select):
    
    print('Loading data')
    
    x_train,y_train,x_train_target = x_train_all[0],x_train_all[1],x_train_all[2]
    x_val,y_val,x_val_target = x_val_all[0],x_val_all[1],x_val_all[2]
    x_test,y_test,x_test_target = x_test_all[0],x_test_all[1],x_test_all[2]
    print("Length of original x_train: %d"%(len(x_train)))
    print("Length of original x_val: %d, the sum is: %d"%(len(x_val), sum(y_val)))
    print("Length of original x_test: %d, the sum is: %d"%(len(x_test), sum(y_test)))
    
    # get the tokenizer
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # tokenization
    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = \
                    convert_data_to_ids(tokenizer, x_train_target, x_train)
    x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len = \
                    convert_data_to_ids(tokenizer, x_val_target, x_val)
    x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len = \
                    convert_data_to_ids(tokenizer, x_test_target, x_test)
    
    x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_len]
    x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len]
    x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_len]
    
    print(len(x_train), sum(y_train))
    print("Length of final x_train: %d"%(len(x_train)))
    
    return x_train_all,x_val_all,x_test_all


def data_loader(x_all, batch_size, model_select, mode, model_name, **kwargs):
    
    x_input_ids = torch.tensor(x_all[0], dtype=torch.long).cuda()
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    y = torch.tensor(x_all[3], dtype=torch.long).cuda()
    x_len = torch.tensor(x_all[4], dtype=torch.long).cuda()

    if model_name == 'student' and mode == 'train':
        y2 = torch.tensor(kwargs['y_train2'], dtype=torch.float).cuda()  # load teacher predictions
        tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len,y2)
    else:
        tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len)

    if mode == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
        data_loader_distill = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader, data_loader_distill
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

        return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader


def sep_test_set(input_data,dataset_name):
    
    # split the combined test set for each target
    if dataset_name == 'mt':
        data_list = [input_data[:355], input_data[890:1245], input_data[355:618],\
                     input_data[1245:1508], input_data[618:890], input_data[1508:1780]]
    elif dataset_name == 'wtwt':
        data_list = [input_data[:387], input_data[387:1361], input_data[1361:3041], input_data[3041:4217]]
    elif dataset_name == 'semeval':
        data_list = [input_data[:220], input_data[220:505], input_data[505:800], input_data[800:1080]]
    elif dataset_name == 'am':
        data_list = [input_data[:787], input_data[787:1396], input_data[1396:2127], input_data[2127:2796],
                     input_data[2796:3293], input_data[3293:3790], input_data[3790:4507], input_data[4507:5109]]
    elif dataset_name == 'covid':
        data_list = [input_data] # covid dataset only contains one target
    elif dataset_name == 'all':
        data_list = [input_data[:387], input_data[387:1361], input_data[1361:3041], input_data[3041:4217],\
                     input_data[4217:4572], input_data[4572:4835], input_data[4835:5107], input_data[5107:5462],\
                     input_data[5462:5725], input_data[5725:5997], input_data[5997:6217], input_data[6217:6502],\
                     input_data[6502:6797], input_data[6797:7077], input_data[7077:7441], input_data[7441:8228],\
                     input_data[8228:8837], input_data[8837:9568], input_data[9568:10237], input_data[10237:10734],\
                     input_data[10734:11231], input_data[11231:11948], input_data[11948:12550]]
    
    return data_list