import os 
import json
import pdb
from natsort import natsorted

try:
    from sklearn.metrics import f1_score
except ImportError:
    f1_score = None



dataset = 'empathic'
dir_path = "./output_pydantic"
key_dict = {"A": "Emotions", "B": "Engagement", "C": "Conversational Mechanics", "D": "Knowledge State", "E":"Intention", "F":"Social Context and Relationships", "G":"Social Norms and Routines"}



exps = os.listdir(dir_path)
string_list = []
for exp in exps: 
    
    exp_list = exp.split("_")
    

    if dataset not in exp:
        continue

    model = exp.split(dataset)[0]
    task = exp.split(dataset)[1]
    
    result_path = os.path.join(dir_path, exp)
    with open(result_path, 'r') as file:
        data = json.load(file)




    if len(data) == 0:
        continue

    correct_count = 0
    pred_labels = []
    gold_labels = []
    partial_match_count = 0

    for sample in data: 

        if len(sample['response']) == 0:
            continue

        
        if 'pydantic' in dir_path:
            answer= sample['response'][0]
        else:
            answer = sample['response']
  
        answer_list = sample['response']
        if 'detection' in task:
            pred = None
            gold = None
            if 'social competence' in answer.lower() or 'A' in answer:
                pred = 'competence'
                if sample['label'] == False:
                    correct_count += 1 
            elif 'social error' in answer.lower() or 'B' in answer:
                pred = 'error'
                if sample['label'] == True:
                    correct_count += 1
            elif 'C' in answer or 'none' in answer.lower(): 
                pred = 'none'
                if sample['label'] is None:
                    correct_count += 1 
            
            if sample['label'] is True:
                gold = 'error'
            elif sample['label'] is False:
                gold = 'competence'
            else:
                gold = 'none'
            pred_labels.append(pred)
            gold_labels.append(gold)

        if 'agreed_multiple_subj' in task:
            gold_count = sum(1 for v in sample['label'].values() if v == True)
            gold = 'True' if gold_count > 1 else 'False'
            pred = None
            if 'true' in answer.lower():
                pred = 'True'
                if pred == gold:
                    correct_count += 1
            elif 'false' in answer.lower():
                pred = 'False'
                if pred == gold:
                    correct_count += 1
            pred_labels.append(pred)
            gold_labels.append(gold)
            continue

        if 'attribute' in task:
            
            
            answer_k = None
            answer_v = None
            for k,v in key_dict.items():
                if k in answer:
                    answer_k = k
                    answer_v = v

            if answer_v == None:
                continue 
                
            
            if sample['label'][answer_v]: 

                correct_count += 1 

            gold_attrs = [k for k, v in sample['label'].items() if v == True]
            pred_attrs = [key_dict[k] for k in key_dict if k in answer_list]
            if any(p in gold_attrs for p in pred_attrs):
                partial_match_count += 1

            gold_vec = [1 if sample['label'].get(v, False) else 0 for v in key_dict.values()]
            pred_vec = [1 if v in pred_attrs else 0 for v in key_dict.values()]
            pred_labels.append(pred_vec)
            gold_labels.append(gold_vec)


        if 'attribute_agreed_multiple'in exp or 'attribute_disagree' in exp:
            inv_key_dict = {v: k for k, v in key_dict.items()}
        
            answer_k = None
            answer_v = None


            correct = True 
            for k,v in sample['label'].items():
                if v == True:
                    a_b_c = inv_key_dict[k]
                    if a_b_c in answer_list: 
                        pass 
                    else:
                        correct = False 
            
                if v == False:
                    a_b_c = inv_key_dict[k]
                    if a_b_c in answer_list: 
                        correct = False 
                    else:
                        pass

            correct_count += int(correct)
                    
        if 'post' in task or 'pre' in task or 'rationale' in task or'correction' in task:
            final_find_parenth = None
            for answer_v in ["1", "2", "3", "4", "5"]:
                find_parenth = answer.find(answer_v)
                if not find_parenth == -1:
                    final_find_parenth = answer_v
                    
            if final_find_parenth == None:
                continue

            if final_find_parenth in sample['label']:
                correct_count += 1 

    accuracy = round(correct_count/len(data),3)
    res_string = "Accuracy for {}: {}".format(exp, accuracy)
    
    print(res_string)

    if f1_score is not None and len(pred_labels) > 0:
        if 'detection' in task:
            safe_preds = [p if p is not None else 'unknown' for p in pred_labels]
            safe_golds = [g if g is not None else 'unknown' for g in gold_labels]
            labels_set = sorted(set(safe_golds + safe_preds))
            macro_f1 = f1_score(safe_golds, safe_preds, labels=labels_set, average='macro', zero_division=0)
            print("  Macro-F1 for {}: {:.3f}".format(exp, macro_f1))
        elif 'agreed_multiple_subj' in task:
            safe_preds = [p if p is not None else 'unknown' for p in pred_labels]
            safe_golds = [g if g is not None else 'unknown' for g in gold_labels]
            labels_set = sorted(set(safe_golds + safe_preds))
            macro_f1 = f1_score(safe_golds, safe_preds, labels=labels_set, average='macro', zero_division=0)
            print("  Macro-F1 for {}: {:.3f}".format(exp, macro_f1))
        elif 'attribute' in task and 'attribute_agreed_multiple' not in exp:
            macro_f1 = f1_score(gold_labels, pred_labels, average='macro', zero_division=0)
            print("  Macro-F1 for {}: {:.3f}".format(exp, macro_f1))
            pm = round(partial_match_count / len(data), 3)
            print("  Partial Match for {}: {}".format(exp, pm))