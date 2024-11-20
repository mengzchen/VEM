import  json
import  numpy as np
from sklearn.metrics import accuracy_score

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def str2dic(content):
    exclude_keys = {"thought", "plan"}
    try:
        content = json.loads(content)
        content = {k: v for k, v in content.items() if k not in exclude_keys}
        content = flatten_dict(content)

        return content
    except:
        return None

def get_part_metrix(gt, pred, part):
    gt = {k: v for k, v in gt.items() if k in part}
    pred = {k: v for k, v in pred.items() if k in part}

    all_keys = set(gt.keys()).union(set(pred.keys()))
    gt_vector, pred_vector = [], []

    for key in all_keys:
        gt_value, pred_value = gt.get(key), pred.get(key)
        # print(f"{gt_value} ==> {pred_value}")
        gt_vector.append(1 if gt_value is not None else 0)
        pred_vector.append(1 if gt_value == pred_value else 0)
    
    acc = accuracy_score(gt_vector, pred_vector)

    return acc
    

def compute_ufo(anns):
    error_count = 0
    accs_p1, accs_p2, accs_p3 = [], [], []
    step_scc = []
    task_dic = {}

    for ann in anns:
        task_id = ann["images"][0].split("/")[3]
        gt = str2dic(ann["messages"][1]["content"])
        pred = str2dic(ann["prediction"])
        
        if gt == None or pred == None:
            error_count += 1
            continue
        
        accs_p1.append(get_part_metrix(gt, pred, part={"control_label", "control_name"}))
        accs_p2.append(get_part_metrix(gt, pred, part={"function", "args"}))
        accs_p3.append(get_part_metrix(gt, pred, part={"status"}))

        if task_id not in task_dic.keys():
            task_dic[task_id] = []
        if accs_p1[-1] == 1 and accs_p2[-1] ==1 and accs_p3[-1] ==1:
            step_scc.append(1)
            task_dic[task_id].append(1)
        else:
            step_scc.append(0)
            task_dic[task_id].append(0)

    task_scc = []
    for k, v in task_dic.items():
        if 0 not in v:
            task_scc.append(1)
        else:
            task_scc.append(0)

    print(f'elem_acc:     {np.array(accs_p1).mean()}')
    print(f'args_acc:     {np.array(accs_p2).mean()}')
    print(f'status_acc:   {np.array(accs_p3).mean()}')
    print(f'step_success: {np.array(step_scc).mean()}')
    print(f'task_success: {np.array(task_scc).mean()}')
    print(f'error_count:  {error_count}')
