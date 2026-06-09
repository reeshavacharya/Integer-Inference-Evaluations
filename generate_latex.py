import os
import json
import glob

def format_val(v):
    if v is None:
        return "-"
    return f"{v:.6f}"

def format_delta(fp, int_v, m_name=""):
    if fp is None or int_v is None:
        return "-"
    if fp == 0:
        return "-"
    delta = ((int_v - fp) / fp) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f}\\%"

def parse_val(v):
    if isinstance(v, dict):
        return v
    else:
        return {"ACC": v}

def extract_classification(arch):
    data = {}
    files = glob.glob(f"{arch}/benchmark-results/*/*.json")
    for f in files:
        basename = os.path.basename(f)[:-5]
        if "leaky" in basename:
            continue
        if "fp32" not in basename and "int32" not in basename:
            continue
            
        activation = "RELU" if "relu" in basename else "GELU"
        precision = "fp32" if "fp32" in basename else "int32"
        dataset = os.path.basename(os.path.dirname(f))
        
        with open(f) as file:
            content = json.load(file)
            top_key = list(content.keys())[0]
            try:
                metrics = parse_val(content[top_key][precision])
            except KeyError:
                continue
            
        if dataset not in data:
            data[dataset] = {}
            
        for m_name, m_val in metrics.items():
            if m_name == "ACC":
                m_name = "Accuracy%"
            
            if m_name not in data[dataset]:
                data[dataset][m_name] = {}
            if activation not in data[dataset][m_name]:
                data[dataset][m_name][activation] = {}
            data[dataset][m_name][activation][precision] = m_val
    return data

def extract_unet():
    data = {}
    files = glob.glob("unet/benchmark-results/*/*.json")
    for f in files:
        basename = os.path.basename(f)[:-5]
        if "leaky_relu" in basename:
            continue
        
        activation = "RELU" if "relu" in basename else "GELU"
        precision = "fp32" if "fp32" in basename else "int32"
        dataset = os.path.basename(os.path.dirname(f))
        
        try:
            with open(f) as file:
                content = json.load(file)
        except Exception:
            continue
            
        top_key = list(content.keys())[0]
        metrics = content[top_key]
            
        if dataset not in data:
            data[dataset] = {}
            
        for m_name in ['dice', 'iou', 'acc']:
            if m_name not in metrics:
                continue
            m_val = metrics[m_name]
            if m_name == 'acc':
                m_name = 'Acc'
                if m_val <= 1.0:
                    m_val *= 100
            elif m_name == 'dice':
                m_name = 'Dice'
            elif m_name == 'iou':
                m_name = 'IOU'
                
            if m_name not in data[dataset]:
                data[dataset][m_name] = {}
            if activation not in data[dataset][m_name]:
                data[dataset][m_name][activation] = {}
            data[dataset][m_name][activation][precision] = m_val
    return data

def generate_latex_table(title, data, order, name_map):
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{" + title + r" Benchmark Results}")
    latex.append(r"\begin{tabular}{|l|l|c|c|c|c|c|c|}")
    latex.append(r"\hline")
    latex.append(r"\multirow{2}{*}{Datasets} & \multirow{2}{*}{Metric} & \multicolumn{3}{c|}{ReLU} & \multicolumn{3}{c|}{GELU} \\")
    latex.append(r"\cline{3-8}")
    latex.append(r"& & FP32 & INT32 & $\Delta$ & FP32 & INT32 & $\Delta$ \\")
    latex.append(r"\hline")
    
    for ds in order:
        if ds not in data:
            continue
            
        metrics = data[ds]
        ds_display_name = name_map.get(ds, ds)
        
        # We need to print metrics in a consistent order
        metric_keys = list(metrics.keys())
        if 'AUC' in metric_keys and 'Accuracy%' in metric_keys:
            # force AUC first, then Accuracy%
            metric_keys = ['AUC', 'Accuracy%']
        elif 'Dice' in metric_keys and 'IOU' in metric_keys and 'Acc' in metric_keys:
            metric_keys = ['Dice', 'IOU', 'Acc']
            
        first_metric = True
        for m_name in metric_keys:
            acts = metrics[m_name]
            relu = acts.get("RELU", {})
            gelu = acts.get("GELU", {})
            
            relu_fp32 = relu.get("fp32")
            relu_int32 = relu.get("int32")
            gelu_fp32 = gelu.get("fp32")
            gelu_int32 = gelu.get("int32")
            
            relu_delta = format_delta(relu_fp32, relu_int32, m_name)
            gelu_delta = format_delta(gelu_fp32, gelu_int32, m_name)
            
            ds_str = ds_display_name.replace('_', '\\_') if first_metric else ""
            
            row = f"{ds_str} & {m_name.replace('%', r'\%')} & {format_val(relu_fp32)} & {format_val(relu_int32)} & {relu_delta} & {format_val(gelu_fp32)} & {format_val(gelu_int32)} & {gelu_delta} \\\\"
            latex.append(row)
            first_metric = False
        latex.append(r"\hline")
        
    latex.pop() # remove last hline
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")
    return "\n".join(latex)

def main():
    classification_order = ["mnist", "cifar10", "brain_mri", "octmnist", "organamnist", "bloodmnist", "pneumoniamnist"]
    classification_map = {
        "mnist": "MINST",
        "cifar10": "CIFAR10",
        "brain_mri": "BRAIN-MRI",
        "octmnist": "OCTMNIST",
        "organamnist": "ORGANAMNIST",
        "bloodmnist": "BLOODMNIST",
        "pneumoniamnist": "PNEUMONIAMNIST"
    }
    
    unet_order = ["flood", "brain-mri-seg", "busi", "skin-lesion"]
    unet_map = {
        "flood": "FLOOD",
        "brain-mri-seg": "BRAIN-MRI",
        "busi": "BUSI",
        "skin-lesion": "SKIN-LESION"
    }

    with open("results-table.tex", "w") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage{booktabs}" + "\n")
        f.write(r"\usepackage{multirow}" + "\n")
        f.write(r"\usepackage[margin=0.5in]{geometry}" + "\n")
        f.write(r"\begin{document}" + "\n\n")
        
        for arch in ["lenet", "resnet", "VGGNet"]:
            data = extract_classification(arch)
            table = generate_latex_table(arch.capitalize(), data, classification_order, classification_map)
            f.write(table + "\n")
            
        data = extract_unet()
        table = generate_latex_table("UNet", data, unet_order, unet_map)
        f.write(table + "\n")
        
        f.write(r"\end{document}" + "\n")

if __name__ == "__main__":
    main()
