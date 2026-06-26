import os
import argparse
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, matthews_corrcoef, confusion_matrix

def compute_failure_detection_metrics(A: torch.Tensor, B: torch.Tensor):
    assert A.shape == B.shape, "A and B must have the same shape"
    A = A.flatten() # Labels
    B = B.flatten() # Predictions (binarized)
    
    # Use confusion matrix for reliable TP, TN, FP, FN
    try:
        TN, FP, FN, TP = confusion_matrix(A, B, labels=[0, 1]).ravel()
    except ValueError:
        # Fallback if confusion matrix fails
        TP = ((A == 1) & (B == 1)).sum().item()
        TN = ((A == 0) & (B == 0)).sum().item()
        FP = ((A == 0) & (B == 1)).sum().item()
        FN = ((A == 1) & (B == 0)).sum().item()

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_accuracy = (TPR + TNR) / 2
    
    total = A.numel()
    num_successes = (A == 0).sum().item()
    beta = num_successes / total if total > 0 else 0.0
    weighted_accuracy = beta * TPR + (1 - beta) * TNR
    
    mcc = abs((
        matthews_corrcoef(A, B)
        if (TP + TN + FP + FN) > 0
        else float("nan")
    ))

    return TPR, TNR, balanced_accuracy, weighted_accuracy, beta, mcc

def truncate_lists_to_tensor(list_of_lists):
    if not list_of_lists:
        return 0, 0, torch.tensor([])
    nb_of_episodes_label = len(list_of_lists)
    min_len = min(len(lst) for lst in list_of_lists)
    truncated = [torch.tensor(lst[:min_len]) for lst in list_of_lists]
    return min_len, nb_of_episodes_label, torch.stack(truncated).squeeze()

def truncate_to_match_shape(A, B, C, nb_of_episodes):
    tensors = [A, B, C]
    lengths = [t.numel() for t in tensors]
    min_length = min(lengths)

    if lengths.count(min_length) == 3:
        return A, B, C

    truncated_tensors = []
    for t in tensors:
        L = t.numel()
        if L == min_length:
            truncated_tensors.append(t)
        else:
            diff = L - min_length
            if nb_of_episodes > 0 and diff % nb_of_episodes == 0:
                cut_per_episode = diff // nb_of_episodes
                episode_length = L // nb_of_episodes
                t_reshaped = t.view(nb_of_episodes, episode_length)
                t_truncated = t_reshaped[:, :-cut_per_episode]
            else:
                t_truncated = t[:-diff]
            truncated_tensors.append(t_truncated.flatten())

    return tuple(truncated_tensors)

def main():
    parser = argparse.ArgumentParser(description="Compute threshold scores for FIDeL.")
    parser.add_argument("--score_dir", type=str, default="./results/score", help="Directory containing score results")
    parser.add_argument("--labels_dir", type=str, default="./Labels", help="Directory containing ground truth labels")
    parser.add_argument("--safe_labels_dir", type=str, default=None, help="Directory containing safety ground truth labels (optional)")
    parser.add_argument("--task_name", type=str, required=True, help="Task name (e.g., domotic_setTheTable_anomaly)")
    parser.add_argument("--ad_type", type=str, default="Representation", help="Anomaly detection type")
    parser.add_argument("--encoder", type=str, default="dinoV2", help="Encoder type")
    parser.add_argument("--threshold_type", type=str, default="conformal_prediction_time_all", help="Threshold type")
    parser.add_argument("--distance_type", type=str, default="euclidean", help="Distance type")
    
    args = parser.parse_args()

    task_name = args.task_name
    AD_type = args.ad_type

    if AD_type in ["AE", "lopO", "logpZ0"]:
        eval_name = f"score_{args.encoder}_{task_name}_{AD_type}_{args.threshold_type}"
    elif AD_type == "Representation_no_variability":
        eval_name = f"score_{task_name}_{AD_type}_{args.threshold_type}"
    elif AD_type == "Representation":
        eval_name = f"score_{args.encoder}_{task_name}_{AD_type}_{args.distance_type}_{args.threshold_type}"
    else:
        raise NotImplementedError(f"AD type {AD_type} not implemented")

    ground_truth_folder = os.path.join(args.labels_dir, task_name)
    safe_ground_truth_folder = args.safe_labels_dir

    if not os.path.exists(ground_truth_folder):
        print(f"Ground truth folder not found: {ground_truth_folder}")
        return

    csv_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith("_labels.csv")])
    all_labels = []
    for csv_file in csv_files:
        path = os.path.join(ground_truth_folder, csv_file)
        labels = np.loadtxt(path, delimiter=",", dtype=int).reshape(-1, 1)
        all_labels.append(labels)

    min_len, nb_of_episodes_label, tensor_labels = truncate_lists_to_tensor(all_labels)

    all_safe_labels = []
    has_safe_labels = False
    if safe_ground_truth_folder and os.path.exists(safe_ground_truth_folder):
        safe_csv_files = sorted([f for f in os.listdir(safe_ground_truth_folder) if f.endswith("_labels.csv")])
        for safe_csv_file in safe_csv_files:
            path = os.path.join(safe_ground_truth_folder, safe_csv_file)
            labels = np.loadtxt(path, delimiter=",", dtype=int).reshape(-1, 1)
            all_safe_labels.append(labels)
        _, _, safe_tensor_labels = truncate_lists_to_tensor(all_safe_labels)
        has_safe_labels = True

    eval_score_file_path = os.path.join(args.score_dir, f"{eval_name}.pt")
    
    if os.path.exists(eval_score_file_path):
        loaded_data = torch.load(eval_score_file_path, weights_only=False)
        eval_score = abs(loaded_data["eval_score"])
        
        # Determine key for threshold scores
        threshold_key = "failure_values" if "failure_values" in loaded_data else "anomaly_values"
        threshold_eval_score = loaded_data.get(threshold_key, None)
        safe_eval_score = loaded_data.get("safety_values", None)
        param_list = loaded_data.get("param_list", [])
    else:
        print(f"Score file not found: {eval_score_file_path}")
        return


    tensor_labels, eval_score, _ = truncate_to_match_shape(
        tensor_labels, eval_score, eval_score, nb_of_episodes_label
    )
    labels_flatten = tensor_labels.flatten()

    if has_safe_labels and safe_eval_score is not None:
        safe_labels_flatten = safe_tensor_labels.flatten()

    # Raw score evaluation
    fpr, tpr, thresholds = roc_curve(labels_flatten, eval_score)
    roc_auc = auc(fpr, tpr)
    y_score = eval_score.cpu().numpy()
    
    precision, recall, pr_thresholds = precision_recall_curve(labels_flatten, y_score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = pr_thresholds[best_idx]

    eval_score_binarized = eval_score >= best_thresh
    TPR_raw, TNR_raw, balanced_accuracy_raw, weighted_accuracy_raw, beta_raw, MCC = compute_failure_detection_metrics(labels_flatten, eval_score_binarized)

    print(f"**************** {AD_type} - {task_name} ***************")
    print("=========== raw_score ==========")
    print(f"Best F1 = {best_f1:.3f} at threshold = {best_thresh:.3f}")
    print(f"Global AUC-ROC: {roc_auc:.2f}")
    print(f"TPR_raw: {TPR_raw:.3f}, TNR_raw: {TNR_raw:.3f}, balanced_accuracy_raw: {balanced_accuracy_raw:.3f}, weighted_accuracy_raw: {weighted_accuracy_raw:.3f}, MCC: {MCC:.3f}")

    if threshold_eval_score is not None and param_list:
        if threshold_eval_score.ndim == 1:
            threshold_eval_score = threshold_eval_score.unsqueeze(1)
            
        for idx in range(threshold_eval_score.shape[1]):
            threshold_col = threshold_eval_score[:, idx]
            params = param_list[idx]

            _, _, threshold_col_truncated = truncate_to_match_shape(
                tensor_labels, eval_score, threshold_col, nb_of_episodes_label
            )

            TPR, TNR, balanced_accuracy, weighted_accuracy, beta, MCC_t = compute_failure_detection_metrics(labels_flatten, threshold_col_truncated)
            
            print(f"\n========== Threshold Combination {idx + 1} ==========")
            print(f"Params: {params}")
            print(f"TPR: {TPR:.3f}, TNR: {TNR:.3f}, Balanced Accuracy: {balanced_accuracy:.3f}, Weighted Accuracy: {weighted_accuracy:.3f}, MCC: {MCC_t:.3f}")

            if has_safe_labels and safe_eval_score is not None:
                safe_col = safe_eval_score[:, idx]
                _, _, safe_col_truncated = truncate_to_match_shape(
                    tensor_labels, eval_score, safe_col, nb_of_episodes_label
                )
                TPR_safety, TNR_safety, balanced_accuracy_safety, weighted_accuracy_safety, beta_safety, MCC_safety = compute_failure_detection_metrics(safe_labels_flatten, safe_col_truncated)
                print(f"Safety -> TPR: {TPR_safety:.3f}, TNR: {TNR_safety:.3f}, Balanced Accuracy: {balanced_accuracy_safety:.3f}, Weighted Accuracy: {weighted_accuracy_safety:.3f}, MCC: {MCC_safety:.3f}")

if __name__ == "__main__":
    main()
