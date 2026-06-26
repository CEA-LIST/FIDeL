import os
import glob
import re

replacements = {
    r"Implémente la méthode de Conformal Prediction temporelle ET spatiale en une seule classe": r"Implements temporal AND spatial Conformal Prediction in a single class",
    r"Implémente la méthode de Conformal Prediction temporelle avec choix de scalA\(t\)": r"Implements temporal Conformal Prediction with choice of scalA(t)",
    r"Implémente la méthode de Conformal Prediction spatiale avec choix de scalA\(t\)": r"Implements spatial Conformal Prediction with choice of scalA(t)",
    r"Implémente une méthode de Conformal Prediction Globale \(invariante dans le temps\)\.": r"Implements a Global Conformal Prediction method (time-invariant).",
    r"Au lieu de chercher une couverture conjointe sur toute la trajectoire \(qui explose avec le \.max\(\)\),": r"Instead of seeking joint coverage over the entire trajectory (which explodes with .max()),",
    r"cette méthode garantit une couverture marginale : \(1 - alpha\)% des frames normales seront sous le seuil\.": r"this method guarantees marginal coverage: (1 - alpha)% of normal frames will be below the threshold.",
    r"cfg: config contenant N \(utilisé si scalA\(t\) est locale\)": r"cfg: config containing N (used if scalA(t) is local)",
    r"alpha: niveau de confiance \(ex: 0\.60, 0\.95, 0\.99\)": r"alpha: confidence level (e.g. 0.60, 0.95, 0.99)",
    r"alpha: niveau de rejet \(ex: 0\.05 pour 95% confiance\)": r"alpha: rejection level (e.g. 0.05 for 95% confidence)",
    r"Étape 1: Split DcalA_time / DcalB_time - DcalA_space / DcalB_space": r"Step 1: Split DcalA_time / DcalB_time - DcalA_space / DcalB_space",
    r"Étape 1: Split DcalA / DcalB": r"Step 1: Split DcalA / DcalB",
    r"Étape 2:": r"Step 2:",
    r"Moyenne μₜ": r"Mean μₜ",
    r"Moyenne µp": r"Mean µp",
    r"Moyenne μp sur les patches": r"Mean μp on patches",
    r"Étape 3: scalA\(t\), scalA\(p\)": r"Step 3: scalA(t), scalA(p)",
    r"Étape 3: scalA\(t\)": r"Step 3: scalA(t)",
    r"Étape 4: calcul du h_time, h_space et des seuils ηₜ et ηp": r"Step 4: compute h_time, h_space and thresholds ηₜ and ηp",
    r"Étape 4: calcul du h et du seuil final ηₜ": r"Step 4: compute h and final threshold ηₜ",
    r"Étape 4: calcul du h et du seuil final ηp": r"Step 4: compute h and final threshold ηp",
    r"Variante 1": r"Variant 1",
    r"Variante 2": r"Variant 2",
    r"constante pour tous les t": r"constant for all t",
    r"constante pour tous les p": r"constant for all p",
    r"sur DcalA": r"on DcalA",
    r"Calcule h comme le quantile \(1 - alpha\) des max_deviation normalisés sur DcalB": r"Computes h as the (1 - alpha) quantile of the normalized max_deviation on DcalB",
    r"max sur les patchs": r"max on patches",
    r"max sur le temps": r"max on time",
    r"nombre total d'éléments": r"total number of elements",
    r"nombre de True \(puisqu'ils valent 1\)": r"number of True (since they equal 1)",
    r"si 10% des patchs détecte une anomalie, on passe le flag à true": r"if 10% of patches detect an anomaly, flag is set to true",
    r"si 10% des patchs détecte une anomalie, on déclenche": r"if 10% of patches detect an anomaly, trigger",
    r"dans cette version, les param sont des listes": r"in this version, parameters are lists",
    r"le but c'est d'extraires plusieurs scores d'un coup avec tous les param": r"the goal is to extract multiple scores at once with all parameters",
    r"Calcul du seuil basé sur la valeur absolue de l'écart normalisé \(z-score\)": r"Threshold calculation based on the absolute value of the normalized deviation (z-score)",
    r"Convertit le niveau de confiance \(alpha\) en seuil de score z": r"Converts confidence level (alpha) to z-score threshold",
    r"Alpha -> seuil z-score \(ex: 1\.96 pour 95%\)": r"Alpha -> z-score threshold (e.g. 1.96 for 95%)",
    r"Calcule pour chaque timestep la moyenne et l'écart-type des moyennes locales": r"Computes the mean and standard deviation of local means for each timestep",
    r"de chaque épisode, sans padding, on utilise seulement les voisins disponibles\.": r"of each episode, without padding, using only available neighbors.",
    r"Nombre de voisins à gauche et à droite": r"Number of neighbors to the left and right",
    r"liste de Tuple\[torch\.Tensor, torch\.Tensor\]: moyennes et écarts-types, shape \[nb_timesteps\], pour N_all": r"list of Tuple[torch.Tensor, torch.Tensor]: means and standard deviations, shape [nb_timesteps], for N_all",
    r"On prépare un tenseur pour stocker les moyennes locales": r"Prepare a tensor to store local means",
    r"moyenne des 2N scores pour chaque timestep": r"mean of 2N scores for each timestep",
    r"Moyenne et écart-type sur les épisodes": r"Mean and standard deviation across episodes",
    r"Retourne True si le score est anormal \(au-dessus du seuil\)": r"Returns True if the score is anomalous (above the threshold)",
    r"Comparaison vectorisée sur les seuils alpha \(shape: \[nb_alpha\]\)": r"Vectorized comparison on alpha thresholds (shape: [nb_alpha])",
    r"Comparaison vectorisée sur les seuils N \(shape: \[nb_N\]\)": r"Vectorized comparison on N thresholds (shape: [nb_N])",
    r"Concatène tous les résultats dans un seul vecteur booléen \[nb_alpha \+ nb_N\]": r"Concatenates all results into a single boolean vector [nb_alpha + nb_N]",
    r"moyenne globale des scores experts \(Tensor scalaire\)": r"global mean of expert scores (scalar Tensor)",
    r"écart-type global des scores experts \(Tensor scalaire\)": r"global standard deviation of expert scores (scalar Tensor)",
    r"Aplatir tous les scores experts, car le score Representation est invariant temporellement": r"Flatten all expert scores, because the Representation score is time-invariant",
    r"Pour chaque niveau de confiance alpha, le seuil est simplement le quantile \(1 - alpha\)": r"For each confidence level alpha, the threshold is simply the (1 - alpha) quantile",
    r"On compare le score au seuil global \(indépendant de t\)": r"Compare the score to the global threshold (independent of t)",
    r"!!! /!\\ Ne marche que si encodeur de type DinoV2 qui output des features par patch !!!": r"!!! /!\\ Only works if the encoder is DinoV2 which outputs features per patch !!!"
}

files = glob.glob('/home/kant/Documents/Publications/ICRA_2026/website/FIDeL/src/threshold/*.py')
files.extend(glob.glob('/home/kant/Documents/Publications/ICRA_2026/website/FIDeL/src/cfgs/threshold_type/*.yaml'))

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for fr, en in replacements.items():
        content = re.sub(fr, en, content)
        
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

print("Translation script completed.")
