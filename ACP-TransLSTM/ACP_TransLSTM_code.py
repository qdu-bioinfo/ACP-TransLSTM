import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F

'''A functional extract_features is defined for
extracting a matrix of features from a given peptide sequence.'''
amino_acid_groups = {
    'A': 1, 'G': 1, 'V': 1, 'I': 2, 'L': 2, 'F': 2, 'P': 2, 'Y': 3, 'M': 3, 'T': 3, 'S': 3,
    'H': 4, 'N': 4, 'Q': 4, 'W': 4, 'R': 5, 'K': 5, 'D': 6, 'E': 6, 'C': 7
}
def extract_features(peptide_sequence, k):
    n = 7 ** k
    l = len(peptide_sequence)
    matrix = [[0] * (l - k + 1) for _ in range(n)]
    for i in range(l - k + 1):
        kmer = peptide_sequence[i:i + k]
        kmer_code = 0
        for aa in kmer:
            group_num = amino_acid_groups.get(aa, 0)
            kmer_code = kmer_code * 7 + group_num
        if kmer_code < n:
            matrix[kmer_code][i] = 1
    return matrix

'''Used to encode amino acid sequences as vector numbers.'''
def encode_peptide_sequence(sequence):
    amino_acids_to_numbers = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    if isinstance(sequence, str):
        encoded_vector = [amino_acids_to_numbers.get(aa, 0) for aa in sequence]
        if len(encoded_vector) < 55:
            while len(encoded_vector) < 55:
                encoded_vector.insert(0, 0)
        elif len(encoded_vector) > 55:
            encoded_vector = encoded_vector[:55]
        return encoded_vector
    else:
        return [0] * 55

'''Used to calculate the frequency of occurrence of each amino acid in a protein sequence'''
def calculate_aac(protein_sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac_vector = [0] * 20
    for aa in str(protein_sequence):
        if aa in amino_acids:
            index = amino_acids.index(aa)
            aac_vector[index] += 1
    sequence_length = len(str(protein_sequence))
    return [count / sequence_length for count in aac_vector]

'''Used to calculate the frequency of occurrence of each dipeptide in a protein sequence'''
def calculate_dpc(protein_sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    dpc_vector = [0] * len(dipeptides)
    sequence_length = len(str(protein_sequence))
    for i in range(sequence_length - 2):
        dipeptide = str(protein_sequence)[i] + str(protein_sequence)[i + 2]
        if dipeptide in dipeptides:
            index = dipeptides.index(dipeptide)
            dpc_vector[index] += 1
    return [count / (sequence_length - 2) for count in dpc_vector]

'''Used to calculate the frequency of occurrence of specific amino acid pairs in a protein sequence.
   The specific amino acid pairs are grouped according to their functional properties, such as aliphatic, aromatic, positive charge, negative charge, and uncharged.'''

def calculate_group_pair_frequency(sequence, k=0):
    groups = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    group_keys = list(groups.keys())
    group_pair_counts = {}
    for key1 in group_keys:
        for key2 in group_keys:
            group_pair_counts[key1 + '.' + key2] = 0
    total_count = 0
    for i in range(len(sequence)):
        if i + k + 1 < len(sequence):
            aa1 = sequence[i]
            aa2 = sequence[i + k + 1]
            for group_key, group_aa in groups.items():
                if aa1 in group_aa:
                    group1 = group_key
                if aa2 in group_aa:
                    group2 = group_key
            if group1 and group2:
                group_pair_counts[group1 + '.' + group2] += 1
                total_count += 1
    frequencies = []
    for pair, count in group_pair_counts.items():
        frequencies.append(count / total_count if total_count > 0 else 0)
    return frequencies

'''A Transformer-based encoder for processing sequence data.
The encoder takes a sequence of amino acids as input and uses a positional encoding scheme to add positional information to the input. The positional encoding'''
def positional_encoding(pos, d_model):
    def get_angles(position, i):
        return position / np.power(10000., 2. * (i // 2.) / float(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    pe_sin = np.sin(angle_rates[:, 0::2])
    pe_cos = np.cos(angle_rates[:, 1::2])
    pos_encoding = np.concatenate([pe_sin, pe_cos], axis=-1)
    return pos_encoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src.float()
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dff):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dff)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        src = src.float()
        output = src
        for layer in self.layer_stack:
            output = layer(output, mask, src_key_padding_mask)
        return self.norm(output)

def process_peptide_sequence(peptide_sequence):
    if isinstance(peptide_sequence, float):
        peptide_sequence = str(peptide_sequence)
    peptide_list = list(peptide_sequence)
    if len(peptide_list) < 50:
        padded_sequence = peptide_list + [0] * (50 - len(peptide_list))
    else:
        padded_sequence = peptide_list[:50]
    char_to_num = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13,
                   'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    amino_acid_to_vector = {
        'A': [0.96, 16.00, 1.43, 89.30, 9.36, 7.90, 0.92, -0.04],
        'C': [0.42, 168.00, 0.94, 102.50, 2.56, 1.90, 1.16, -0.38],
        'D': [0.42, -78.00, 0.92, 114.40, 0.94, 5.50, 0.48, 0.19],
        'E': [0.53, -106.00, 1.67, 138.80, 0.94, 7.10, 0.61, 0.23],
        'F': [0.59, 189.00, 1.19, 190.80, 10.99, 3.90, 1.25, -0.38],
        'G': [0.00, -13.00, 0.46, 63.80, 6.17, 7.10, 0.61, 0.09],
        'H': [0.57, 50.00, 0.98, 157.50, 0.47, 2.10, 0.93, -0.04],
        'I': [0.84, 151.00, 1.04, 163.00, 13.73, 5.20, 1.81, -0.34],
        'K': [0.73, -141.00, 1.27, 165.10, 0.58, 6.70, 0.70, 0.33],
        'L': [0.92, 145.00, 1.36, 163.10, 16.64, 8.60, 1.30, -0.37],
        'M': [0.86, 124.00, 1.53, 165.80, 3.93, 2.40, 1.19, -0.30],
        'N': [0.39, -74.00, 0.64, 122.40, 2.31, 4.00, 0.60, 0.13],
        'P': [-2.50, -20.00, 0.49, 121.60, 1.96, 5.30, 0.40, 0.19],
        'Q': [0.80, -73.00, 1.22, 146.90, 1.14, 4.40, 0.95, 0.14],
        'R': [0.77, -70.00, 1.18, 190.30, 0.27, 4.90, 0.93, 0.07],
        'S': [0.53, -70.00, 0.70, 94.20, 5.58, 6.60, 0.82, 0.12],
        'T': [0.54, -38.00, 0.78, 119.60, 4.68, 5.30, 1.12, 0.03],
        'V': [0.63, 123.00, 0.98, 138.20, 12.43, 6.80, 1.81, -0.29],
        'W': [0.58, 145.00, 1.01, 226.40, 2.20, 1.20, 1.54, -0.33],
        'Y': [0.72, 53.00, 0.69, 194.60, 3.13, 3.10, 1.53, -0.29]
    }
    extended_array = []
    for aa in padded_sequence:
        if aa == 0:
            extended_array.append([0] * 8)
        else:
            try:
                vector = amino_acid_to_vector[list(char_to_num.keys())[list(char_to_num.values()).index(aa)]]
                extended_array.append(vector * 8)
            except ValueError:
                extended_array.append([0] * 8)
    extended_array = np.array(extended_array).reshape(50, 8)

    pos_encoding = torch.tensor(positional_encoding(50, 8), dtype=torch.float32)
    encoded_array = torch.tensor(extended_array, dtype=torch.float32) + pos_encoding
    encoder = Encoder(d_model=8, nhead=2, num_encoder_layers=6, dff=16)
    encoded_tensor = encoder(encoded_array.unsqueeze(0))
    feature_vector = F.avg_pool1d(encoded_tensor.squeeze(0).transpose(0, 1), kernel_size=encoded_tensor.size(1)).squeeze(1)
    return feature_vector.detach().numpy()


def run_cross_validation():
    X = []
    y = []
    protein_seq_dict = {}
    label = []

    with open('ACP530.txt', 'r') as fp:
        current_label = None
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[1]
                if label_temp == '1':
                    current_label = 1
                else:
                    current_label = 0
                label.append(current_label)
            else:
                seq = line[:-1]
                features = extract_features(seq, k=3)
                if current_label not in protein_seq_dict:
                    protein_seq_dict[current_label] = [features]
                else:
                    protein_seq_dict[current_label].append(features)

    pca = PCA(n_components=50)
    for label_value in protein_seq_dict:
        sequences = protein_seq_dict[label_value]
        max_length = max(len(feature) for sequence in sequences for feature in sequence)
        padded_sequences = [[feature + [0] * (max_length - len(feature)) for feature in sequence] for sequence in sequences]
        all_features = np.array([feature for sequence in padded_sequences for feature in sequence])
        transformed_features = pca.fit_transform(all_features)
        protein_seq_dict[label_value] = transformed_features.tolist()

    for lbl, vectors in protein_seq_dict.items():
        for vec in vectors:
            encoded_seq = encode_peptide_sequence(vec[0])
            aac_features = calculate_aac(vec[0])
            dpc_features = calculate_dpc(vec[0])
            new_feature = process_peptide_sequence(vec[0])
            all_vectors = []
            for k in range(6):
                vector_k = calculate_group_pair_frequency(seq, k=k)
                all_vectors.extend(vector_k)
            combined_features = np.concatenate((vec, encoded_seq, aac_features, dpc_features, all_vectors, new_feature))
            X.append(combined_features)
            y.append(lbl)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    z = tf.random.normal(shape=tf.shape(X), mean=0.0, stddev=0.003)
    X = X + z
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_sensitivities = []
    fold_specificities = []
    fold_precisions = []
    fold_f1_scores = []
    fold_mccs = []
    fold_aucs = []

    for fold, (train_indices, test_indices) in enumerate(kf.split(X)):
        train_indices_tensor = tf.convert_to_tensor(train_indices, dtype=tf.int32)
        test_indices_tensor = tf.convert_to_tensor(test_indices, dtype=tf.int32)
        X_train, X_test = tf.gather(X, train_indices_tensor), tf.gather(X, test_indices_tensor)
        y_train, y_test = tf.gather(y, train_indices_tensor), tf.gather(y, test_indices_tensor)

        normalization_layer = tf.keras.layers.Normalization()
        normalization_layer.adapt(X_train)
        X_train = normalization_layer(X_train)
        X_test = normalization_layer(X_test)

        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((len(combined_features), 1)),
            tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=5, batch_size=128)
        y_pred = model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()
        accuracy = accuracy_score(y_test, y_pred_classes)
        sensitivity = recall_score(y_test, y_pred_classes)
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred_classes)
        mcc = matthews_corrcoef(y_test, y_pred_classes)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_classes).ravel()
        accuracy = accuracy_score(y_test, y_pred_classes)
        sensitivity = recall_score(y_test, y_pred_classes)
        specificity = tn / (tn + fp)
        precision = precision_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes)
        mcc = matthews_corrcoef(y_test, y_pred_classes)
        auc = roc_auc_score(y_test, y_pred.flatten())

        print(f"Fold {fold + 1}:")
        print(f"Accuracy: {accuracy}")
        print(f"Sensitivity/Recall: {sensitivity}")
        print(f"Specificity: {specificity}")
        print(f"Precision: {precision}")
        print(f"F1 Score: {f1}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc}")
        print(f"AUC: {auc}")
        print("————————————————————————————————————————————————————")

        fold_accuracies.append(accuracy)
        fold_sensitivities.append(sensitivity)
        fold_specificities.append(specificity)
        fold_precisions.append(precision)
        fold_f1_scores.append(f1)
        fold_mccs.append(mcc)
        fold_aucs.append(auc)

    print(f"Average Accuracy: {sum(fold_accuracies) / num_folds}")
    print(f"Average Sensitivity/Recall: {sum(fold_sensitivities) / num_folds}")
    print(f"Average Specificity: {sum(fold_specificities) / num_folds}")
    print(f"Average Precision: {sum(fold_precisions) / num_folds}")
    print(f"Average F1 Score: {sum(fold_f1_scores) / num_folds}")
    print(f"Average Matthews Correlation Coefficient (MCC): {sum(fold_mccs) / num_folds}")
    print(f"Average AUC: {sum(fold_aucs) / num_folds}")


if __name__ == "__main__":
    run_cross_validation()