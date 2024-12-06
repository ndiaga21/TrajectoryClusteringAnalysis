
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet, leaves_list
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch

import Levenshtein
from tslearn.metrics import dtw, dtw_path_from_metric, gak

import tqdm

import timeit
import logging
#import logging

##########



class TCA:

    def __init__(self, data, id, alphabet, states, colors='viridis'):
        self.data = data
        self.id = id
        self.alphabet = alphabet
        self.states = states
        self.colors = colors
        self.leaf_order = None
        self.substitution_cost_matrix = None
        logging.basicConfig(level=logging.INFO)
        
        # if len(self.colors) != len(self.state_label):
        #     logging.error("Number of colors and states mismatch")
        #     raise ValueError("The number of colors must match the number of states")
        
        assert(isinstance(data, pd.DataFrame)), "data must be a pandas DataFrame"
        assert(data.shape[1] > 1), "data must have more than one column"
        assert(data.id.duplicated().sum() == 0), "There are duplicates in the data. Yout dataset must be in long and tidy format "

        # if not set(self.state_label).issubset(data.columns):
        #     logging.error("States not found in data columns")
        #     raise ValueError("States not found in data columns")

        print("Dataset :")
        print("data shape: ", self.data.shape)
        mapping_df = pd.DataFrame({'alphabet':self.alphabet, 'label':self.states, 'label encoded':range(1,1+len(self.alphabet))})
        print("state coding:\n", mapping_df)

        # print("colors: ", self.colors)

        # Convert the sequences to a NumPy array for faster processing
        # Ajouter une colonne "Sequence" qui contient la séquence de soins pour chaque individu
        data_ready_for_TCA = self.data.copy()
        data_ready_for_TCA['Sequence'] = data_ready_for_TCA.drop(self.id, axis=1).apply(lambda x: '-'.join(x.astype(str)), axis=1)
        # data_ready_for_TCA.reset_index(inplace=True)
        data_ready_for_TCA = data_ready_for_TCA[['id', 'Sequence']]
        self.sequences = data_ready_for_TCA['Sequence'].apply(lambda x: np.array([k for k in x.split('-') if k != 'nan'])).to_numpy()

        # Create a dictionary mapping labels to encoded values
        self.label_to_encoded = mapping_df.set_index('alphabet')['label encoded'].to_dict()
        # print("label_to_encoded:\n", self.label_to_encoded)
        
        logging.info("TCA object initialized successfully")


    def compute_substitution_cost_matrix(self, method='constant', custom_costs=None):
        """
        Crée une matrice de substitution pour les séquences données.

        Parameters
        ----------
        sequences : pandas.Series
            Une série de séquences de caractères.
        method : str, optional
            La méthode utilisée pour calculer les coûts de substitution. Les options disponibles sont 'constant', 'custom' et 'frequency'.
            Par défaut, 'constant' est utilisé.
        custom_costs : dict, optional
            Un dictionnaire contenant les coûts de substitution personnalisés pour chaque paire de caractères.
            Requis si la méthode est définie sur 'custom'.

        """
        num_states = len(self.alphabet)

        # Initialiser la matrice de substitution
        substitution_matrix = np.zeros((num_states, num_states))

        if method == 'constant':
            # Calcul des coûts de substitution par défaut : coût de 2 pour toutes les substitutions
            for i in range(num_states):
                for j in range(num_states):
                    if i != j:
                        substitution_matrix[i, j] = 2

        elif method == 'custom':
            # Calcul des coûts de substitution personnalisés
            for i in range(num_states):
                for j in range(num_states):
                    if i != j:
                        state_i = self.alphabet[i]
                        state_j = self.alphabet[j]
                        try:
                            key = state_i + ':' + state_j
                            cost = custom_costs[key]
                        except:
                            key = state_j + ':' + state_i  
                            cost = custom_costs[key]
                        substitution_matrix[i, j] = cost

        elif method == 'frequency':
            # Calcul des coûts de substitution basés sur la fréquence des substitutions
            substitution_frequencies = np.zeros((num_states, num_states))

            for sequence in self.sequences:
                sequence = [char if char != 'nan' else '-' for char in sequence.split('-')]
                for i in range(len(sequence) - 1):
                    state_i = self.alphabet.index(sequence[i])
                    state_j = self.alphabet.index(sequence[i + 1])
                    substitution_frequencies[state_i, state_j] += 1

            substitution_probabilities = substitution_frequencies / substitution_frequencies.sum(axis=1, keepdims=True)

            for i in range(num_states):
                for j in range(num_states):
                    if i != j:
                        substitution_matrix[i, j] = 2 - substitution_probabilities[i, j] - substitution_probabilities[j, i]

        substitution_cost_matrix = pd.DataFrame(substitution_matrix, index=self.alphabet, columns=self.alphabet)

        return substitution_cost_matrix


    def optimal_matching(self, seq1, seq2, substitution_cost_matrix, indel_cost=None):
        if indel_cost is None:
            indel_cost = max(substitution_cost_matrix.values.flatten()) / 2
        m, n = len(seq1), len(seq2)
        score_matrix = np.zeros((m+1, n+1))
        
        # Initialisation de la matrice de scores
        score_matrix[:, 0] = indel_cost * np.arange(m+1)
        score_matrix[0, :] = indel_cost * np.arange(n+1)

        for i in range(1, m+1):
            for j in range(1, n+1):
                cost_substitute = substitution_cost_matrix.iloc[self.alphabet.index(seq1[i - 1]), self.alphabet.index(seq2[j - 1])]
                match = score_matrix[i-1, j-1] + cost_substitute
                delete = score_matrix[i-1, j] + indel_cost
                insert = score_matrix[i, j-1] + indel_cost
                score_matrix[i, j] = min(match, delete, insert)

        # Score d'alignement optimal
        optimal_score = score_matrix[m, n]

        return optimal_score


    def replace_labels(self, sequence, label_to_encoded):
        vectorized_replace = np.vectorize(label_to_encoded.get)
        return vectorized_replace(sequence)


    def compute_distance_matrix(self, metric='hamming', substitution_cost_matrix=None):
        """
        Calculate the distance matrix for the treatment sequences.

        Parameters:
        metric (str): The distance metric to use. Default is 'hamming'.
            -> 'hamming': The Hamming distance is used to calculate the pairwise distances between sequences.
                          It is the proportion of positions at which the corresponding elements are different.
                          Sequences must have the same length.
            -> 'levenshtein': The Levenshtein distance is used to calculate the pairwise distances between sequences.
                              It is the minimum number of single-character edits required to change one sequence into the other.
                              Sequences can have different lengths.
                           

        Returns:
        distance_matrix (numpy.ndarray): A condensed distance matrix containing the pairwise distances between treatment sequences.
        """
        logging.info(f"Calculating distance matrix using metric: {metric}...")
        start_time = timeit.default_timer()

        if metric == 'hamming':
            self.distance_matrix = np.array(pdist(self.data.replace(self.label_to_encoded), metric=metric))

        elif metric == 'levenshtein':
            self.distance_matrix = np.zeros((len(self.data), len(self.data)))
            # self.distance_matrix = pdist(self.data, lambda x, y: Levenshtein.distance(x, y))
            for i in tqdm.tqdm(range(len(self.sequences))):
                for j in range(i + 1, len(self.sequences)):
                    seq1, seq2 = self.sequences[i], self.sequences[j]                  
                    distance = Levenshtein.distance(seq1, seq2)
                    self.distance_matrix[i, j] = distance
                    self.distance_matrix[j, i] = distance  # Symmetric matrix

        elif metric == 'optimal_matching':
            if substitution_cost_matrix is None:
                logging.error("Substitution cost matrix not found. Please compute the substitution cost matrix first.")
                raise ValueError("Substitution cost matrix not found. Please compute the substitution cost matrix first.")
            self.distance_matrix = np.zeros((len(self.data), len(self.data)))
            print("substitution cost matrix: \n", substitution_cost_matrix)
            print("indel cost: ", max(substitution_cost_matrix.values.flatten()) / 2)
            
            # self.distance_matrix = np.array(pdist(self.data.drop(self.id, axis=1), metric= lambda x, y: self.optimal_matching(x, y, substitution_cost_matrix)/max(len(x), len(y))))
            
            for i in tqdm.tqdm(range(len(self.sequences))):
                for j in range(i + 1, len(self.sequences)):
                    seq1, seq2 = self.sequences[i], self.sequences[j]
                    distance = self.optimal_matching(seq1, seq2, substitution_cost_matrix)
                    max_length = max(len(seq1), len(seq2))
                    normalized_dist = distance / max_length
                    self.distance_matrix[i, j] = normalized_dist
                    self.distance_matrix[j, i] = normalized_dist  # Symmetric matrix

        elif metric == 'dtw':
            self.distance_matrix = np.zeros((len(self.data), len(self.data)))
            for i in tqdm.tqdm(range(len(self.sequences))):
                for j in range(i + 1, len(self.sequences)):
                    seq1, seq2 = self.replace_labels(self.sequences[i], self.label_to_encoded), self.replace_labels(self.sequences[j], self.label_to_encoded)
                    # print(seq1)
                    # print(seq2)
                    distance = dtw(seq1, seq2)
                    max_length = max(len(seq1), len(seq2))
                    normalized_dist = distance / max_length
                    self.distance_matrix[i, j] = normalized_dist
                    self.distance_matrix[j, i] = normalized_dist

        elif metric == 'dtw_path_from_metric':
            self.distance_matrix = np.zeros((len(self.data), len(self.data)))
            for i in tqdm.tqdm(range(len(self.sequences))):
                for j in range(i + 1, len(self.sequences)):
                    seq1, seq2 = self.replace_labels(self.sequences[i], self.label_to_encoded), self.replace_labels(self.sequences[j], self.label_to_encoded)
                    # print(seq1)
                    # print(seq2)
                    distance = dtw_path_from_metric(seq1, seq2, metric=np.abs(seq1 - seq2))
                    max_length = max(len(seq1), len(seq2))
                    normalized_dist = distance / max_length
                    self.distance_matrix[i, j] = normalized_dist
                    self.distance_matrix[j, i] = normalized_dist
        
        elif metric == 'gak':
            self.distance_matrix = np.zeros((len(self.data), len(self.data)))
            for i in tqdm.tqdm(range(len(self.sequences))):
                for j in range(i + 1, len(self.sequences)):
                    seq1, seq2 = self.replace_labels(self.sequences[i], self.label_to_encoded), self.replace_labels(self.sequences[j], self.label_to_encoded)
                    # print(seq1)
                    # print(seq2)
                    distance = gak(seq1, seq2)
                    max_length = max(len(seq1), len(seq2))
                    normalized_dist = distance / max_length
                    self.distance_matrix[i, j] = normalized_dist
                    self.distance_matrix[j, i] = normalized_dist

        c_time = timeit.default_timer() - start_time
        logging.info(f"Time taken for computation: {c_time:.2f} seconds")

        # Distance matrix must be symetric
        assert(np.allclose(self.distance_matrix, self.distance_matrix.T)), "Distance matrix is not symmetric"


        return self.distance_matrix
    
    def hierarchical_clustering(self, distance_matrix, method='ward', optimal_ordering=True):
        """
        Perform hierarchical clustering on the distance matrix.

        Parameters:
        distance_matrix (numpy.ndarray): A condensed distance matrix containing the pairwise distances between treatment sequences.
        method (str): The linkage algorithm to use. Default is 'ward'.
        optimal_ordering (bool): If True, the linkage matrix will be reordered so that the distance between successive leaves is minimal.

        Returns:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.
        """
        logging.info(f"Computing the linkage matrix using method: {method}...")

        # Convert the distance matrix to a condensed distance matrix
        condensed_distance_matrix = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distance_matrix, method=method, optimal_ordering=optimal_ordering)
        logging.info("Linkage matrix computed successfully")

        # cophenetic correlation coefficient for a given hierarchical clustering
        # measures how faithfully the hierarchical clustering preserves the pairwise distances between the original observations
        # a value of c close to 1 indicates a good fit between the clustering and the original distances.
        # c, _ = cophenet(linkage_matrix, condensed_distance_matrix)
        # logging.info(f"Cophenetic correlation coefficient: {c:.2f}")

        self.leaf_order = leaves_list(linkage_matrix)

        return linkage_matrix

    def assign_clusters(self, linkage_matrix, num_clusters):
        """
        Assign patients to clusters based on the dendrogram.

        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.
        num_clusters (int): The number of clusters to form.

        Returns:
        numpy.ndarray: An array of cluster labels assigned to each patient.
        """
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        return clusters



####################################### Plot methods #######################################


    def plot_treatment_percentages(self, clusters=None ):
        """
        Plot the percentage of patients under each state over time.
        If clusters are provided, plot the treatment percentages for each cluster.


        Returns:
        None
         
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("self.data should be a pandas DataFrame")
        if clusters is None:
           
            df = self.data.copy()
            # Initialize an empty list to store data for plotting
            plot_data = []

            # Collect data for each treatment
            for treatment, treatment_label,color in zip(self.state_numeric, self.state_label,self.colors):
                treatment_data = df[df.eq(treatment).any(axis=1)]
                months = treatment_data.columns
                percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                plot_data.append(pd.DataFrame({'Month': months, 'Percentage': percentages, 'Treatment': treatment_label}))
                plt.plot(months, percentages, label=f'{treatment_label}', color=color)

            plt.title('Percentage of Patients under Each State Over Time')
            plt.xlabel('Time')
            plt.ylabel('Percentage of Patients')
            plt.legend(title='State')
            plt.show()

        else :
            num_clusters = len(np.unique(clusters))
            colors = self.colors
            events_value = self.state_numeric
            events_keys = self.state_label
            num_rows = (num_clusters + 1) // 2
            num_cols = min(2, num_clusters)

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            if num_clusters == 2:
                axs = np.array([axs])
            if num_clusters % 2 != 0:
                fig.delaxes(axs[-1, -1])

            for cluster_label in range(1, num_clusters + 1):
                cluster_indices = np.where(clusters == cluster_label)[0]
                cluster_data = self.data.iloc[cluster_indices]

                row = (cluster_label - 1) // num_cols
                col = (cluster_label - 1) % num_cols

                ax = axs[row, col]

                for treatment, treatment_label, color in zip(events_value, events_keys, colors):
                    treatment_data = cluster_data[cluster_data.eq(treatment).any(axis=1)]
                    months = treatment_data.columns
                    percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                    ax.plot(months, percentages, label=f'{treatment_label}', color=color)
                
                ax.set_title(f'Cluster {cluster_label}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Percentage of Patients')
                ax.legend(title='State')
            
            plt.tight_layout()
            plt.show()

    def plot_dendrogram(self, linkage_matrix):
        """
        Plot a dendrogram based on the hierarchical clustering of treatment sequences.

        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.

        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix)
        plt.title('Dendrogram of Treatment Sequences')
        plt.xlabel('Patients')
        plt.ylabel('Distance')
        plt.show()

    def plot_clustermap(self, linkage_matrix):
        """
        Plot a clustermap of the treatment sequences with a custom legend.

        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.

        Returns:
        None
        """
        plt.figure(figsize=(8, 8))
        sns.clustermap(self.data.drop(self.id, axis=1).replace(self.label_to_encoded),
                       cmap=self.colors,
                       metric='precomputed',
                       method='ward',
                       row_linkage=linkage_matrix,
                       row_cluster=True, 
                       col_cluster=False,
                       dendrogram_ratio=(.1, .2),
                       cbar_pos=None)
        
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.yticks([])
        plt.title("Clustermap of Treatment Sequences")

        # convert viridis colors as a list
        viridis_colors_list = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(self.alphabet))]

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=viridis_colors_list[i], label=self.alphabet[i]) for i in range(len(self.alphabet))]
        plt.legend(handles=legend_handles, labels=self.states, loc='upper right', ncol=1, title='Statuts')

        # plt.tight_layout()
        plt.show()

    # def get_clustermap_function(self):
        

    def plot_inertia(self, linkage_matrix):
        """
        Plot the inertia diagram to help determine the optimal number of clusters.

        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.

        Returns:
        None
        """
        last = linkage_matrix[-10:, 2]
        last_rev = last[::-1]
        idxs = np.arange(2, len(last) + 2)

        plt.figure(figsize=(10, 6))
        plt.step(idxs, last_rev, c="black")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title("Inertia Diagram")
        plt.show()
    
    def plot_cluster_heatmaps(self, clusters, sorted=True):
        """
        Plot heatmaps for each cluster, ensuring the data is sorted by leaves_order.

        Parameters:
        clusters (numpy.ndarray): The cluster assignments for each patient.
        leaves_order (list): The order of leaves from the hierarchical clustering.
        sorted (bool): Whether to sort the data within each cluster. Default is True.

        Returns:
        None
        """
        # Reorder the data according to leaves_order
        leaves_order = self.leaf_order
        reordered_data = self.data.iloc[leaves_order]
        reordered_clusters = clusters[leaves_order]

        num_clusters = len(np.unique(clusters))
        cluster_data = {}
        for cluster_label in range(1, num_clusters + 1):
            cluster_indices = np.where(reordered_clusters == cluster_label)[0]
            cluster_df = reordered_data.iloc[cluster_indices]
            if sorted:
                cluster_df = cluster_df.sort_values(by=cluster_df.columns.tolist())
            cluster_data[cluster_label] = cluster_df

        # Determine the size of each cluster
        heights = [len(cluster_df)*0.2 for cluster_df in cluster_data.values()]
        print("heights: ", heights)

        num_rows = num_clusters
        num_cols = min(1, num_clusters)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, sum(heights)* 0.01), sharex=True, gridspec_kw={'height_ratios': heights})
        
        if num_clusters == 2:
            axs = np.array([axs])
        if num_clusters % 2 != 0:
            fig.delaxes(axs[-1, -1])            

        # print(cluster_data.items())
        
        for cluster_label, (cluster_df, ax) in enumerate(zip(cluster_data.items(), axs)):
            # print(cluster_df[1])
            # print(cluster_df)
            # print(ax)
            sns.heatmap(cluster_df[1].drop(self.id, axis=1).replace(self.label_to_encoded), cmap=self.colors, cbar=False, ax=ax, yticklabels=False)
            ax.tick_params(axis='x', rotation=45)
            ax.text(1.05, 0.5, f'cluster {cluster_label} (n={len(cluster_df[1])})', transform=ax.transAxes, ha='left', va='center')
            # axs[row].set_ylabel('Patients id')
        axs[-1].set_xlabel('Time in months')

        # for i, (cluster_label, cluster_df) in enumerate(cluster_data.items()):
        #     row = i // num_cols
        #     # col = i % num_cols
        #     height = len(cluster_df) / 2  # Adjust the height based on the number of patients
        #     fig, ax = plt.subplots(figsize=(15, height))
        #     sns.heatmap(cluster_df.drop(self.id, axis=1).replace(self.label_to_encoded), cmap=self.colors, cbar=False, ax=ax, yticklabels=False)
        #     ax.tick_params(axis='x', rotation=45)
        #     ax.text(1.05, 0.5, f'cluster {cluster_label} (n={len(cluster_df)})', transform=ax.transAxes, ha='left', va='center')
        #     ax.tick_params(axis='y', size=height)
        #     ax.set_xlabel('Time in months')

        # convert viridis colors as a list
        viridis_colors_list = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(self.alphabet))]

        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=viridis_colors_list[i], label=self.alphabet[i]) for i in range(len(self.alphabet))]
        plt.legend(handles=legend_handles, labels=self.states, loc='lower right', ncol=1, title='Statuts')

        plt.tight_layout()
        plt.show()

    def bar_treatment_percentage(self, clusters=None):
        """
        Plot the percentage of patients under each state over time using bar plots.
        If clusters are provided, plot the treatment percentages for each cluster.

        Parameters:
        clusters (numpy.ndarray): Cluster assignments for each patient (optional).

        Returns:
        None
        """
        if clusters is None:
            df = self.data.copy()
            # Initialize an empty list to store data for plotting
            plot_data = []

            # Collect data for each treatment
            for treatment, treatment_label,color in zip(self.state_numeric, self.state_label,self.colors):
                treatment_data = df[df.eq(treatment).any(axis=1)]
                months = treatment_data.columns
                percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                plot_data.append(pd.DataFrame({'Month': months, 'Percentage': percentages, 'Treatment': treatment_label}))
                plt.bar(months, percentages, label=f'{treatment_label}', color=color)

            plt.title('Percentage of Patients under Each State Over Time')
            plt.xlabel('Time')
            plt.ylabel('Percentage of Patients')
            plt.legend(title='State')
            plt.show()

        else:
            num_clusters = len(np.unique(clusters))
            num_rows = (num_clusters + 1) // 2  
            num_cols = min(2, num_clusters)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            if num_clusters == 2:
                axs = np.array([axs])
            if num_clusters % 2 != 0:
                fig.delaxes(axs[-1, -1])

            for cluster_label in range(1, num_clusters + 1):
                cluster_indices = np.where(clusters == cluster_label)[0]
                cluster_data = self.data.iloc[cluster_indices]

                row = (cluster_label - 1) // num_cols
                col = (cluster_label - 1) % num_cols

                ax = axs[row, col]

                for treatment, treatment_label, color in zip(self.state_numeric, self.state_label, self.colors):
                    treatment_data = cluster_data[cluster_data.eq(treatment).any(axis=1)]
                    months = treatment_data.columns
                    percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                    ax.bar(months, percentages, label=f'{treatment_label}', color=color)

                ax.set_title(f'Cluster {cluster_label}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Percentage of Patients')
                ax.legend(title='State')

            plt.tight_layout()
            plt.show()

    def plot_stacked_bar(self, clusters):
        """
        Plot stacked bar charts showing the percentage of patients under each treatment over time for each cluster.

        Parameters:
        clusters (numpy.ndarray): The cluster assignments for each patient.

        Returns:
        None
        """
        num_clusters = len(np.unique(clusters))
        num_rows = (num_clusters + 1) // 2  
        num_cols = min(2, num_clusters)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        if num_clusters == 2:
            axs = np.array([axs])
        if num_clusters % 2 != 0:
            fig.delaxes(axs[-1, -1])

        for cluster_label in range(1, num_clusters + 1):
            cluster_indices = np.where(clusters == cluster_label)[0]
            cluster_data = self.data.iloc[cluster_indices]
            
            row = (cluster_label - 1) // num_cols
            col = (cluster_label - 1) % num_cols
            
            ax = axs[row, col]
            
            stacked_data = []
            for treatment in self.state_numeric:
                treatment_data = cluster_data[cluster_data.eq(treatment).any(axis=1)]
                months = treatment_data.columns
                percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                stacked_data.append(percentages.values)
            
            months = range(len(months))
            bottom = np.zeros(len(months))
            for i, data in enumerate(stacked_data):
                ax.bar(months, data, bottom=bottom, label=self.state_label[i], color=self.colors[i])
                bottom += data
            
            ax.set_title(f'Cluster {cluster_label}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Percentage of Patients')
            ax.legend(title='Treatment')
        
        plt.tight_layout()
        plt.show()


####################################### MAIN #######################################
####################################### MAIN #######################################
####################################### MAIN #######################################
####################################### MAIN #######################################

def main():
    df = pd.read_csv('data/dataframe_test.csv')

    # tranformer vos données en format large si c'est n'est pas le cas 
    # state_mapping = {"EM": 2, "FE": 4, "HE": 6, "JL": 8, "SC": 10, "TR": 12}
    # colors = ['blue', 'orange', 'green', 'red', 'yellow', 'gray']
    # df_numeriques = df.replace(state_mapping)
    # print(df_numeriques.head())
    # print(df_numeriques.columns)
    # print(df_numeriques.shape)
    # print(df_numeriques.info())
    # print(df_numeriques.isnull().sum())
    # print(df_numeriques.describe())
    # print(df_numeriques.dtypes) 

    # Sélectionner les colonnes pertinentes pour l'analyse
    selected_cols = df[['id', 'month', 'care_status']]

    # Créer un tableau croisé des données en format large
    #       -> Chaque individu est sur une ligne.
    #       -> Les mesures dans le temps (Temps1, Temps2, Temps3) sont des colonnes distinctes.
    pivoted_data = selected_cols.pivot(index='id', columns='month', values='care_status')
    pivoted_data['id'] = pivoted_data.index
    pivoted_data = pivoted_data[['id'] + [col for col in pivoted_data.columns if col != 'id']]

    # Renommer les colonnes avec un préfixe "month_"
    pivoted_data.columns = ['id'] + ['month_' + str(int(col)+1) for col in pivoted_data.columns[1:]]
    # print(pivoted_data.columns)

    # Sélectionner un échantillon aléatoire de 10% des données
    pivoted_data_random_sample = pivoted_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Filter individuals observed for at least 18 months
    valid_18months_individuals = pivoted_data.dropna(thresh=19).reset_index(drop=True)

    # Select only the first 18 months for analysis
    valid_18months_individuals = valid_18months_individuals[['id'] + [f'month_{i}' for i in range(1, 19)]]

    # print(pivoted_data_random_sample.head())
    # print(data_ready_for_TCA.duplicated().sum())

    # tca = TCA(df_numeriques,state_mapping,colors)
    tca = TCA(data=pivoted_data_random_sample,
              id='id',
              alphabet=['D', 'C', 'T', 'S'],
              states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"])
   
    # tca.plot_treatment_percentages(df_numeriques)

    custom_costs = {'D:C': 1, 'D:T': 2, 'D:S': 3, 'C:T': 1, 'C:S': 2, 'T:S': 1}
    costs = tca.compute_substitution_cost_matrix(method='custom', custom_costs=custom_costs)
    distance_matrix = tca.compute_distance_matrix(metric='dtw', substitution_cost_matrix=costs)
    print("distance matrix :\n",distance_matrix)

    linkage_matrix = tca.hierarchical_clustering(distance_matrix)

    # tca.plot_dendrogram(linkage_matrix)
    tca.plot_clustermap(linkage_matrix)
    # tca.plot_inertia(linkage_matrix)

    clusters = tca.assign_clusters(linkage_matrix, num_clusters=4)
    
    tca.plot_cluster_heatmaps(clusters, sorted=False)
    # tca.plot_cluster_treatment_percentage(clusters)
    # tca.bar_cluster_treatment_percentage(clusters)
    # tca.plot_stacked_bar(clusters)

if __name__ == "__main__":
    main()