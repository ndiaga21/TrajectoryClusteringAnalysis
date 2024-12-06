
#include <stdio.h>
#include <stdlib.h>

double optimal_matching(const char* seq1, const char* seq2, double* substitution_cost_matrix, int num_elements, double indel_cost) {
    int m = strlen(seq1);
    int n = strlen(seq2);
    double* score_matrix = (double*)malloc((m + 1) * (n + 1) * sizeof(double));
    
    for (int i = 0; i <= m; i++) {
        score_matrix[i * (n + 1)] = i * indel_cost;
    }
    for (int j = 0; j <= n; j++) {
        score_matrix[j] = j * indel_cost;
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int idx1 = strchr(seq1, seq1[i - 1]) - seq1;
            int idx2 = strchr(seq2, seq2[j - 1]) - seq2;
            double cost_substitute = substitution_cost_matrix[idx1 * num_elements + idx2];
            double match = score_matrix[(i - 1) * (n + 1) + (j - 1)] + cost_substitute;
            double delete = score_matrix[(i - 1) * (n + 1) + j] + indel_cost;
            double insert = score_matrix[i * (n + 1) + (j - 1)] + indel_cost;
            score_matrix[i * (n + 1) + j] = fmin(fmin(match, delete), insert);
        }
    }

    double optimal_score = score_matrix[m * (n + 1) + n];
    free(score_matrix);
    return optimal_score;
}
