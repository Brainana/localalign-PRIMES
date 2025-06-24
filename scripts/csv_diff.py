import pandas as pd

matrix1_file = "testing_data/testing_tm_score_matrix.csv"
matrix2_file = "pred_tm_score_matrix_transformer.csv"
difference_file = "tm_score_diff_transformer.csv"

df1 = pd.read_csv(matrix1_file, index_col=0)
df2 = pd.read_csv(matrix2_file, index_col=0)

if df1.shape != df2.shape:
    raise ValueError("The two matrices do not have the same dimensions.")

difference = df1 - df2

difference.to_csv(difference_file)

print("Difference saved.")
