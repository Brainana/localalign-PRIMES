import pandas as pd

# Define file paths.
matrix1_file = "tm_score_matrix.csv"
matrix2_file = "pairwise_cosine_similarity.csv"
difference_file = "tm_score_diff.csv"

# Load the CSV files.
# We assume the CSV files have an index column (row labels).
df1 = pd.read_csv(matrix1_file, index_col=0)
df2 = pd.read_csv(matrix2_file, index_col=0)

# Optional: Check that the matrices have the same shape.
if df1.shape != df2.shape:
    raise ValueError("The two matrices do not have the same dimensions.")

# Calculate the difference (element-wise).
difference = df1 - df2

# Save the resulting difference matrix to a CSV file.
difference.to_csv(difference_file)

print(f"Difference matrix saved to {difference_file}")