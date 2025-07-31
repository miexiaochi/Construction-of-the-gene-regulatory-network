import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from arboreto.utils import load_tf_names
from joblib import Parallel, delayed  

def add_stage_info(expression_data: pd.DataFrame):
    """
    Adds stage information (GV and MII) to the data.
    :param expression_data: Gene expression data
    :return: DataFrame with a new "Stage" column
    """
    # Assuming the first three rows are GV stage, and the last three rows are MII stage
    stages = ['GV'] * 3 + ['MII'] * 3  # Here we assume first 3 samples are GV and last 3 are MII
    expression_data['Stage'] = stages * (expression_data.shape[0] // 6)  # Repeating the stages every 6 samples
    return expression_data

def process_target_gene(target_gene, expression_data):
    """
    Performs regression analysis for each target gene and computes feature importance, returning regulatory relationships.
    :param target_gene: The current target gene
    :param expression_data: Gene expression DataFrame
    :return: Regulatory relationships between the target gene and other genes
    """
    grn_edges = []
    # Extract target gene expression values
    y = expression_data[target_gene].values

    # Use other genes as features, including "Stage" as an additional feature
    X = expression_data.drop(columns=[target_gene, 'Stage']).values  # Dropping target gene and "Stage" column

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train using Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predict and compute error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Get feature importance for each gene
    feature_importances = model.feature_importances_

    # Determine the weights for regulatory edges based on feature importance
    for i, importance in enumerate(feature_importances):
        if importance > 0:
            source_gene = expression_data.columns[i]
            grn_edges.append((source_gene, target_gene, importance))

    print(f"Processed {target_gene}, MSE: {mse}")
    return grn_edges

def grn_inference_parallel(expression_data: pd.DataFrame, tf_names: list):
    """
    Performs parallel computation for Gene Regulatory Network (GRN) inference.
    :param expression_data: Gene expression data
    :param tf_names: List of transcription factor gene names
    :return: GRN containing source gene, target gene, and regulatory weight
    """
    # Parallel computation using joblib's Parallel and delayed
    grn_edges_list = Parallel(n_jobs=-1)(  # Using all available cores
        delayed(process_target_gene)(target_gene, expression_data) 
        for target_gene in expression_data.columns if target_gene != 'Stage'
    )

    # Combine regulatory relationships from all gene pairs
    grn_edges = [edge for sublist in grn_edges_list for edge in sublist]

    # Convert result into a DataFrame
    grn_df = pd.DataFrame(grn_edges, columns=["Source", "Target", "Weight"])
    return grn_df

if __name__ == "__main__":
    # Assuming expression_data is the gene expression data with genes as columns and samples as rows
    in_file = 'rpkm.txt'  # Example input data
    expression_data = pd.read_csv(in_file, sep='\t', header=0, index_col=0)

    # Add GV and MII stage information
    expression_data = add_stage_info(expression_data)

    # Assuming tf_names is a list of transcription factors
    tf_file = 'tf.txt'  # Example transcription factor file
    tf_names = load_tf_names(tf_file)

    # Call the parallel inference function
    grn = grn_inference_parallel(expression_data, tf_names)

    # Save the Gene Regulatory Network
    out_file = 'grn_output_parallel_with_stage.tsv'
    grn.to_csv(out_file, sep='\t', index=False)
    print(f"GRN inference completed, saved to {out_file}")
