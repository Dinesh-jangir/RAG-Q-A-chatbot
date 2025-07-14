import os
import pandas as pd


def load_and_clean(path):
    """
    Load CSV from path and convert all columns to string, filling missing values.
    """
    df = pd.read_csv(path)
    # Ensure all columns are strings and fill NaN with 'Unknown'
    df = df.astype(str).fillna('Unknown')
    return df


def docs_from_df(df):
    """
    Convert each DataFrame row into a single text document.
    """
    docs = []
    for _, row in df.iterrows():
        # ...existing code...
        text = (
            f"Loan_ID: {row['Loan_ID']}. Gender: {row['Gender']}. Married: {row['Married']}. "
            f"Dependents: {row['Dependents']}. Education: {row['Education']}. "
            f"Self_Employed: {row['Self_Employed']}. LoanAmount: {row['LoanAmount']}. "
            f"Loan_Amount_Term: {row['Loan_Amount_Term']}. Credit_History: {row['Credit_History']}. "
            f"Property_Area: {row['Property_Area']}. Status: {row['Loan_Status']}."
        )
# ...existing code...
        docs.append(text)
    return docs


if __name__ == '__main__':
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    csv_path = os.path.join(data_dir, 'Training_Dataset.csv')
    output_path = os.path.join(data_dir, 'docs.txt')
# ...existing code...

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load and clean data
    df = load_and_clean(csv_path)

    # Generate documents
    docs = docs_from_df(df)

    # Write out to docs.txt
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(docs))





        