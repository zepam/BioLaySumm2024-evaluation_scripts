def convert_parquet_to_json(input_file, output_file):
    import pandas as pd

    # Read the Parquet file
    df = pd.read_parquet(input_file)
    df.info()

    # Convert to JSON and save to the output file
    df.to_json(output_file, orient='records', lines=True)

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Convert Parquet file to JSON.")
    # parser.add_argument("input_file", help="Path to the input Parquet file.")
    # parser.add_argument("output_file", help="Path to the output JSON file.")

    # args = parser.parse_args()

    convert_parquet_to_json('train.parquet', 'elife_train.json')

    # display number of rows in the json file
    import pandas as pd
    df = pd.read_json('elife_train.json', lines=True)
    print(f"Number of rows in the JSON file: {len(df)}")