import boto3
import pandas as pd
import io
import json
import logging
from botocore.exceptions import ClientError

# logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 and SageMaker configuration
s3_bucket = '100csvs'
input_prefix = '100csv/'
output_prefix = 'outputs/'
sagemaker_endpoint = 'canvas-prediction'

# Initializin clients
s3 = boto3.client('s3')
sagemaker_runtime = boto3.client('runtime.sagemaker', region_name='us-east-1')

def invoke_sagemaker_endpoint(input_data):
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=sagemaker_endpoint,
            ContentType="text/csv",
            Body=input_data,
            Accept="application/json"
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except ClientError as e:
        logger.error(f"Error invoking SageMaker endpoint: {e}")
        raise

def process_predictions(predictions, df):
    if not isinstance(predictions, list) or len(predictions) != len(df):
        logger.error(f"Unexpected predictions format. Expected a list of length {len(df)}, got: {type(predictions)}")
        return df

    for i, pred in enumerate(predictions):
        if isinstance(pred, dict):
            for key, value in pred.items():
                df.at[i, key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        else:
            logger.warning(f"Unexpected prediction format at index {i}: {pred}")

    return df

def process_csv_file(file_key):
    logger.info(f"Processing file: {file_key}")
    
    try:
        # Reading CSV file from S3
        response = s3.get_object(Bucket=s3_bucket, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        
        # Prepare input data
        df = pd.read_csv(io.StringIO(file_content))
        input_data = df.to_csv(index=False, header=False)
        
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"First few rows of input data:\n{df.head()}")
        
        # Get predictions
        prediction_response = invoke_sagemaker_endpoint(input_data)
        
        logger.info(f"Received prediction. Structure: {prediction_response.keys()}")
        
        # Process predictions and combine with input features
        if 'predictions' in prediction_response:
            df = process_predictions(prediction_response['predictions'], df)
        else:
            logger.error("No 'predictions' key in the response")
            return
        
        # Saving the result back to S3
        output_key = output_prefix + file_key.split('/')[-1]
        output_buffer = io.StringIO()
        df.to_csv(output_buffer, index=False)
        s3.put_object(Bucket=s3_bucket, Key=output_key, Body=output_buffer.getvalue())
        
        logger.info(f"Processed and saved result for {file_key}")
    except Exception as e:
        logger.error(f"Error processing file {file_key}: {str(e)}")

def process_all_csv_files():
    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=input_prefix)
        csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found in {s3_bucket}/{input_prefix}")
            return
        
        for csv_file in csv_files:
            process_csv_file(csv_file)
    except Exception as e:
        logger.error(f"Error listing CSV files: {str(e)}")

# Start processing all CSVs :)
if __name__ == "__main__":
    process_all_csv_files()