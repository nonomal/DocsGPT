# How to deploy LLM's on Sagemaker for DocsGPT

This guide uses some of the methods from the [Phil Schmid's guides](https://www.philschmid.de/) so if you want to dive deeper into the topic, check out his guides.

### 1. Create a new python notebook on Sagemaker and prep dependencies and permissions

Install dependencies

```python
!pip install "sagemaker>=2.175.0" --upgrade --quiet
```

Check permissions

```python
import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")
print(f"sagemaker session bucket: {sess.default_bucket()}")
```

Get huggingfaces llm image uri for the container

```python
from sagemaker.huggingface import get_huggingface_llm_image_uri

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  "huggingface",
  version="1.1.0",
)

# print ecr image uri
print(f"llm image uri: {llm_image}")
```

### 2. Prepare the Model

Running this code will create a model with some default parameters. You can change these parameters to suit your needs.
There are two ways you can choose which model to use.

You can either use the model_id from the huggingface.co/models page or you can use the model_data from a previous training job.

```python
import json
from sagemaker.huggingface import HuggingFaceModel

# sagemaker config
instance_type = "ml.g5.xlarge"
number_of_gpu = 1
health_check_timeout = 600

# Define Model and Endpoint configuration parameter
config = {
  'HF_MODEL_ID': "/opt/ml/model", # model_id from hf.co/models
  #'HF_MODEL_ID': "Arc53/DocsGPT-7B",  
  'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(7000),  # Max length of input text
  'MAX_TOTAL_TOKENS': json.dumps(8000),  # Max length of the generation (including input text)
  'MAX_BATCH_TOTAL_TOKENS': json.dumps(8192),  # Limits the number of tokens that can be processed in parallel during the generation
  'MAX_BATCH_PREFILL_TOKENS': json.dumps(7000),
}

# create HuggingFaceModel with the image uri
llm_model = HuggingFaceModel(
  model_data="s3://docsgpt/models/hf-tensors/docsgpt-7b-O-hq-64-alpha-2023-11-22-15-04-04-455/model.tar.gz",
  role=role,
  image_uri=llm_image,
  env=config
)
```

### 3. Deploy the Model

Running this line will create Model in the Sagemaker console. Next it will create an endpoint configuration and finally it will create an endpoint.

```python
llm = llm_model.deploy(
  initial_instance_count=1,
  endpoint_name="docsgpt-7b",
  instance_type=instance_type,
  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
)
```

### 4. Connect it to the application

Change you .env file and set the following variables:

```python
SAGEMAKER_ENDPOINT: str = None # SageMaker endpoint name (docsgpt-7b)
SAGEMAKER_REGION: str = None # SageMaker region name
SAGEMAKER_ACCESS_KEY: str = None # SageMaker access key
SAGEMAKER_SECRET_KEY: str = None # SageMaker secret key
```

> **_NOTE:_** If you are using the same AWS account for the application and SageMaker, you can leave the access and secret keys empty.

Also make sure you switch to appropriate embeddings if you want everything runs locally for example

```python
EMBEDDINGS_NAME=huggingface_sentence-transformers/all-mpnet-base-v2
```
