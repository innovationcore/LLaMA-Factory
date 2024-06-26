# ClearML

# Infrastrcture 

## Training
### Server: 10.33.31.21
### Container: llmfactory-clearml  
### Launch script: /mnt/scratch/cody/llm/run_llmfactory_clearml.sh

## Inference
### Server: 10.33.31.21
### Container: ghcr.io/predibase/lorax  
### Launch script: /mnt/scratch/cody/llm/run_lorax_background.sh

## Adapters
### Server: 10.33.31.21
### Adapter repo: /mnt/scratch/cody/llm/models/adapters

# Process 

## (1) clearml.conf must be placed in ~/clearml.conf

```json
api { 
    web_server: https://clearml.ai.uky.edu
    api_server: https://clearml.ai.uky.edu:8008
    # llm_trainer
    credentials {
        "access_key" = <ACCESS KEY>
        "secret_key"  = <SECRET KEY>
    }
}
sdk {
    aws {
        s3 {
            credentials: [
                {
                  host: "10.33.31.21:9000" # url has no http:// or s3://, only the domain
                  bucket: "llmadapters" # Does it have importance?
                  key: <ACCESS KEY>
                  secret: <SECRET KEY>
                  multipart: false
                  secure: false
                },
                {
                  host: "10.33.31.21:9000" # url has no http:// or s3://, only the domain
                  bucket: "datasets" # Does it have importance?
                  key: <ACCESS KEY>
                  secret: <SECRET KEY>
                  multipart: false
                  secure: false
                }
            ]
        }
    }
    development.default_output_uri: "s3://10.33.31.21:9000/llmadapters" # This one works for task artifacts!
}
```

## (2) Upload custom dataset to S3 and register with ClearML

```
python clearml_dataset_manager.py 
```
- Uploads data/example_custom_dataset to S3 bucket: 'datasets' prefix: 'custom_dataset'
- Registers dataset contained in S3 bucket: 'datasets' prefix: 'custom_dataset' as a ClearML dataset, 'custom_dataset'

## (3) Generate training template, which will make use of the dataset registered in (2)

```
python clearml_training_wrapper.py
```
- Downloads custom dataset to pre-determined location
- Launches training script 
- Reports training stats
- Uploads resulting Lora adapter

## (4) Launch new training from template, with configuration changes
```
python clearml_train_launcher.py
```
- Clones template training task
- Modifies training configuration
- Launches new training task via ClearML

## (5) Pull down trained Lora adapters
```
python adapter_downloader.py
```
- Scans S3 bucket with adapters
- Download any new adapter
- Save adapters to location where they can be used by LoraX dynamically

## (6) Start Lorax server with specific base model

```
docker run --gpus all -d --shm-size 1g -p 8080:80 -v <path to data>:/data ghcr.io/predibase/lorax:latest --model-id $model
```

## (7) Reference specific Lora adapter on api request

- adapter_id: The path of the adapter local to the inference server
- adapter_source: use local, we need to get S3 to work

```
curl 10.33.31.21:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Is it better to burn out, or fade away? [/INST]",
        "parameters": {
        	   "best_of": 2,
            "do_sample": true,
            "max_new_tokens": 75,
            "adapter_id": "/data/adapters/llm_factory_trainer-3958082d-117e-4fff-9293-23d0c6250d77",
            "adapter_source":"local"
        }
    }' \
    -H 'Content-Type: application/json'    
```
