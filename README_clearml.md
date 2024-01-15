# ClearML

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


