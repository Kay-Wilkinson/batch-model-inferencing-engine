# batch-model-inferencing-engine

A batch inference engine for Hugging Face transformer models (causal LMs and classifiers), 
supporting CSV-based datasets, configurable batching, and GPU-accelerated execution. 
The tool reports throughput, latency and resource usage, and is designed as reusable infrastructure for 
large-scale experiment runs and behaviour analysis.



Noe that the POC does deviate from some expected architectural design choices such as workload orchestration,
decoupled components, pub/sub etc. 


# Design Notes:

## Engine Behaviour:
Input: This will be a dataset of text prompts (script out synthetic data generation)
Engine: 
    * Loads a HuggingFace model + tokeniser
    * Splits data into batches
    * Runs inference on batches (on GPU if available, default to CPU if not)
    * Collect outputs
    * Measures performance
Output:
    * CSV with inputs and outputs
    * Printed metrics at the end
Nice to haves:
    * Switchable task type: generation vs classification
    * Configurable batch size, model name, max_new_tokens
    * Retry behaviour, configurable back-off attempts

## Architecture
The POC is just that...so this is kept simple to three layers:
1. Application Layer (an entrypoint script that parses arguments, constructs config, initialises the engine and writes results)
2. Engine Layer (Handle batching, call a runner for each Batch, tracks metrics, handles retries)
3. Model Layer (Task specific runners - text generation, sequence classification)


