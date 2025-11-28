from batch_infer.config import InferenceConfig
from batch_infer.engine.runners.core_runners import BatchInferenceRunner
from batch_infer.io import read_texts_from_csv, write_results_to_csv
from batch_infer.models.model_runners import CausalLMRunner


def main():
    cfg = InferenceConfig(
        model_name="distilgpt2",
        task="generation",
        batch_size=8,
        max_new_tokens=32,
        device="cuda",  # or detect automatically
        input_column="text",
    )

    model_runner = CausalLMRunner(cfg)
    engine = BatchInferenceRunner(cfg, model_runner)

    texts = read_texts_from_csv("data/sample_input.csv", col=cfg.input_column)
    results = engine.run(texts)

    write_results_to_csv("data/output.csv", results)


if __name__ == "__main__":
    main()
