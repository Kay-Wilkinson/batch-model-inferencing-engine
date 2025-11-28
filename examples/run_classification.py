from batch_infer.config import InferenceConfig, RetryConfig
from batch_infer.engine.runners.core_runners import BatchInferenceRunner
from batch_infer.models.model_runners import ClassifierRunner
from batch_infer.io import read_texts_from_csv, write_results_to_csv


def main() -> None:
    cfg = InferenceConfig(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        task="classification",
        batch_size=16,
        device="cuda",
        retry=RetryConfig(max_retries=3),
        input_column="text",
    )

    model_runner = ClassifierRunner(cfg)
    engine = BatchInferenceRunner(cfg, model_runner)

    texts = read_texts_from_csv("data/sentiment_synth.csv", col=cfg.input_column)
    results = engine.run(texts)

    write_results_to_csv("data/sentiment_preds.csv", results)


if __name__ == "__main__":
    main()
