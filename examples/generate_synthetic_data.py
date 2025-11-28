from batch_infer.models.synthetic_data_generator import SyntheticDataGenerator, samples_to_csv_rows
import csv

samples = SyntheticDataGenerator.generate_sentiment(1000)
rows = samples_to_csv_rows(samples)

with open("data/sentiment_synth.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(rows)
