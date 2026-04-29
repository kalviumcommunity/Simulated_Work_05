import csv
from datetime import datetime

def log_experiment(model_name, metrics):
    with open("logs/experiment_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        
        writer.writerow([
            datetime.now(),
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            "experiment run"
        ])