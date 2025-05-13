from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Specify the path to the event file
event_file = "../yaib_logs/sinusoids/configs/imputation_models/optimal_hyper_params/SAITS/MCAR/sinusoids/config/2025-01-27T15-25-40/repetition_0/fold_0/lightning_logs/version_0/events.out.tfevents.1738009657.gpu101.22288.1"

# Load the event file
event_acc = EventAccumulator(event_file)
event_acc.Reload()

# List all tags (e.g., scalars, histograms, etc.)
print("Available tags:", event_acc.Tags())

# Access scalar metrics
if 'scalars' in event_acc.Tags():
    for tag in event_acc.Tags()['scalars']:
        scalars = event_acc.Scalars(tag)
        print(f"Tag: {tag}")
        for s in scalars:
            print(f"Step: {s.step}, Value: {s.value}")
