import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

def plot_sample_data(base_dir, dyn_df, num_samples=3):
    sample_stays = dyn_df['stay_id'].unique()[:num_samples]

    for stay_id in sample_stays:
        sample_data = dyn_df[dyn_df['stay_id'] == stay_id]
        plt.figure(figsize=(12, 6))
        plt.plot(sample_data['time'], sample_data['dyn1'], label='Feature 1')
        plt.plot(sample_data['time'], sample_data['dyn2'], label='Feature 2')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Stay ID {stay_id}')
        plt.legend()
        plt.grid(True)
        plt.savefig(base_dir / f'sample_plot_stay_{stay_id}.png')
        plt.close()

def create_sinusoid_data(num_samples=200, num_stays=500):
    # Create directory structure
    base_dir = Path("demo_data/simulated/sinusoids")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')  # 1-hour intervals

    # Initialize lists to hold the data
    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        amplitude = np.random.uniform(0.5, 2.0)  # Random amplitude between 0.5 and 2

        # Create sin and cos waves with the same amplitude
        sin_wave = amplitude * np.sin(np.linspace(0, 6 * np.pi, num_samples))
        cos_wave = amplitude * np.cos(np.linspace(0, 6 * np.pi, num_samples))

        # Append dynamic data
        for t, sin_val, cos_val in zip(time_intervals, sin_wave, cos_wave):
            dyn_data.append([stay_id, t, sin_val, cos_val])

        # Append static and outcome data
        static_data.append([stay_id, sin_val + cos_val])
        outcome_data.append([stay_id, amplitude])

    # Create DataFrames
    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])
    print('dyn_df shape: ', dyn_df.shape)
    print('sta_df shape: ', sta_df.shape)
    print('outc_df shape: ', outc_df.shape)

    # Save to parquet files
    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    # Save plots
    plot_sample_data(base_dir, dyn_df)

    print(f"Synthetic data created in {base_dir}")

def create_trend_sinusoid_data(num_samples=200, num_stays=500, trend_slope=0.01):
    # Create directory structure
    base_dir = Path("demo_data/simulated/trend_sinusoids")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')  # 1-hour intervals

    # Initialize lists to hold the data
    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        amplitude = np.random.uniform(0.5, 2.0)  # Random amplitude between 0.5 and 2

        # Create sin and cos waves with the same amplitude and add a trend
        time_index = np.arange(num_samples)
        sin_wave = amplitude * np.sin(np.linspace(0, 6 * np.pi, num_samples)) + trend_slope * time_index
        cos_wave = amplitude * np.cos(np.linspace(0, 6 * np.pi, num_samples)) + trend_slope * time_index

        # Append dynamic data
        for t, sin_val, cos_val in zip(time_intervals, sin_wave, cos_wave):
            dyn_data.append([stay_id, t, sin_val, cos_val])

        # Append static and outcome data
        static_data.append([stay_id, sin_val + cos_val])
        outcome_data.append([stay_id, amplitude])

    # Create DataFrames
    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])
    print('dyn_df shape: ', dyn_df.shape)
    print('sta_df shape: ', sta_df.shape)
    print('outc_df shape: ', outc_df.shape)

    # Save to parquet files
    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    # Save plots
    plot_sample_data(base_dir, dyn_df)

    print(f"Synthetic data with trend created in {base_dir}")

def create_linear_data(num_samples=200, num_stays=500):
    base_dir = Path("demo_data/simulated/linear")
    base_dir.mkdir(parents=True, exist_ok=True)

    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')

    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        slope_var1 = np.random.uniform(0.1, 0.5)
        slope_var2 = np.random.uniform(0.1, 0.5)
        intercept = np.random.uniform(0, 10)
        noise = np.random.normal(0, 1, num_samples)

        linear_data_var1 = slope_var1 * np.arange(num_samples) + intercept + noise
        linear_data_var2 = slope_var2 * np.arange(num_samples) + intercept + noise

        for t, var1, var2 in zip(time_intervals, linear_data_var1, linear_data_var2):
            dyn_data.append([stay_id, t, var1, var2])

        static_data.append([stay_id, slope_var1 + intercept])
        outcome_data.append([stay_id, slope_var1])

    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])

    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    plot_sample_data(base_dir, dyn_df)

    print(f"Linear data created in {base_dir}")

def create_linear_noisy_data(num_samples=200, num_stays=500):
    base_dir = Path("demo_data/simulated/linear_noisy")
    base_dir.mkdir(parents=True, exist_ok=True)

    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')

    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        slope_var1 = np.random.uniform(0.1, 0.5)
        slope_var2 = np.random.uniform(0.1, 0.5)
        intercept = np.random.uniform(0, 10)
        noise = np.random.normal(0, 2, num_samples)

        linear_data_var1 = slope_var1 * np.arange(num_samples) + intercept + noise
        linear_data_var2 = slope_var2 * np.arange(num_samples) + intercept + noise

        for t, var1, var2 in zip(time_intervals, linear_data_var1, linear_data_var2):
            dyn_data.append([stay_id, t, var1, var2])

        static_data.append([stay_id, slope_var1 + intercept])
        outcome_data.append([stay_id, slope_var1])

    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])

    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    plot_sample_data(base_dir, dyn_df)

    print(f"Linear Noisy data created in {base_dir}")

def create_exponential_decay_data(num_samples=200, num_stays=500, decay_rate=0.01):
    base_dir = Path("demo_data/simulated/exponential_decay")
    base_dir.mkdir(parents=True, exist_ok=True)

    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')

    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        initial_value_var1 = np.random.uniform(1, 10)
        initial_value_var2 = np.random.uniform(1, 10)
        decay_data_var1 = initial_value_var1 * np.exp(-decay_rate * np.arange(num_samples))
        decay_data_var2 = initial_value_var2 * np.exp(-decay_rate * np.arange(num_samples))

        for t, var1, var2 in zip(time_intervals, decay_data_var1, decay_data_var2):
            dyn_data.append([stay_id, t, var1, var2])

        static_data.append([stay_id, initial_value_var1])
        outcome_data.append([stay_id, decay_rate])

    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])

    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    plot_sample_data(base_dir, dyn_df)

    print(f"Exponential decay data created in {base_dir}")

def create_seasonal_data(num_samples=200, num_stays=500):
    base_dir = Path("demo_data/simulated/seasonal")
    base_dir.mkdir(parents=True, exist_ok=True)

    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')

    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        amplitude = np.random.uniform(0.5, 2.0)
        noise = np.random.normal(0, 0.2, num_samples)
        seasonal_data = amplitude * np.sin(2 * np.pi * np.arange(num_samples) / 24) + noise

        for t, val in zip(time_intervals, seasonal_data):
            dyn_data.append([stay_id, t, val, val+np.random.uniform(0, 0.2)])

        static_data.append([stay_id, amplitude])
        outcome_data.append([stay_id, amplitude])

    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])

    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    plot_sample_data(base_dir, dyn_df)

    print(f"Seasonal data created in {base_dir}")

def create_seasonal_data_noisy(num_samples=200, num_stays=500):
    base_dir = Path("demo_data/simulated/seasonal_noisy")
    base_dir.mkdir(parents=True, exist_ok=True)

    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')

    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        amplitude = np.random.uniform(0.5, 2.0)
        noise = np.random.normal(0, 0.8, num_samples)
        seasonal_data = amplitude * np.sin(2 * np.pi * np.arange(num_samples) / 24) + noise

        for t, val in zip(time_intervals, seasonal_data):
            dyn_data.append([stay_id, t, val, val+np.random.uniform(0, 0.2)])

        static_data.append([stay_id, amplitude])
        outcome_data.append([stay_id, amplitude])

    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])

    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    plot_sample_data(base_dir, dyn_df)

    print(f"Seasonal Noisy data created in {base_dir}")


def create_global_local_seasonal_data(num_samples=200, num_stays=500):
    base_dir = Path("demo_data/simulated/global_local_seasonal")
    base_dir.mkdir(parents=True, exist_ok=True)

    stay_ids = np.arange(1, num_stays + 1)
    time_intervals = pd.to_timedelta(np.arange(0, num_samples), unit='h')

    dyn_data = []
    static_data = []
    outcome_data = []

    for stay_id in stay_ids:
        global_amplitude = np.random.uniform(0.5, 2.0)
        local_amplitude = np.random.uniform(0.1, 0.5)
        global_period = np.random.uniform(50, 100)  # Longer period for global seasonality
        local_period = 24  # Shorter period for daily (local) seasonality
        noise = np.random.normal(0, 0.3, num_samples)  # Add some noise

        global_seasonal_data = global_amplitude * np.sin(2 * np.pi * np.arange(num_samples) / global_period)
        local_seasonal_data = local_amplitude * np.sin(2 * np.pi * np.arange(num_samples) / local_period)

        combined_seasonal_data = global_seasonal_data + local_seasonal_data + noise

        for t, val in zip(time_intervals, combined_seasonal_data):
            dyn_data.append([stay_id, t, val, val + np.random.uniform(0, 0.2)])

        static_data.append([stay_id, global_amplitude])
        outcome_data.append([stay_id, global_amplitude])

    dyn_df = pd.DataFrame(dyn_data, columns=['stay_id', 'time', 'dyn1', 'dyn2'])
    sta_df = pd.DataFrame(static_data, columns=['stay_id', 'sta1'])
    outc_df = pd.DataFrame(outcome_data, columns=['stay_id', 'label'])

    dyn_df.to_parquet(base_dir / 'dyn.parquet', index=False)
    sta_df.to_parquet(base_dir / 'sta.parquet', index=False)
    outc_df.to_parquet(base_dir / 'outc.parquet', index=False)

    plot_sample_data(base_dir, dyn_df)

    print(f"Global and Local Seasonal data created in {base_dir}")


if __name__ == "__main__":
    np.random.seed(1234)
    create_sinusoid_data(num_samples=200, num_stays=2000)
    create_trend_sinusoid_data(num_samples=200, num_stays=2000)
    create_linear_data(num_samples=200, num_stays=2000)
    create_linear_noisy_data(num_samples=200, num_stays=2000)
    create_exponential_decay_data(num_samples=200, num_stays=2000)
    create_seasonal_data(num_samples=200, num_stays=2000)
    create_seasonal_data_noisy(num_samples=200, num_stays=2000)
    create_global_local_seasonal_data(num_samples=200, num_stays=2000)
