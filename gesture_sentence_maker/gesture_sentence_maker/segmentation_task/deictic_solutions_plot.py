import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gesture_sentence_maker

def deictic_solutions_plot_save(
        deictic_solutions,
        nlp_timestamps,  # List of normalized timestamps for vertical lines
        hand_velocity_threshold = 0.05,  # Threshold for hand steadiness
        T=1.0,  # Maximum allowed time gap [seconds]
    ):
    # Get the first timestamp to normalize the time axis
    first_stamp = deictic_solutions[0]["target_object_stamp"]

    # Build the DataFrame
    data = []
    for solution in deictic_solutions:
        # Normalize the timestamps
        stamp = solution["target_object_stamp"] - first_stamp
        object_names = solution["object_names"]
        hand_velocity = solution["hand_velocity"]  # Assuming hand_velocity is a list or array
        object_likelihoods = solution["object_likelihoods"]
        row = {'stamp': stamp, 'hand_velocity': hand_velocity}
        for name, likelihood in zip(object_names, object_likelihoods):
            row[name] = likelihood
        data.append(row)

    df = pd.DataFrame(data)
    df.set_index('stamp', inplace=True)
    df.sort_index(inplace=True)

    # Insert NaNs where the time gap exceeds T to break the lines
    times = df.index.values
    time_diffs = np.diff(times)
    large_gaps = np.where(time_diffs > T)[0]

    for idx in large_gaps[::-1]:  # Reverse to avoid reindexing issues
        insert_idx = times[idx] + (times[idx + 1] - times[idx]) / 2
        nan_row = pd.DataFrame([[np.nan]*len(df.columns)], index=[insert_idx], columns=df.columns)
        df = pd.concat([df.iloc[:idx + 1], nan_row, df.iloc[idx + 1:]])

    df.sort_index(inplace=True)

    # Get the list of object names
    object_names = df.columns.tolist()
    object_names.remove('hand_velocity')  # Remove 'hand_velocity' from object names

    # Define markers and colors for each object
    markers = ['o', 's', '^', 'D', 'v', '*']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Shade regions where hand_velocity is below threshold
    below_threshold = df['hand_velocity'] <= hand_velocity_threshold

    # Find contiguous regions where hand_velocity is below threshold
    df['below_threshold'] = below_threshold
    df['group'] = (df['below_threshold'] != df['below_threshold'].shift()).cumsum()

    steady_groups = df[df['below_threshold']].groupby('group')

    for name, group in steady_groups:
        start_time = group.index[0]
        end_time = group.index[-1]
        plt.axvspan(start_time, end_time, color='lightblue', alpha=0.3)

    # Plot the object likelihoods
    for i, name in enumerate(object_names):
        plt.plot(df.index, df[name],
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 label=name)

    # Add vertical dashed lines at the specified NLP timestamps
    for idx, t in enumerate(nlp_timestamps):
        plt.axvline(x=t-first_stamp, linestyle='--', color='k',
                    label='NLP Selection' if idx == 0 else None)

    plt.xlabel('Time (seconds since first stamp)')
    plt.ylabel('Likelihoods')
    plt.title('Deictic Solutions over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure to a PNG file
    plt.savefig(f'{gesture_sentence_maker.path}/deictic_solutions.png')
