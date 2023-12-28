import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from utils import get_data

def plot_signal(data, start_index, end_index):
    plt.figure(figsize=(10, 4))
    plt.plot(data[start_index:end_index])
    plt.title(f"Signal from index {start_index} to {end_index}")
    plt.xlabel("Index")
    plt.ylabel("Signal Value")
    plt.grid(True)
    plt.ylim(0, 0.2)
    plt.show()

def save_numpy_array(data_slice, folder_name, subject_number, start_index):
    # Create filename
    filename = os.path.join(folder_name, f"subject_{subject_number}_index_{start_index}.npy")

    # Save the numpy array slice
    np.save(filename, data_slice)
    print(f"Data slice saved to {filename}")

def main():
    while True:
        print("Enter subject number: ")
        subject_number = input().strip()
        print("Enter start index: ")
        start_index = int(input().strip())

        # Load data based on subject number
        data = get_data(subject_number)  # Replace this with your data loading logic
        data = data['data_df']['L_Active(g)'].values.flatten()

        # Determine the slice range
        end_index = start_index + 300

        # Plot the signal
        plot_signal(data, start_index, end_index)

        # Get label from user
        print("Enter label (0, 1, or 2): ")
        label = input().strip()
        if str(label) == '3':
            print('abord, continuing')
            continue

        # Create folder if not exists
        folder_name = f"svm_data/label_{label}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the numpy array slice
        data_slice = data[start_index:end_index]
        save_numpy_array(data_slice, folder_name, subject_number, start_index)

if __name__ == "__main__":

    main()
