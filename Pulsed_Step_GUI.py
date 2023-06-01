import nidaqmx
from nidaqmx.stream_writers import CounterWriter
from nidaqmx.constants import AcquisitionType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import asksaveasfile
from tkinter import messagebox

# Create GUI window
window = tk.Tk()
window.title("Data Acquisition")
window.geometry("300x200")

# Function to start data acquisition
def start_acquisition():
    # User inputs
    startfreq = int(start_freq_entry.get())
    endfreq = int(end_freq_entry.get())
    freqstep = int(freq_step_entry.get())
    averages = int(averages_entry.get())
    numpts = int(numpts_entry.get())

    # Background Calculations
    endfreq += freqstep
    samprate = numpts * startfreq * averages
    numptstot = numpts * averages

    data = pd.DataFrame(columns=['Frequency %d' % (startfreq + x) for x in range(0, endfreq - startfreq, freqstep)])

    # DAQmx Initialization
    CO1 = nidaqmx.Task()
    CO1.co_channels.add_co_pulse_chan_time("Dev1/ctr0")
    CO1.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
    cw = CounterWriter(CO1.out_stream, True)
    CO1.start()

    AI1 = nidaqmx.Task()
    AI1.ai_channels.add_ai_voltage_chan("Dev1/ai0")
    AI1.timing.cfg_samp_clk_timing(samprate, sample_mode=AcquisitionType.FINITE, samps_per_chan=numptstot)
    AI1.triggers.start_trigger.cfg_dig_edge_start_trig("PFI12")

    # DAQmx Data Collection
    try:
        for x in range(startfreq, endfreq, freqstep):
            cw.write_one_sample_pulse_frequency(frequency=x, duty_cycle=0.5)
            data['Frequency %d' % x] = AI1.read(number_of_samples_per_channel=numptstot)

        # Stop all DAQmx tasks to release device memory
        CO1.stop()
        CO1.close()
        AI1.stop()
        AI1.close()

        # Data Processing
        output = pd.DataFrame({'Frequency': np.arange(startfreq, endfreq, freqstep)})
        output['Average Signal'] = data.mean()

        # Fourier Transform
        fft = np.fft.fft(output['Average Signal'])
        t = np.arange(fft.shape[0])
        freq = np.fft.fftfreq(t.shape[-1])

        # File Output
        file = asksaveasfile(initialfile='Untitled.csv', defaultextension='.csv',
                             filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file:
            output['Freq'] = freq
            output['fft'] = fft.real
            output['fft_imag'] = fft.imag
            output.to_csv(file, sep=',', index=False, line_terminator='\n')

        # Plotting
        plt.plot(freq, fft.real, freq, fft.imag)
        plt.show()

    except nidaqmx.DaqError:
        messagebox.showerror("Error", "Data acquisition failed.")

    # Close the GUI window after acquisition
    window.destroy()

# Create GUI labels and entry fields
start_freq_label = tk.Label(window, text="Start Frequency:")
start_freq_label.pack()
start_freq_entry = tk.Entry(window)
start_freq_entry.pack()

end_freq_label = tk.Label(window, text="End Frequency:")
end_freq_label.pack()
end_freq_entry = tk.Entry(window)
end_freq_entry.pack()

freq_step_label = tk.Label(window, text="Frequency Step:")
freq_step_label.pack()
freq_step_entry = tk.Entry(window)
freq_step_entry.pack()

averages_label = tk.Label(window, text="Averages:")
averages_label.pack()
averages_entry = tk.Entry(window)
averages_entry.pack()

numpts_label = tk.Label(window, text="Number of Data Points:")
numpts_label.pack()
numpts_entry = tk.Entry(window)
numpts_entry.pack()

# Button to start data acquisition
start_button = tk.Button(window, text="Start Acquisition", command=start_acquisition)
start_button.pack()

# Run the GUI event loop
window.mainloop()

