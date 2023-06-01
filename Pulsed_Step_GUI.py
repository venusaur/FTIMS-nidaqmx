#--------------------------------------#
                #Imports#
#--------------------------------------#
#tkinter imports:
from tkinter import * 
from tkinter.filedialog import asksaveasfile

#NI DAQMX Imports:
import nidaqmx
from nidaqmx.stream_writers import CounterWriter
from nidaqmx.constants import AcquisitionType
from nidaqmx.constants import TerminalConfiguration

#Matplotlib imports:
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Misc other imports:
import time
import threading
import numpy as np
import pandas as pd

#-----------GUI Configuration----------#
root = Tk()
root.title("IMS Control")
root.configure(bg = "black")
root.resizable(width = 0, height = 0)
plotcolor = "r"
root.geometry("1250x850")

#--------------------------------------#
                #User inputs#
#--------------------------------------#
startfreq = 5   #Starting Frequency
endfreq = 100            #Ending Frequency
freqstep = 1    #Steps in frequency between scans

averages = 5    #Number of averages at each frequency step - SOFTWARE WILL USE THIS VALUE TO CALCULATE TIME REQUIRED AT LOWEST FREQUENCY USE THE TRIGGER TO AVERAGE THIS SAME AMOUNT OF TIME AT EACH STEP
numpts = 100    #Number of data points to take at each frequency setp


#--------------------------------------#
        #Background Calculations#
#--------------------------------------#
endfreq = endfreq + freqstep   #Corrects for numpy arange's inability to iterate to a number (otherwise it would stop 1 freqstep below the end freq)
samprate = (numpts*startfreq)*averages   #Calculates the sampling rate based on averges, startfreq, and numpts
numptstot = numpts*averages             #Calculates how many points to take after trigger

data = np.empty([numptstot,(endfreq-startfreq)])   
names = ['Frequency %d' % (startfreq)]

for x in np.arange (1, ((endfreq-startfreq)), freqstep):
    names.append("Frequency %d" % (x+startfreq))

data = pd.DataFrame(data, columns = names)  #Initializes DataFrame for collecting data. DataFrame at this point is arranged with 1 column per frequency to be stepped.


#--------------------------------------#
        #NIDAQMX intializations#
#--------------------------------------#
CO1 = nidaqmx.Task()
CO1.co_channels.add_co_pulse_chan_time("Dev1/ctr0")
CO1.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
cw = CounterWriter(CO1.out_stream, True)
CO1.start()

AI1 = nidaqmx.Task()
AI1.ai_channels.add_ai_voltage_chan("Dev1/ai0")
AI1.timing.cfg_samp_clk_timing(samprate, sample_mode=AcquisitionType.FINITE, samps_per_chan=numptstot)
AI1.triggers.start_trigger.cfg_dig_edge_start_trig("PFI12")


#--------------------------------------#
        #Globals#
#--------------------------------------#

dtime = [] #Global array for drift time (x) axis
data = [] #Global array for collection of data within collection for loop
datavg = [] #Global array for holding the most recent average taken
datait = [] #Global array to hold all data data in the last experiment
stop = False #Stop immediately - stop all averages
stop1 = False #Stops after current iteration
start = False #Internal global for monitoring if the user has previously sent a start command - prevents double clicking on start
HVstatus = False #Internal global for tracking status of HV output
maxv = 20000

def plotupdate(): #Gets data from the most recent run and redraws the plots
   global dtime
   global datavg
   fig.clear()
   plot1 = fig.subplots()
   plot1.plot(dtime,datavg, plotcolor)
   plot1.set_xlabel("Time (ms)")
   plot1.set_ylabel("Intensity (a.u.)")
   canvas.draw()
   canvas.get_tk_widget().pack()

def button_stop_command():
  # If the STOP button is pressed then terminate both data collection loops (average and iteration)
  global stop
  global stop1
  if start == True:
    stop = True
    stop1 = True

def button_stop_command_it():
  # If the STOP button is pressed then terminate the iteration loop but allows average loop to continue until complete
  global stop
  global stop1
  if start == True:
    stop = False
    stop1 = True

def button_start_command():
    global start  
    if start != True: #Checks if a start command has already been received and ignores any start commands from the user while this function is running.
        global dtime
        global datait
        global datavg
        global stop
        global stop1
        global maxv
        stop = False
        start = True
        startfreq = int(startfreq_input.get())
        endfreq = int(endfreq_input.get())
        freqstep = int(step_input.get())
        averages = int(avg_input.get())
        numpts = int(nump_input.get())
        endfreq += freqstep
        samprate = numpts * startfreq * averages
        numptstot = numpts * averages

        data = pd.DataFrame(columns=['Frequency %d' % (startfreq + x) for x in range(0, endfreq - startfreq, freqstep)])

        for x in range(startfreq, endfreq, freqstep):
            cw.write_one_sample_pulse_frequency(frequency=x, duty_cycle=0.5)
            data['Frequency %d' % x] = AI1.read(number_of_samples_per_channel=numptstot)
            print(x)
        # Stop all DAQmx tasks to release device memory
        CO1.stop()
        CO1.close()
        AI1.stop()
        AI1.close()

        #--------------------------------------#
                #Data Processing#
        #--------------------------------------#
        output = pd.DataFrame({'Frequency': np.arange(startfreq, endfreq, freqstep)})
        output['Average Signal'] = data.mean()

        #--------------------------------------#
                #Fourier Transform#
        #--------------------------------------#
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
            output.to_csv(file, sep=',', index=False)

        # Plotting
        #plt.plot(freq, fft.real, freq, fft.imag)
        #plt.show()

def button_starter(): #Threading split to allow start command to run while GUI is still active
  t = threading.Thread(target=button_start_command)
  t.start()

def load_command():  #TO DO - Create ability to load data file - Will need to uncomment the button_load grid command below to make functional
   time.sleep(0.1)

def save_command():  #Opens file dialog for saving data as a *.csv file. Data maintained in memory until next analysis is begun.
   global start  #Imports global start to check if data is currently being obtained.
   if start != True: #Prevents call while data is being obtained
        global datait #Imports data for saving
        global dtime
        datasave = pd.DataFrame(np.transpose(datait)) #Creates new dataframe using the iteration data transposed.
        datasave.columns = ["Iteration " + str(i) for i in range(1, datasave.shape[1] + 1)] #Renames each column to iteration number
        datasave.insert(value = dtime, loc=0, column = "Time (ms)")
        datasave.to_csv((asksaveasfile(initialfile = 'Untitled.csv', defaultextension='.csv', 
                                       filetypes=[("CSV Files","*.csv"),("All Files", "*.*")])), 
                                       sep=',',index=False, lineterminator = '\n')   #Calls a file dialog box to save the data
        #print(datasave)

#--------------------------------------#
        #GUI Frames #
#--------------------------------------#
pltframe = Frame(master = root, bg = "white", highlightbackground = "black", highlightthickness=1) #Frame for main IMS Plot
dataframe = Frame(master = root, bg = "black") #Frame for reporting back data - utilizes white label widgets
inputframe = Frame(master = root, bg = "black") #Frame for user input - utilizes blue label and entry widgets
hvoutframe = Frame(master = root, bg= "black")
hvinframe = Frame(master = root, bg = "black")

#--------------------------------------#
        #GUI Widgets#
#--------------------------------------#
#Buttons:
button_start = Button(root, text="START", padx=30, pady=10, command=button_starter, bg = "#36db23") #Start Analysis Button
button_stopinst = Button(root, text="STOP Immediately", padx=5, pady=10, command=button_stop_command, bg = "red") #Stop Analysis Button
button_stopafter = Button(root, text = "STOP after iteration", padx = 5, pady = 10, command = button_stop_command_it, bg = "red") #Stop analysis after next iteration button
button_load = Button(root, text = "Load File", padx = 30, pady=10, command = load_command) #Load old file 
button_save = Button(root, text = "Save File", padx = 30, pady=10, command = save_command) #Save current data
#button_hvonoff = Button(root, text = "Turn HV ON", padx = 30, pady = 10, command = startstopHV, bg = "#36db23")
#button_updateHV = Button(hvoutframe, text = "Update HV Values",padx = 30, pady = 10, command = updateHV)

#Data Output Labels:
iteration = Label(master = dataframe, text = "Iteration 0", bg = "black", fg = "white", font = ("Arial", 16, "bold")) #Label for current iteration
average = Label(master = dataframe, text = "Average 0", bg = "black", fg = "white", font = ("Arial", 16, "bold")) #Label for current average
IMSVin = Label(master = hvinframe,text = "IMS Voltage: 0", bg = "black", fg = "white", font = ("Arial", 16, "bold"))
ESIVin = Label(master = hvinframe,text = "ESI Bias: 0", bg = "black", fg = "white", font = ("Arial", 16, "bold"))
IMSVinV = Label(master = hvinframe, text= " V", bg = "black", fg = "white", font = ("Arial", 16, "bold"))
ESIVinV = Label(master = hvinframe, text= " V", bg = "black", fg = "white", font = ("Arial", 16, "bold"))

#GUI User Inputs:
avg_input = Entry(master = inputframe, justify = "right", font = ("Arial", 16, "bold"), width = 6) #User input for number of averages
avg_input.insert(0, averages) #Sets default averages to 10
avg_inputl = Label(master = inputframe, text = "Averages: ", bg = "black", fg = "#0597ff", font = ("Arial", 16, "bold")) #Label for averages

startfreq_input = Entry(master = inputframe, justify = "right", font = ("Arial", 16, "bold"), width = 6) #User input for scantime
startfreq_input.insert(0, startfreq) #Sets default scantime to 50 ms
startfreq_input1 = Label(master = inputframe, text = "Set Freq: ", bg = "black", fg = "#0597ff", font = ("Arial", 16, "bold")) #Label for averages

nump_input = Entry(master = inputframe, justify = "right", font = ("Arial", 16, "bold"), width = 6) #User input for number of points to measure
nump_input.insert(0, numpts) #Sets default number of points to 4000
nump_inputl = Label(master = inputframe, text = "Number of Points: ", bg = "black", fg = "#0597ff", font = ("Arial", 16, "bold")) #Label for number of points input

endfreq_input = Entry(master = inputframe, justify = "right", font = ("Arial", 16, "bold"), width = 6) #User input for pulse width
endfreq_input.insert(0, endfreq) #Sets default to 0.2 ms pulse width
endfreq_inputl = Label(master = inputframe, text = "End Freq: ", bg = "black", fg = "#0597ff", font = ("Arial", 16, "bold")) #Label for averages

IMSV_input = Entry(master = hvoutframe, justify = "right", font = ("Arial", 16, "bold"), width = 6)

step_input = Entry(master = inputframe, justify = "right", font = ("Arial", 16, "bold"), width = 6) #User input for number of points to measure
step_input.insert(0, freqstep) #Sets default to 0.2 ms pulse width
step_input1 = Label(master = inputframe, text = "Step Freq: ", bg = "black", fg = "#0597ff", font = ("Arial", 16, "bold")) #Label for averages

IMSV_input = Entry(master = hvoutframe, justify = "right", font = ("Arial", 16, "bold"), width = 6)


IMSV_input.insert(0, 10000)
IMSV_inputl = Label(master = hvoutframe, text = "IMS Voltage: ", bg = "black", fg = "#0597ff", font = ("Arial", 16, "bold"))
ESIV_input = Entry(master = hvoutframe, justify = "right", font = ("Arial", 16, "bold"), width = 6)

ESIV_input.insert(0,3000)
ESIV_inputl = Label(master = hvoutframe, text = "ESI Bias: ", bg = "black", fg = "#0597ff", font = ("Arial", 16, "bold"))

#--------------------------------------#
        #Matplotlib Plots#
#--------------------------------------#
# Main IMS Plot:
#pltlabel = Label(pltframe, text = "IMS Plot", font = ("Arial", 16, "bold"), bg = "white")
#pltlabel.pack(side = TOP)
fig = Figure(figsize = (8,4), dpi = 80)
plot1 = fig.subplots()
plot1.plot(dtime,datavg)
plot1.set_xlabel("Time (ms)")
plot1.set_ylabel("Intensity (a.u.)")
canvas = FigureCanvasTkAgg(fig, master = pltframe)
canvas.draw()
canvas.get_tk_widget().pack(side = BOTTOM)
toolbar = NavigationToolbar2Tk(canvas, pltframe)
toolbar.update()
canvas.get_tk_widget().pack(side = TOP)

#-------------------------------------#
        #GUI Layout#
#--------------------------------------#
button_start.grid(row=1,column=1, pady = 10) #Start Button Position
button_stopinst.grid(row=1, column=4, pady = 10) #Stop Button Position
button_stopafter.grid(row=1, column=5, pady = 10)
pltframe.grid(row = 2, column = 1, columnspan = 5, rowspan = 2) #Main IMS Plot Position
#button_load.grid(row = 1, column = 2, padx = 10, pady = 10)
button_save.grid(row = 1, column = 3, padx = 10, pady = 10)
#button_hvonoff.grid(row = 1, column = 6, padx = 10, pady = 10, sticky = "w")
dataframe.grid(row = 2, column = 6, sticky = "nw", padx = 5)
iteration.grid(row = 0, pady = 10, sticky = "w")
average.grid(row = 1, pady = 10, sticky = "w")
inputframe.grid(column = 0, row = 2, padx = 5, rowspan = 2, sticky = "n")
avg_inputl.grid(row = 0, column = 0, sticky = "e", pady = 10)
avg_input.grid(row = 0, column = 1, sticky = "w", pady = 10)
startfreq_input1.grid(row = 1, column = 0, sticky = "e")
startfreq_input.grid(row = 1, column = 1, sticky = "w")
nump_inputl.grid(row = 3, column = 0, sticky = "e", pady = 10)
nump_input.grid(row = 3, column = 1, sticky = "w", pady = 10)
endfreq_inputl.grid(row = 2, column = 0, sticky = "e")
endfreq_input.grid(row = 2, column = 1, sticky = "w")
step_input1.grid(row = 4, column = 0, sticky = "e")
step_input.grid(row = 4, column = 1, sticky = "w")



hvoutframe.grid(row = 4, column = 0, rowspan = 2, pady = 5, sticky = "ne", padx = 5)
hvinframe.grid(row = 4,column = 6, rowspan = 2, pady = 5, sticky = "n", padx = 5)

IMSVin.grid(row = 1, column = 0, sticky = "w")
ESIVin.grid(row = 2, column = 0, sticky = "w", pady = 30)
IMSVinV.grid(row = 1, column = 2, sticky = "e", padx = 10)
ESIVinV.grid(row = 2, column = 2, sticky = "e", padx = 10)
#button_updateHV.grid(row = 0, column = 0, columnspan = 2, pady = 10)
IMSV_inputl.grid(row = 1, column = 0,sticky = "e", pady = 10)
ESIV_inputl.grid(row = 2, column = 0,sticky = "e", pady = 10)
IMSV_input.grid(row = 1, column = 1,sticky = "w", pady = 10)
ESIV_input.grid(row = 2, column = 1,sticky = "w", pady = 10)

root.mainloop() #Main loop - runs until window is closed and looks for user inputs

#Stop and close all NiDAQMX tasks upon window close:
CO1.stop()
CO1.close()


AI1.stop()
AI1.close()
