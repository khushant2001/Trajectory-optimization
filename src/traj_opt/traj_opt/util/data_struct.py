import csv
import numpy as np
import matplotlib.pyplot as plt

class DataStruct:
    def __init__(self, ):
        self.timestamps = []
        self.data = []
        self.input = []
        self.names = []
        self.rawdata = []

    def to_csv(self, filename=None):
        try:
            if(filename==None):
                print("No filename given!")
                raise NameError
            keys = list(self.data[0].keys())
            keys.insert(0, 'timestamp')
            for a in list(self.input[0].keys()): keys.append(a)
            file = open(filename, 'w',newline='')
            writer = csv.writer(file, delimiter=',')
            writer.writerow(keys)
            for i in range(len(self.timestamps)):
                row = []
                row.append(self.timestamps[i])
                for j in list(self.data[i].values()):
                    row.append(j)
                for h in list(self.input[i].values()):
                    row.append(h)
                writer.writerow(row)
            file.close()
        except Exception as e: 
            print(e)

    def from_csv(self, filename=None):
        try:
            if(filename==None):
                print("No filename given!")
                raise NameError

            file = open(filename, 'r',newline='')
            self.rawdata = np.genfromtxt(file, delimiter=',',names=True,dtype=np.float64)
            self.names = self.rawdata.dtype.names

            file.close()
            # file = open(filename, 'r',newline='')
            # reader = csv.DictReader(file, fieldnames=names, delimiter=',')
            # listdata = list(reader)
            # for i in range(1,len(listdata)):
            #     self.timestamps.append(listdata[i]['timestamp'])
            #     del listdata[i]['timestamp']
            #     self.data.append(listdata[i])

            # file.close()

        except Exception as e: 
            print(e)

    def plot_unorganized(self):
        time = self.rawdata['timestamp']
        roll = self.rawdata['stabilizerroll']
        pitch = self.rawdata['stabilizerpitch']*-1 # for some reason it's inverted
        desiredpitchrate = self.rawdata['controllerpitchRate'] # pitchrate at controller is in deg/sec
        pitchrate = self.rawdata['stateEstimateZratePitch']*180.0/(np.pi*1000.0) # state estimate reading is in millirad/sec
        inputpitch = self.rawdata['pitch']

        plt.figure(1)
        plt.title("Pitch Step Input Response")
        plt.plot(time, inputpitch, 'r-', label='desired-pitch')
        plt.plot(time, pitch, 'g-', label='actual-pitch')
        plt.legend()
        plt.figure(2)
        plt.title("Pitch Rate")
        plt.plot(time, desiredpitchrate, 'r-', label='desired-pitchrate')
        plt.plot(time, pitchrate, 'g-', label='actual-pitchrate')
        plt.legend()
        
        plt.show(block=True)


if __name__ == "__main__":
    d = DataStruct()
    d.from_csv('csvs/test1.csv')
    d.plot_unorganized()
    print('d')