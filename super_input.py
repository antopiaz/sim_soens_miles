#%%
import numpy as np

class SuperInput():
    def __init__(self,**entries):
        self.type ="MNIST", # random for rand gen
        self.duration = 100,
        self.channels = 28*28,
        self.slow_down = 10,
        self.name = 'Super_Input'
        self.__dict__.update(entries)
        
        if self.type == "MNIST":
            print("Generating MNNIST dataset...")
            mnist_indices, mnist_spikes = self.MNIST()
            self.spike_arrays = [mnist_indices,mnist_spikes]
            self.spike_rows = self.array_to_rows(self.spike_arrays)


    def gen_rand_input(self,spiking_indices,max_amounts):
        # self.input = np.random.randint(self.duration, size=())
        input = [ [] for _ in range(self.channels) ]
        spikers = np.random.randint(self.channels,size=spiking_indices)
        sum = []
        for n in spikers:
            input[n] = np.sort(np.random.randint(self.duration,size=np.random.randint(max_amounts)))
            sum.append(len(input[n]))
        print("Total number of spikes:", np.sum(sum))
        print("Spiking at neurons: ", spikers)
        return input

    def gen_ordered_input(self):
        pass

    def load_input(self):
        pass

    def rows_to_array(self,input):
        spikes = [ [] for _ in range(self.channels) ]
        count = 0
        for n in range(self.channels):
            if np.any(input[n]):
                # print(n)
                spikes[0].append(np.ones(len(input[n]))*n)
                spikes[1].append(input[n])
            count+=1
        spikes[0] =np.concatenate(spikes[0])
        spikes[1] = np.concatenate(spikes[1])
        return spikes

    def array_to_rows(self,array):
        rows = [ [] for _ in range(self.channels) ]
        for i in range(len(array[0])):
            rows[array[0][i]].append(array[1][i])
        return rows

    def MNIST(self):
        import keras
        import brian2
        from keras.datasets import mnist
        print("load")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("loaded")
        # simplified classification (0 1 and 8)
        X_train = X_train[(y_train == 1) | (y_train == 0) | (y_train == 2)]

        # pixel intensity to Hz (255 becoms ~63Hz)
        X_train = X_train / 4 
        numbers = [0,1,2]
        classes = ["zero", "one", "two"]

        # Generate spiking data
        brian2.start_scope()
        self.units = []
        self.times = []
        self.data = {}
        channels = 28*28
        X = X_train[self.index].reshape(channels)
        P = brian2.PoissonGroup(channels, rates=(X/self.slow_down)*brian2.Hz)
        MP = brian2.SpikeMonitor(P)
        net = brian2.Network(P, MP)
        net.run(self.duration*brian2.ms)
        spikes_i = np.array(MP.i[:])
        spikes_t = np.array(MP.t[:])
        self.indices = spikes_i
        self.times = spikes_t*1000

        return self.indices, self.times
