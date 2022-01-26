import numpy as np
import itertools


class CellStainingProfile:
    def __init__(self, channels, names, threshold=0, sep=' '):
        self.channels = channels
        self.names = names
        self.threshold = threshold
        self.sep = sep
        self.lst = list(itertools.product([0, 1], repeat=len(channels)))
        self.lst = sorted(self.lst, key=lambda c: sum(c), reverse=True)

    def get_profile(self, cell, stainings):
        inside_cell = cell.mask > 0
        labels = []
        profile = []
        channels = {name: stainings[:, channel] for channel, name in zip(self.channels, self.names)}
        channels_true = {}
        channels_false = {}
        for name, channel in channels.items():
            z_stack = [np.logical_and.reduce([z > self.threshold, inside_cell]) for z in channel]
            channels_true[name] = z_stack
            channels_false[name] = np.bitwise_not(z_stack)

        for combination in self.lst:
            if combination == (0, 0, 0):
                continue
            combination_names = []
            combination_values = []
            for i in range(len(self.channels)):
                name = self.names[i]
                if combination[i] == 1:
                    combination_names.append(name)
                    combination_values.append(channels_true[name])
                else:
                    combination_values.append(channels_false[name])            
            s = np.count_nonzero(np.logical_and.reduce(combination_values))
            labels.append(f' {self.sep} '.join(combination_names))
            profile.append(s)

        return profile, labels
