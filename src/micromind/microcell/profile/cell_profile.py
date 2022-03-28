import itertools

import numpy as np


class CellStainingProfile:
    def __init__(self, infos):
        self.infos = infos
        self.lst = list(itertools.product([0, 1], repeat=len(infos.channels)))
        self.lst = sorted(self.lst, key=lambda c: sum(c), reverse=True)

    def get_profile(self, cell, stainings):
        inside_cell = cell.mask > 0
        labels = []
        profile = []
        channels_true = {}
        channels_false = {}
        for i, channel in self.infos.channels.items():
            channel_data = stainings[:, i]
            name = channel["name"]
            threshold = channel["threshold"]
            z_stack = [
                np.logical_and.reduce([z >= threshold[0], inside_cell])
                for z in channel_data
            ]
            channels_true[name] = z_stack
            channels_false[name] = np.bitwise_not(z_stack)

        for combination in self.lst:
            combination_names = []
            combination_values = []
            for i, channel in self.infos.channels.items():
                name = channel["name"]
                if combination[i] == 1:
                    combination_names.append(name)
                    combination_values.append(channels_true[name])
                else:
                    combination_values.append(channels_false[name])
            s = np.count_nonzero(np.logical_and.reduce(combination_values))
            label_name = f"{self.infos.separator}".join(combination_names)
            labels.append(label_name)
            profile.append(s)

        for i, channel in self.infos.channels.items():
            name = channel["name"]
            labels.append(f"{name}+")
            profile.append(np.count_nonzero(channels_true[name]))
        return profile, labels
