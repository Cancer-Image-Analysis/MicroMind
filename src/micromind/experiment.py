import numpy as np
from micromind.microcell.cell import Cell2D, CellHourglass


class ExperimentalCondition:
    pass


class TimeCondition(ExperimentalCondition):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset


class Experiment:
    def __init__(self):
        self.conditions = {}

    def add_condition(self, condition):
        self.conditions[condition.name] = condition

    def run(self):
        pass


class HeterogeneityExp(Experiment):
    def __init__(self, profiling, reader):
        super().__init__()
        self.profiling = profiling
        self.method = self.profiling.infos.method
        for name, condition in self.profiling.infos.conditions.items():
            self.add_condition(TimeCondition(name, reader.read_dataset(dataset_filename=condition["filename"])))

    def run(self):
        results = {"all": [], "n": [], "image": []}
        print(self.conditions.keys())
        for name in self.conditions.keys():
            results[name] = []
            X = self.conditions[name].dataset.train_x
            y = self.conditions[name].dataset.train_y
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]
                z, h, w = xi[0].shape[-3:]
                compose_stack = np.zeros(shape=(z, len(xi), h, w))
                for j, xij in enumerate(xi):
                    compose_stack[:, j] = xij
                labels_image = yi[0]
                label_numbers = np.unique(labels_image)

                if self.method == "standard":
                    for n in label_numbers:
                        if n == 0:
                            continue
                        mask = (labels_image == n).astype('uint8')
                        results["n"].append(n)
                        results["image"].append(f"{name}-{i}")
                        cell = Cell2D.from_mask(mask, 'cell')

                        profile, labels = self.profiling.get_profile(cell, compose_stack)
                        results["all"].append(dict(zip(labels, profile)))
                        results[name].append(dict(zip(labels, profile)))
                elif self.method == "hourglass":
                    assert len(label_numbers) == 3, "not ok"
                    ctl_mask = (labels_image == 1).astype('uint8')
                    ctl_cell = Cell2D.from_mask(ctl_mask, f"CTL-{i}")

                    target_mask = (labels_image == 2).astype('uint8')
                    target_cell = Cell2D.from_mask(target_mask, f"target-{i}")

                    angle = ctl_cell.angle_with_x_axis(target_cell)

                    ctl_hourglass = CellHourglass(ctl_cell.name, ctl_cell.mask, ctl_cell.x, ctl_cell.y, angle)
                    profile, labels = self.profiling.get_profile(ctl_hourglass, compose_stack)
                    results["all"].append(dict(zip(labels, profile)))
                    results[name].append(dict(zip(labels, profile)))
        return results
