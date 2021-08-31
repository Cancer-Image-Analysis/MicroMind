from pathlib import Path
import zipfile
import pandas as pd

from abc import ABC
from dataclasses import dataclass, field, InitVar

from micromind.io.image import (imread_color, imread_tiff, imread_czi,
                                imwrite, imwrite_tiff)

PNG = '.png'
JPG = '.jpg'
CSV = '.csv'
TIF = '.tif'
ZIP = '.zip'
LSM = '.lsm'
CZI = '.czi'


@dataclass
class DriveEntity(ABC):
    path: InitVar[str]
    _path: Path = field(init=False)

    def __post_init__(self, path):
        self._path = Path(path)

    def __getattr__(self, attr):
        return getattr(self._path, attr)

    def __truediv__(self, key):
        return self._path / key


@dataclass
class Directory(DriveEntity):
    def __post_init__(self, path):
        super().__post_init__(path)
        if not self.exists():
            _err = f'The directory {self._path} does not exist!'
            raise ValueError(_err)
        if not self.is_dir():
            _err = f'The path {self._path} is not pointing to a directory!'
            raise ValueError(_err)

    def write(self, filename, filedata):
        filepath = self / filename
        extension = filepath.suffix
        filepath = str(filepath)
        if extension == PNG:
            imwrite(filepath, filedata)
        if extension == TIF or extension == LSM:
            imwrite_tiff(filepath, filedata)

    def read(self, filename):
        filepath = self / filename
        if not filepath.exists():
            _err = f'The file {filepath} does not exist!'
            raise ValueError(_err)
        if not filepath.is_file():
            _err = f'The path {filepath} is not pointing to a file!'
            raise ValueError(_err)
        extension = filepath.suffix
        filepath = str(filepath)
        if extension == PNG or extension == JPG:
            return imread_color(filepath)
        if extension == CSV:
            return pd.read_csv(filepath)
        if extension == TIF or extension == LSM:
            return imread_tiff(filepath)
        if extension == CZI:
            return imread_czi(filepath)

    def unzip(self, filename):
        filepath = self / filename
        extension = filepath.suffix
        filepath = str(filepath)
        unzip_folder = filename.replace(ZIP, '')
        new_location = self / unzip_folder
        if new_location.exists():
            return Directory(new_location)

        if extension == ZIP:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(str(new_location))
            return Directory(new_location)

    def next(self, next_location):
        filepath = self / next_location
        filepath.mkdir(parents=True, exist_ok=True)
        return Directory(filepath)

    def ls(self, regex='*'):
        return self.glob(regex)

    def save_as_csv(self, data, filename):
        filepath = self / filename
        if isinstance(data, dict):
            df = pd.DataFrame(data=data)
        elif isinstance(data, pd.DataFrame):
            df = data
        df.to_csv(str(filepath))


class WorkingDirectory(Directory):
    pass


class File(DriveEntity):
    def __init__(self, path):
        super().__init__(path)
        if not self.exists():
            raise ValueError(f'The file {self} does not exist!')
        if not self.is_dir():
            raise ValueError(f'The path {self} is not pointing to a file!')
