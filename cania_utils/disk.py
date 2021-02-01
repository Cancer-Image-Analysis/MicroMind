from pathlib import Path
import zipfile
import pandas as pd

from cania_utils.image import imread_color, imread_tiff, imwrite

PNG = '.png'
JPG = '.jpg'
CSV = '.csv'
TIF = '.tif'
ZIP = '.zip'
LSM = '.lsm'


class Disk(object):
    def __init__(self, location):
        self.location = Path(location)
        if not self.location.exists():
            raise FileNotFoundError()

    def write(self, filename, filedata):
        filepath = self.location / filename
        extension = filepath.suffix
        filepath = str(filepath)
        if extension == PNG:
            imwrite(filepath, filedata)

    def read(self, filename):
        filepath = self.location / filename
        extension = filepath.suffix
        filepath = str(filepath)
        if extension == PNG or extension == JPG:
            return imread_color(filepath)
        if extension == CSV:
            return pd.read_csv(filepath)
        if extension == TIF or extension == LSM:
            return imread_tiff(filepath)

    def unzip(self, filename):
        filepath = self.location / filename
        extension = filepath.suffix
        filepath = str(filepath)
        unzip_folder = filename.replace(ZIP, '')
        new_location = self.location / unzip_folder
        if new_location.exists():
            return Disk(new_location)

        if extension == ZIP:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(str(new_location))
            return Disk(new_location)

    def next(self, next_location):
        filepath = self.location / next_location
        filepath.mkdir(parents=True, exist_ok=True)
        return Disk(filepath)

    def ls(self, regex='*'):
        return self.location.glob(regex)

    def save_as_csv(self, data, filename):
        filepath = self.location / filename
        if isinstance(data, dict):
            df = pd.DataFrame(data=data)
        elif isinstance(data, pd.DataFrame):
            df = data
        df.to_csv(str(filepath))
