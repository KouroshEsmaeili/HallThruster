from dataclasses import dataclass, field
import numpy as np
import os


@dataclass
class MagneticField:
    file: str = ""
    z: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    B: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # Alternate constructor using keyword arguments.
    @classmethod
    def from_keywords(cls, file: str = "", z=None, B=None):
        if z is None:
            z = np.array([],dtype=np.float64)
        if B is None:
            B = np.array([],dtype=np.float64)
        return cls(file=file, z=z, B=B)


def load_magnetic_field_inplace(magnetic_field: MagneticField, include_dirs=None) -> None:
    if include_dirs is None:
        include_dirs = []
    # If either z or B is empty, attempt to load from file.
    if magnetic_field.z.size == 0 or magnetic_field.B.size == 0:
        # Append the current working directory.
        dirs = list(include_dirs)
        dirs.append(os.getcwd())
        for dir_path in dirs:
            file_path = os.path.join(dir_path, magnetic_field.file)
            if os.path.exists(file_path):
                # Load the data assuming comma-delimited values.
                # This returns a 2D array with at least two columns.
                data = np.loadtxt(file_path, delimiter=',')
                # Ensure data is at least 2D.
                if data.ndim == 1:
                    data = np.atleast_2d(data)
                magnetic_field.z = data[:, 0]
                magnetic_field.B = data[:, 1]
                return
        raise ValueError(f"Magnetic field file '{magnetic_field.file}' not found in directories {dirs}")


def load_magnetic_field(file: str, include_dirs=None) -> MagneticField:
    field_obj = MagneticField(file=file)
    load_magnetic_field_inplace(field_obj, include_dirs=include_dirs)
    return field_obj
