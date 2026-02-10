import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Csv:
    columns: List[str]
    data: dict

    def __getitem__(self, item):
        return self.data[item]


def read_csv_reader(csv_reader: str, delimiter=','):
    data = {}
    columns = []

    reader = csv.reader(io.StringIO(csv_reader), delimiter=delimiter)

    for i, line in enumerate(reader):
        line = [x.strip() for x in line]
        if i == 0:
            columns = line
            data = {x: [] for x in line}
        else:
            for c, attr in enumerate(line):
                data[columns[c]].append(attr)

    return Csv(
        columns=columns,
        data=data
    )


def read_csv(path: Path, delimiter=',') -> Csv:
    with open(path) as file:
        return read_csv_reader(file.read(), delimiter=delimiter)
