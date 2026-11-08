"""Bounded-memory index of native CSV geometry; no coordinate sampling or decimation."""
from __future__ import annotations
import csv
from pathlib import Path
from tools.vtc_growth_io import require, number


class CsvRows:
    def __init__(self, path: Path):
        self.path=path

    def __iter__(self):
        with self.path.open(encoding='utf-8-sig',newline='') as stream:
            yield from csv.DictReader(stream)


class MeshIndex:
    def __init__(self, path: Path):
        self.path=path;self.blocks={};self.cache_day=None;self.cache={}
        with path.open('rb') as stream:
            self.header=stream.readline().decode('utf-8-sig').strip().split(',')
            previous=None
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                day=int(line.split(b',',1)[0])
                if day!=previous:
                    require(day not in self.blocks,'noncontiguous native geometry state')
                    self.blocks[day]=[offset,0];previous=day
                self.blocks[day][1]+=1

    def state(self, day: int) -> dict:
        if self.cache_day==day:return self.cache
        result={}
        if day in self.blocks:
            offset,count=self.blocks[day]
            with self.path.open('rb') as stream:
                stream.seek(offset)
                for _ in range(count):
                    row=dict(zip(self.header,next(csv.reader([stream.readline().decode('utf-8')]))))
                    key=(day,row['organ_id'],int(row['vertex']))
                    require(key not in result,'duplicate mesh vertex')
                    result[key]=tuple(number(row[k],k) for k in ('x_m','y_m','z_m'))
        self.cache_day=day;self.cache=result
        return result

    def __getitem__(self,key):return self.state(key[0])[key]
    def __contains__(self,key):return key in self.state(key[0])
