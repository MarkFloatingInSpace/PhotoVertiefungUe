"""
Import a block from Inpho Match-AT
"""

"""
Accompanying / side-car files:
   - .cnt-files document the settings used in the most recent adjustment.
   - .xpf-files store the automatically determined image point observations most recently used.
      The origin of their coordinate system is in the center of the top/left pixel.
$BLOCK defines a grouping of images, used for selectively processing a subset of all images - e.g. matching.
"""
import datetime, sqlite3
from collections import deque, namedtuple
from pathlib import Path

import numpy as np

from oriental import log

logger = log.Logger('importMatchAT')

class LineBuffer:
    def __init__(self, fileLike):
        self._fileLike = fileLike
        self._lines = deque()
    def next(self):
        if not self._lines:
            self._lines.extend(self._fileLike.readlines(1024))
        if self._lines:
            return self._lines[0]
        return None
    def pop(self):
        self._lines.popleft()

def readSecondLevelItem(lineBuffer):
    while True:
        line = lineBuffer.next()
        if not line:
            raise Exception('Unexpected end of file encountered.')
        if line.startswith('$'): # top-level item follows.
            return None, None
        lineBuffer.pop()
        if not line.lstrip().startswith('#'):
            break
    tagValue = line.split(':', maxsplit=1)
    if len(tagValue) == 1:
        # There are second-level items without a value:
        # $END_STRIPS, which ends $STRIPS.
        # $END_POINTS, which ends $PHOTO_FOOTPRINT
        tag = tagValue[0].strip()[1:]
        assert tag in {'END_STRIPS', 'END_POINTS'}
        return tag, None
    tag = tagValue[0].strip()[1:]
    value = tagValue[1].replace(r'\ ', ' ')
    while True:
        line = lineBuffer.next()
        if not line:
            raise Exception('Unexpected end of file encountered.')
        if line.lstrip().startswith('$'):
            break
        lineBuffer.pop()
        value += line.replace(r'\ ', ' ')
    return tag, value.strip()

def readTopLevelItem(lineBuffer, valueConverters = {}):
    res = {}
    while True:
        tag, value = readSecondLevelItem(lineBuffer)
        if tag is None:
            line = lineBuffer.next().rstrip()
            if line != '$END':
                raise Exception('$END expected')
            lineBuffer.pop()
            break
        valueConverter = valueConverters.get(tag)
        if valueConverter is not None:
            value = valueConverter(value)
        res[tag] = value
    return res


def readTopLevelItemNonUnique(tag, res, idName, lineBuffer, valueConverters = {}):
    values = readTopLevelItem(lineBuffer, valueConverters)
    key = values.pop(idName).strip()
    dic = res.setdefault(tag, {})
    assert key not in dic
    dic[key] = values

def readCamera(res, lineBuffer):
    pass

def convAATStrips(value):
    res = {}
    stripId = None
    stationIDs = []
    for line in value.split('\n'):
        words = line.split()
        if not line.startswith('          '):
            if stationIDs:
                assert stripId is not None
                res[stripId] = stationIDs
                stripId = None
                stationIDs.clear()
            stripId = int(words[0])
            assert words[1] == 'ElementStation'
            assert words[7] == '{'
            del words[:8]

        for word in words:
            if word == '}':
                break
            stationIDs.append(word)

    if stationIDs:
        assert stripId is not None
        res[stripId] = stationIDs

    return res

def convImageList(value):
    words = value.split()
    assert words[0] == '{'
    imgFns = []
    for word in words[1:]:
        if word == '}':
            break
        imgFns.append(word)

    return imgFns

def convExtOri(value):
    words = value.split()
    assert len(words) == 15
    dt = datetime.datetime.strptime(' '.join(words[:2]), '%H:%M:%S %d/%m/%Y')
    return {
        'datetime' : dt,
        'P0' : np.array(words[3:6], float),
        'R'  : np.array(words[6:], float).reshape((3,3))}

def importMatchAT( prjFn : Path, db : sqlite3.Connection, dbDir : Path ):
    res = {}
    with open(prjFn) as prj:
        lineBuffer = LineBuffer(prj)
        while True:
            line = lineBuffer.next()
            if line is None:
                break
            lineBuffer.pop()
            tag = line.split(maxsplit=1)[0][1:]
            values = None
            if tag == 'AAT':
                values = readTopLevelItem(lineBuffer, {'STRIPS' : convAATStrips})
            elif tag == 'BLOCK':
                readTopLevelItemNonUnique(tag, res, 'BLOCK_ID', lineBuffer, {'IMAGES' : convImageList})
                continue
            elif tag == 'STATION':
                readTopLevelItemNonUnique(tag, res, 'STATION_ID', lineBuffer, {'IMAGES': convImageList})
                continue
            elif tag == 'PHOTO':
                readTopLevelItemNonUnique(tag, res, 'PHOTO_NUM', lineBuffer, {'EXT_ORI': convExtOri})
                continue
            elif tag == 'CAMERA_DEFINITION':
                readCamera(res, lineBuffer)
                continue
            else:
                values = readTopLevelItem(lineBuffer)
            assert values is not None
            assert tag not in res
            res[tag] = values
