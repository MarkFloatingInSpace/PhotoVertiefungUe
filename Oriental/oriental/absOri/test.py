class LocalScope:
    def __enter__(self):
        self.globalsBefore = set( globals().keys() )
        return self

    def __exit__(self, *exc):
        diff = set( globals().keys() ).difference( self.globalsBefore )
        for el in diff:
            del globals()[el]


a = 1
with LocalScope():
    b = 2

print(globals())

diff = 1

import contextlib
@contextlib.contextmanager
def localScope():
    globalsBefore = set( globals().keys() )
    yield
    diff = set( globals().keys() ).difference( globalsBefore )
    for el in diff:
        del globals()[el]

with localScope():
    c = 1
    diff = 2

print(globals())
