# -*- coding: cp1252 -*-
"""Accumulate in a CSV-file tagged output found in a log file.

Parse a log file, use the last OrientAL run it contains, and save the tagged information in the CSV file.
Additionally, output any warnings and errors.
"""

import _prolog
from oriental.utils.argparse import Formatter

import os
import argparse
import csv
import xml.sax
import re

class ContentHandler( xml.sax.ContentHandler ):
    def __init__( self ):
        self._reset()

    def _reset( self ):
        self.tags = []
        self.errors = []
        self.warnings = []
        self.currentTag = None
        self.errWrng = 0

    def startElement( self, name, attrs ):
        if name == 'Run': # we use information from the last run only.
            self._reset()
        elif name == 'Record':
            if 'Tag' in attrs:
                self.currentTag = attrs['Tag']
                self.tags.append( [ self.currentTag, ''] )
            elif attrs['Severity'] == 'error':
                self.errWrng = 1
                self.errors.append('')
            elif attrs['Severity'] == 'warning':
                self.errWrng = 2
                self.warnings.append('')

    def characters( self, content ):
        if self.currentTag:
            self.tags[-1][1] = self.tags[-1][1] + content
        elif self.errWrng:
            theList = [ self.errors, self.warnings ][self.errWrng-1]
            theList[-1] = theList[-1] + content

    def endElement( self, name ):
        if name == 'Record':
            self.currentTag = None
            self.errWrng = 0

class LegacyContentHandler( xml.sax.ContentHandler ):
    def __init__( self ):
        self._reset()

    def setDocumentLocator( self, locator ):
        self.locator = locator

    def _reset( self ):
        self.tags = []
        self.errors = []
        self.warnings = []
        self.content = None
        self.errWrng = 0
        self.nFeatures = 0

    def startElement( self, name, attrs ):
        if name == 'Run': # we use information from the last run only.
            self._reset()
        elif name == 'Record':
            if attrs['Severity'] == 'error':
                self.errWrng = 1
            elif attrs['Severity'] == 'warning':
                self.errWrng = 2
            self.content = ''

    def characters( self, content ):
        if self.content is None:
            return
        self.content = self.content + content

    def endElement( self, name ):
        if name != 'Record':
            if self.content:
                self.content = self.content + ' '
            return
        if self.errWrng == 0:
            self._parseContent()
        elif self.errWrng == 1:
            self.errors.append( self.content )
        elif self.errWrng == 2:
            self.warnings.append( self.content )
        self.content = None
        self.errWrng = 0

    def _parseContent( self ):
        if self.content.startswith('commandline:'):
            match = re.match( r'.*relOri\.py (?P<commandline>.*)', self.content )
            value = match.group('commandline')
            assert value
            self.tags.append( [ 'commandline', value ] )
        elif self.content.startswith('Match '):
            match = re.match( r'Match (?P<images>\d+)', self.content )
            if match:
                self.tags.append( ['#images', int(match.group('images')) ] )
        elif self.content.startswith('Detected features per cell for '):
            mSlice = re.match( r'.*?keep \d+ max.\)(?P<slice>.*)', self.content )
            self.content = mSlice.group('slice')
            numbers = re.finditer( r'\s*(?P<featuresTotal>\d+)', self.content )
            self.nFeatures += sum(( int(el.group('featuresTotal')) for el in numbers ))
        elif self.content.startswith('initial relative orientation of phos'):
            match = re.match( r'^initial relative orientation of phos (?P<pho1>\d+) and (?P<pho2>\d+)$', self.content )
            self.tags.append( ['initial image pair', '{} - {}'.format( int(match.group('pho1')), match.group('pho2') ) ] )
        elif self.content.startswith('block consists of '):
            match = re.match(r'^block consists of (?P<orientedImgs>\d+) phos, (?P<imgPtObs>\d+) imgPts, and (?P<objPts>\d+) objPts$', self.content)
            for name in ('orientedImgs', 'imgPtObs', 'objPts' ):
                self.tags.append( ['#' + name, int(match.group(name)) ] )
        elif self.content.startswith('redundancy: '):
            match = re.match( r'redundancy: (?P<redundancy>\d+)', self.content )
            self.tags.append( ['redundancy', int(match.group('redundancy')) ] )
        elif self.content.startswith('Sigma_0: '):
            match = re.match( r'Sigma_0: (?P<Sigma_0>.+)', self.content )
            self.tags.append( ['Sigma_0', float(match.group('Sigma_0')) ] )
        elif self.content.startswith('Processing time: '):
            match = re.match( r'Processing time: (?P<processingTime>.+)', self.content )
            self.tags.append( [ 'Total time', match.group('processingTime') ] )

    def endDocument( self ):
        d = { key:value for key,value in self.tags }
        if all( (name in d for name in ('#images','#orientedImgs')) ):
            self.tags.append(  [ '#unorientedImgs', d['#images'] - d['#orientedImgs'] ] )
        self.tags.append( ['#FeaturesTotal', self.nFeatures ] )


if __name__ == '__main__':
    docList = __doc__.splitlines()
    parser = argparse.ArgumentParser( description=docList[0],
                                      epilog='\n'.join(docList[1:]),
                                      formatter_class=Formatter )

    parser.add_argument( '--log', default=os.path.join( os.getcwd(), 'relOri', 'relOriLog.xml' ),
                         help='Log file to be processed' )
    parser.add_argument( '--csv', default=os.path.join( os.getcwd(), 'logStats.csv' ),
                         help='Store results into CSV' )
    parser.add_argument('--legacy', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    handler = ContentHandler() if not args.legacy else LegacyContentHandler()
    xml.sax.parse( args.log, handler )

    newFieldNames = [ el[0] for el in handler.tags ]

    # If during the generation of the first log to be processed, the script exited early due to an error, then, we won't have all tags in the CSV header.
    # csv offers no option to edit files. Thus, let's re-write the whole CSV-file if additional field names must be introduced.
    data = []
    fieldNames = newFieldNames
    reWrite = True
    try:
        with open( args.csv, 'r', newline='' ) as csvFile:
            reader = csv.DictReader( csvFile, dialect='excel' )
            fieldNames = [ el for el in reader.fieldnames if el not in ( 'errors', 'warnings' ) ]
            reWrite = set(fieldNames) < set(newFieldNames)
            if reWrite:
                prefix, suffix = fieldNames, newFieldNames
                if len(prefix) < len(suffix):
                    prefix, suffix = suffix, prefix
                suffix = [ el for el in suffix if el not in prefix ]
                fieldNames = prefix + suffix
                data = [ row for row in reader ]
    except OSError:
        pass

    # errors & warnings may be lengthy. Make sure that they appear rightmost in the CSV-file.
    fieldNames += [ 'errors', 'warnings' ]
    handler.tags.append( [ 'errors', ' | '.join( handler.errors ) ] )
    handler.tags.append( [ 'warnings', ' | '.join( handler.warnings ) ] )

    with open( args.csv, 'w' if reWrite else 'a', newline='' ) as csvFile:
        writer = csv.DictWriter( csvFile, fieldNames, dialect='excel' )
        if reWrite:
            writer.writeheader()
        data.append( { key:value for key,value in handler.tags } )
        writer.writerows( data  )
