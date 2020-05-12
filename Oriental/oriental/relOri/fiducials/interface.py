# -*- coding: cp1252 -*-

from oriental.adjust.parameters import ADP
from oriental.ori.transform import AffineTransform2D

from abc import abstractmethod
from contracts import contract, ContractsMeta

# Note: @abstractmethod checks only if a function of that name has been implemented, not if it has the right number of arguments.
#       @contract with metaclass=ContractsMeta only checks the contracts that are defined in the abstract base class'es method
#                 -> methods in a derived class may define additional arguments;
#                    Their values are only checked if a contract for them is defined, and if the overriding method itself is decorated with @contract
class IFiducialDetector(metaclass=ContractsMeta):

    @abstractmethod
    @contract
    def __call__( self,
                  imgFn : str,
                  camSerial : 'str|None' = None, 
                  filmFormatFocal : 'seq[3](number,>0)|None' = None,
                  plotDir : str = '',
                  debugPlot : bool = False
               ) -> 'tuple( array[3](float), ADP, ITransform2D, array[Nx2](float), bool, float )':
        "Compute the fiducial transformation of the passed image"
        pass
