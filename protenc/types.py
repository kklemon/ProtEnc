from typing import Literal, Optional, Union


ProteinEncoderInput = Union[list[str], dict[str, str]]
BatchSize = Optional[Union[Literal['auto'], int]]
ReturnFormat = Literal['torch', 'numpy']
