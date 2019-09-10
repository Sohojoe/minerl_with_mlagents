# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from mlagents.envs.communicator_objects.unity_rl_initialization_input_pb2 import (
    UnityRLInitializationInput as mlagents___envs___communicator_objects___unity_rl_initialization_input_pb2___UnityRLInitializationInput,
)

from mlagents.envs.communicator_objects.unity_rl_input_pb2 import (
    UnityRLInput as mlagents___envs___communicator_objects___unity_rl_input_pb2___UnityRLInput,
)

from typing import (
    Optional as typing___Optional,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


class UnityInput(google___protobuf___message___Message):

    @property
    def rl_input(self) -> mlagents___envs___communicator_objects___unity_rl_input_pb2___UnityRLInput: ...

    @property
    def rl_initialization_input(self) -> mlagents___envs___communicator_objects___unity_rl_initialization_input_pb2___UnityRLInitializationInput: ...

    def __init__(self,
        rl_input : typing___Optional[mlagents___envs___communicator_objects___unity_rl_input_pb2___UnityRLInput] = None,
        rl_initialization_input : typing___Optional[mlagents___envs___communicator_objects___unity_rl_initialization_input_pb2___UnityRLInitializationInput] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> UnityInput: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"rl_initialization_input",u"rl_input"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"rl_initialization_input",u"rl_input"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"rl_initialization_input",b"rl_initialization_input",u"rl_input",b"rl_input"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[b"rl_initialization_input",b"rl_input"]) -> None: ...
