# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: faiss.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='faiss.proto',
  package='faiss',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0b\x66\x61iss.proto\x12\x05\x66\x61iss\"\x1b\n\x06Vector\x12\x11\n\tfloat_val\x18\x05 \x03(\x02\"=\n\rSearchRequest\x12\x1d\n\x06vector\x18\x01 \x01(\x0b\x32\r.faiss.Vector\x12\r\n\x05top_k\x18\x02 \x01(\x04\"%\n\x08Neighbor\x12\n\n\x02id\x18\x01 \x01(\x04\x12\r\n\x05score\x18\x02 \x01(\x02\"4\n\x0eSearchResponse\x12\"\n\tneighbors\x18\x02 \x03(\x0b\x32\x0f.faiss.Neighbor\".\n\x11SearchByIdRequest\x12\n\n\x02id\x18\x01 \x01(\x04\x12\r\n\x05top_k\x18\x02 \x01(\x04\"L\n\x12SearchByIdResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x04\x12\"\n\tneighbors\x18\x02 \x03(\x0b\x32\x0f.faiss.Neighbor2\x88\x01\n\x0c\x46\x61issService\x12\x35\n\x06Search\x12\x14.faiss.SearchRequest\x1a\x15.faiss.SearchResponse\x12\x41\n\nSearchById\x12\x18.faiss.SearchByIdRequest\x1a\x19.faiss.SearchByIdResponseb\x06proto3'
)




_VECTOR = _descriptor.Descriptor(
  name='Vector',
  full_name='faiss.Vector',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='float_val', full_name='faiss.Vector.float_val', index=0,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=49,
)


_SEARCHREQUEST = _descriptor.Descriptor(
  name='SearchRequest',
  full_name='faiss.SearchRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='vector', full_name='faiss.SearchRequest.vector', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='top_k', full_name='faiss.SearchRequest.top_k', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=51,
  serialized_end=112,
)


_NEIGHBOR = _descriptor.Descriptor(
  name='Neighbor',
  full_name='faiss.Neighbor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='faiss.Neighbor.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='score', full_name='faiss.Neighbor.score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=114,
  serialized_end=151,
)


_SEARCHRESPONSE = _descriptor.Descriptor(
  name='SearchResponse',
  full_name='faiss.SearchResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='neighbors', full_name='faiss.SearchResponse.neighbors', index=0,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=153,
  serialized_end=205,
)


_SEARCHBYIDREQUEST = _descriptor.Descriptor(
  name='SearchByIdRequest',
  full_name='faiss.SearchByIdRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='faiss.SearchByIdRequest.id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='top_k', full_name='faiss.SearchByIdRequest.top_k', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=207,
  serialized_end=253,
)


_SEARCHBYIDRESPONSE = _descriptor.Descriptor(
  name='SearchByIdResponse',
  full_name='faiss.SearchByIdResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_id', full_name='faiss.SearchByIdResponse.request_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='neighbors', full_name='faiss.SearchByIdResponse.neighbors', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=255,
  serialized_end=331,
)

_SEARCHREQUEST.fields_by_name['vector'].message_type = _VECTOR
_SEARCHRESPONSE.fields_by_name['neighbors'].message_type = _NEIGHBOR
_SEARCHBYIDRESPONSE.fields_by_name['neighbors'].message_type = _NEIGHBOR
DESCRIPTOR.message_types_by_name['Vector'] = _VECTOR
DESCRIPTOR.message_types_by_name['SearchRequest'] = _SEARCHREQUEST
DESCRIPTOR.message_types_by_name['Neighbor'] = _NEIGHBOR
DESCRIPTOR.message_types_by_name['SearchResponse'] = _SEARCHRESPONSE
DESCRIPTOR.message_types_by_name['SearchByIdRequest'] = _SEARCHBYIDREQUEST
DESCRIPTOR.message_types_by_name['SearchByIdResponse'] = _SEARCHBYIDRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Vector = _reflection.GeneratedProtocolMessageType('Vector', (_message.Message,), {
  'DESCRIPTOR' : _VECTOR,
  '__module__' : 'faiss_pb2'
  # @@protoc_insertion_point(class_scope:faiss.Vector)
  })
_sym_db.RegisterMessage(Vector)

SearchRequest = _reflection.GeneratedProtocolMessageType('SearchRequest', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHREQUEST,
  '__module__' : 'faiss_pb2'
  # @@protoc_insertion_point(class_scope:faiss.SearchRequest)
  })
_sym_db.RegisterMessage(SearchRequest)

Neighbor = _reflection.GeneratedProtocolMessageType('Neighbor', (_message.Message,), {
  'DESCRIPTOR' : _NEIGHBOR,
  '__module__' : 'faiss_pb2'
  # @@protoc_insertion_point(class_scope:faiss.Neighbor)
  })
_sym_db.RegisterMessage(Neighbor)

SearchResponse = _reflection.GeneratedProtocolMessageType('SearchResponse', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHRESPONSE,
  '__module__' : 'faiss_pb2'
  # @@protoc_insertion_point(class_scope:faiss.SearchResponse)
  })
_sym_db.RegisterMessage(SearchResponse)

SearchByIdRequest = _reflection.GeneratedProtocolMessageType('SearchByIdRequest', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHBYIDREQUEST,
  '__module__' : 'faiss_pb2'
  # @@protoc_insertion_point(class_scope:faiss.SearchByIdRequest)
  })
_sym_db.RegisterMessage(SearchByIdRequest)

SearchByIdResponse = _reflection.GeneratedProtocolMessageType('SearchByIdResponse', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHBYIDRESPONSE,
  '__module__' : 'faiss_pb2'
  # @@protoc_insertion_point(class_scope:faiss.SearchByIdResponse)
  })
_sym_db.RegisterMessage(SearchByIdResponse)



_FAISSSERVICE = _descriptor.ServiceDescriptor(
  name='FaissService',
  full_name='faiss.FaissService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=334,
  serialized_end=470,
  methods=[
  _descriptor.MethodDescriptor(
    name='Search',
    full_name='faiss.FaissService.Search',
    index=0,
    containing_service=None,
    input_type=_SEARCHREQUEST,
    output_type=_SEARCHRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SearchById',
    full_name='faiss.FaissService.SearchById',
    index=1,
    containing_service=None,
    input_type=_SEARCHBYIDREQUEST,
    output_type=_SEARCHBYIDRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FAISSSERVICE)

DESCRIPTOR.services_by_name['FaissService'] = _FAISSSERVICE

# @@protoc_insertion_point(module_scope)
