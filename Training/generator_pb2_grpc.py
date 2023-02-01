# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import generator_pb2 as generator__pb2


class GeneratorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.upload_game = channel.unary_unary(
                '/Proto.Generator/upload_game',
                request_serializer=generator__pb2.Game.SerializeToString,
                response_deserializer=generator__pb2.Response.FromString,
                )
        self.upload_batch = channel.unary_unary(
                '/Proto.Generator/upload_batch',
                request_serializer=generator__pb2.Batch.SerializeToString,
                response_deserializer=generator__pb2.Response.FromString,
                )
        self.get_last_update = channel.unary_unary(
                '/Proto.Generator/get_last_update',
                request_serializer=generator__pb2.Empty.SerializeToString,
                response_deserializer=generator__pb2.LastUpdate.FromString,
                )
        self.get_new_network = channel.unary_unary(
                '/Proto.Generator/get_new_network',
                request_serializer=generator__pb2.Empty.SerializeToString,
                response_deserializer=generator__pb2.Network.FromString,
                )


class GeneratorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def upload_game(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def upload_batch(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_last_update(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_new_network(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GeneratorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'upload_game': grpc.unary_unary_rpc_method_handler(
                    servicer.upload_game,
                    request_deserializer=generator__pb2.Game.FromString,
                    response_serializer=generator__pb2.Response.SerializeToString,
            ),
            'upload_batch': grpc.unary_unary_rpc_method_handler(
                    servicer.upload_batch,
                    request_deserializer=generator__pb2.Batch.FromString,
                    response_serializer=generator__pb2.Response.SerializeToString,
            ),
            'get_last_update': grpc.unary_unary_rpc_method_handler(
                    servicer.get_last_update,
                    request_deserializer=generator__pb2.Empty.FromString,
                    response_serializer=generator__pb2.LastUpdate.SerializeToString,
            ),
            'get_new_network': grpc.unary_unary_rpc_method_handler(
                    servicer.get_new_network,
                    request_deserializer=generator__pb2.Empty.FromString,
                    response_serializer=generator__pb2.Network.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Proto.Generator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Generator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def upload_game(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Proto.Generator/upload_game',
            generator__pb2.Game.SerializeToString,
            generator__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def upload_batch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Proto.Generator/upload_batch',
            generator__pb2.Batch.SerializeToString,
            generator__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_last_update(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Proto.Generator/get_last_update',
            generator__pb2.Empty.SerializeToString,
            generator__pb2.LastUpdate.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_new_network(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Proto.Generator/get_new_network',
            generator__pb2.Empty.SerializeToString,
            generator__pb2.Network.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)