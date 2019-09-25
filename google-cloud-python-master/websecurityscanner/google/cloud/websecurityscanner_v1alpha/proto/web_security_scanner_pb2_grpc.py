# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from google.cloud.websecurityscanner_v1alpha.proto import (
    finding_pb2 as google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_finding__pb2,
)
from google.cloud.websecurityscanner_v1alpha.proto import (
    scan_config_pb2 as google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__config__pb2,
)
from google.cloud.websecurityscanner_v1alpha.proto import (
    scan_run_pb2 as google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__run__pb2,
)
from google.cloud.websecurityscanner_v1alpha.proto import (
    web_security_scanner_pb2 as google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2,
)
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


class WebSecurityScannerStub(object):
    """Cloud Web Security Scanner Service identifies security vulnerabilities in web
  applications hosted on Google Cloud Platform. It crawls your application, and
  attempts to exercise as many user inputs and event handlers as possible.
  """

    def __init__(self, channel):
        """Constructor.

    Args:
      channel: A grpc.Channel.
    """
        self.CreateScanConfig = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/CreateScanConfig",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.CreateScanConfigRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__config__pb2.ScanConfig.FromString,
        )
        self.DeleteScanConfig = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/DeleteScanConfig",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.DeleteScanConfigRequest.SerializeToString,
            response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
        )
        self.GetScanConfig = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/GetScanConfig",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.GetScanConfigRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__config__pb2.ScanConfig.FromString,
        )
        self.ListScanConfigs = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/ListScanConfigs",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanConfigsRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanConfigsResponse.FromString,
        )
        self.UpdateScanConfig = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/UpdateScanConfig",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.UpdateScanConfigRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__config__pb2.ScanConfig.FromString,
        )
        self.StartScanRun = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/StartScanRun",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.StartScanRunRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__run__pb2.ScanRun.FromString,
        )
        self.GetScanRun = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/GetScanRun",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.GetScanRunRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__run__pb2.ScanRun.FromString,
        )
        self.ListScanRuns = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/ListScanRuns",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanRunsRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanRunsResponse.FromString,
        )
        self.StopScanRun = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/StopScanRun",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.StopScanRunRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__run__pb2.ScanRun.FromString,
        )
        self.ListCrawledUrls = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/ListCrawledUrls",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListCrawledUrlsRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListCrawledUrlsResponse.FromString,
        )
        self.GetFinding = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/GetFinding",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.GetFindingRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_finding__pb2.Finding.FromString,
        )
        self.ListFindings = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/ListFindings",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingsRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingsResponse.FromString,
        )
        self.ListFindingTypeStats = channel.unary_unary(
            "/google.cloud.websecurityscanner.v1alpha.WebSecurityScanner/ListFindingTypeStats",
            request_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingTypeStatsRequest.SerializeToString,
            response_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingTypeStatsResponse.FromString,
        )


class WebSecurityScannerServicer(object):
    """Cloud Web Security Scanner Service identifies security vulnerabilities in web
  applications hosted on Google Cloud Platform. It crawls your application, and
  attempts to exercise as many user inputs and event handlers as possible.
  """

    def CreateScanConfig(self, request, context):
        """Creates a new ScanConfig.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def DeleteScanConfig(self, request, context):
        """Deletes an existing ScanConfig and its child resources.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetScanConfig(self, request, context):
        """Gets a ScanConfig.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ListScanConfigs(self, request, context):
        """Lists ScanConfigs under a given project.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def UpdateScanConfig(self, request, context):
        """Updates a ScanConfig. This method support partial update of a ScanConfig.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def StartScanRun(self, request, context):
        """Start a ScanRun according to the given ScanConfig.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetScanRun(self, request, context):
        """Gets a ScanRun.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ListScanRuns(self, request, context):
        """Lists ScanRuns under a given ScanConfig, in descending order of ScanRun
    stop time.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def StopScanRun(self, request, context):
        """Stops a ScanRun. The stopped ScanRun is returned.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ListCrawledUrls(self, request, context):
        """List CrawledUrls under a given ScanRun.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetFinding(self, request, context):
        """Gets a Finding.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ListFindings(self, request, context):
        """List Findings under a given ScanRun.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ListFindingTypeStats(self, request, context):
        """List all FindingTypeStats under a given ScanRun.
    """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_WebSecurityScannerServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "CreateScanConfig": grpc.unary_unary_rpc_method_handler(
            servicer.CreateScanConfig,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.CreateScanConfigRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__config__pb2.ScanConfig.SerializeToString,
        ),
        "DeleteScanConfig": grpc.unary_unary_rpc_method_handler(
            servicer.DeleteScanConfig,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.DeleteScanConfigRequest.FromString,
            response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
        ),
        "GetScanConfig": grpc.unary_unary_rpc_method_handler(
            servicer.GetScanConfig,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.GetScanConfigRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__config__pb2.ScanConfig.SerializeToString,
        ),
        "ListScanConfigs": grpc.unary_unary_rpc_method_handler(
            servicer.ListScanConfigs,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanConfigsRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanConfigsResponse.SerializeToString,
        ),
        "UpdateScanConfig": grpc.unary_unary_rpc_method_handler(
            servicer.UpdateScanConfig,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.UpdateScanConfigRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__config__pb2.ScanConfig.SerializeToString,
        ),
        "StartScanRun": grpc.unary_unary_rpc_method_handler(
            servicer.StartScanRun,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.StartScanRunRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__run__pb2.ScanRun.SerializeToString,
        ),
        "GetScanRun": grpc.unary_unary_rpc_method_handler(
            servicer.GetScanRun,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.GetScanRunRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__run__pb2.ScanRun.SerializeToString,
        ),
        "ListScanRuns": grpc.unary_unary_rpc_method_handler(
            servicer.ListScanRuns,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanRunsRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListScanRunsResponse.SerializeToString,
        ),
        "StopScanRun": grpc.unary_unary_rpc_method_handler(
            servicer.StopScanRun,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.StopScanRunRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_scan__run__pb2.ScanRun.SerializeToString,
        ),
        "ListCrawledUrls": grpc.unary_unary_rpc_method_handler(
            servicer.ListCrawledUrls,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListCrawledUrlsRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListCrawledUrlsResponse.SerializeToString,
        ),
        "GetFinding": grpc.unary_unary_rpc_method_handler(
            servicer.GetFinding,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.GetFindingRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_finding__pb2.Finding.SerializeToString,
        ),
        "ListFindings": grpc.unary_unary_rpc_method_handler(
            servicer.ListFindings,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingsRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingsResponse.SerializeToString,
        ),
        "ListFindingTypeStats": grpc.unary_unary_rpc_method_handler(
            servicer.ListFindingTypeStats,
            request_deserializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingTypeStatsRequest.FromString,
            response_serializer=google_dot_cloud_dot_websecurityscanner__v1alpha_dot_proto_dot_web__security__scanner__pb2.ListFindingTypeStatsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "google.cloud.websecurityscanner.v1alpha.WebSecurityScanner",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
