"""Patching ADO Client to retrieve continuation token.

Related to question in following issue:
https://github.com/microsoft/azure-devops-python-api/issues/461

"""

import logging
from typing import Optional, cast

from azure.devops import _models
from azure.devops.client import Client
from azure.devops.client_configuration import ClientConfiguration
from msrest import Deserializer, Serializer
from msrest.service_client import ServiceClient

logger = logging.getLogger("azure.devops.client")


# pylint: disable=super-init-not-called


class ClientPatch(Client):
    """Client.

    :param str base_url: Service URL.
    :param Authentication creds: Authenticated credentials.
    """

    def __init__(self, base_url=None, creds=None):
        self.config = ClientConfiguration(base_url)
        self.config.credentials = creds
        self._client = ServiceClient(creds, config=self.config)
        _base_client_models = {k: v for k, v in _models.__dict__.items() if isinstance(v, type)}
        self._base_deserialize = Deserializer(_base_client_models)
        self._base_serialize = Serializer(_base_client_models)
        self._all_host_types_locations = {}
        self._locations = {}
        self._suppress_fedauth_redirect = True
        self._force_msa_pass_through = True
        self.normalized_url = Client._normalize_url(base_url)
        self.continuation_token_last_request: Optional[str] = None

    def _send(
        self,
        http_method,
        location_id,
        version,
        route_values=None,
        query_parameters=None,
        content=None,
        media_type="application/json",
        accept_media_type="application/json",
        additional_headers=None,
    ):
        request = self._create_request_message(
            http_method=http_method,
            location_id=location_id,
            route_values=route_values,
            query_parameters=query_parameters,
        )
        negotiated_version = self._negotiate_request_version(
            self._get_resource_location(self.normalized_url, location_id), version
        )
        negotiated_version = cast("str", negotiated_version)

        if version != negotiated_version:
            logger.info(
                "Negotiated api version from '%s' down to '%s'. This means the client is newer than the server.",
                version,
                negotiated_version,
            )
        else:
            logger.debug("Api version '%s'", negotiated_version)

        # Construct headers
        headers = {
            "Content-Type": media_type + "; charset=utf-8",
            "Accept": accept_media_type + ";api-version=" + negotiated_version,
        }
        if additional_headers is not None:
            for key in additional_headers:
                headers[key] = str(additional_headers[key])
        if self.config.additional_headers is not None:
            for key in self.config.additional_headers:
                headers[key] = self.config.additional_headers[key]
        if self._suppress_fedauth_redirect:
            headers["X-TFS-FedAuthRedirect"] = "Suppress"
        if self._force_msa_pass_through:
            headers["X-VSS-ForceMsaPassThrough"] = "true"
        if Client._session_header_key in Client._session_data and Client._session_header_key not in headers:
            headers[Client._session_header_key] = Client._session_data[Client._session_header_key]
        response = self._send_request(request=request, headers=headers, content=content, media_type=media_type)
        if Client._session_header_key in response.headers:
            Client._session_data[Client._session_header_key] = response.headers[Client._session_header_key]

        # Patch: Workaround to be able to see the continuation token of the response
        self.continuation_token_last_request = self._get_continuation_token(response)

        return response


def patch_azure_devops_client():
    """Patch the Azure DevOps client to see the continuation token of the response."""
    # pylint: disable=protected-access
    Client.__init__ = ClientPatch.__init__  # ruff: noqa: PGH003
    Client._send = ClientPatch._send  # ruff: noqa: PGH003
