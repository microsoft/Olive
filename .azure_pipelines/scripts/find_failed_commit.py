# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Script to run a build against each commit from a failed CI build."""

# ruff: noqa: T201
# ruff: noqa: T203

import argparse
import os
import sys
import time
from pprint import pprint
from typing import TYPE_CHECKING, List

import client_patch
from azure.devops.connection import Connection
from azure.devops.v7_1.pipelines.models import (
    RepositoryResourceParameters,
    RunPipelineParameters,
    RunResourcesParameters,
)
from msrest.authentication import BasicAuthentication

if TYPE_CHECKING:
    from azure.devops.v7_1.build.build_client import BuildClient
    from azure.devops.v7_1.build.models import Build, Change
    from azure.devops.v7_1.pipelines.models import Run
    from azure.devops.v7_1.pipelines.pipelines_client import PipelinesClient


# DevOps API - https://github.com/Microsoft/azure-devops-python-api
# DevOps Samples - https://github.com/microsoft/azure-devops-python-samples/tree/main
# Access Token - https://learn.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Windows

_organization_url = "https://dev.azure.com/aiinfra/"
_polling_delay = 5 * 60


def _main():
    access_token = os.getenv("DEVOPS_ACCESS_TOKEN")
    if not access_token:
        raise ValueError(
            "Set DEVOPS_ACCESS_TOKEN in the environment. For instructions on how to get one, visit "
            "https://learn.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Windows"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", default="PublicPackages", type=str, help="Project name")
    parser.add_argument("--build-id", required=True, type=str, help="Build Id")
    args = parser.parse_args()

    # Create a connection to the org
    credentials = BasicAuthentication("", access_token)
    connection = Connection(base_url=_organization_url, creds=credentials)

    # Ref: https://github.com/microsoft/azure-devops-python-api/issues/461
    client_patch.patch_azure_devops_client()

    # Get build definition
    build_client: BuildClient = connection.clients.get_build_client()
    build: Build = build_client.get_build(project=args.project_name, build_id=args.build_id)
    pprint({"build": build.as_dict()})

    if build.result != "failed":
        raise ValueError(f"Input build-id={args.build_id} is not a failed build!")

    # Get all the changes associated with the build
    changes: List[Change] = []
    while True:
        changes.extend(
            build_client.get_build_changes(
                project=args.project_name,
                build_id=args.build_id,
                continuation_token=build_client.continuation_token_last_request,
                include_source_change=True,
            )
        )

        if not build_client.continuation_token_last_request:
            break

    print(f"# of commits: {len(changes)}")

    # Run a build against each change
    for change in reversed(changes):
        run_parameters = RunPipelineParameters(
            preview_run=False,
            resources=RunResourcesParameters(
                repositories={
                    "self": RepositoryResourceParameters(
                        ref_name="refs/heads/main",
                        version=change.id,
                    )
                }
            ),
        )
        pipelines_client: PipelinesClient = connection.clients.get_pipelines_client()
        run: Run = pipelines_client.run_pipeline(
            run_parameters=run_parameters, project=args.project_name, pipeline_id=build.definition.id
        )
        pprint({"run": run.as_dict()})

        # Wait for the build to finish
        while True:
            build: Build = build_client.get_build(project=args.project_name, build_id=run.id)
            if build.status == "completed":
                if build.result == "failed":
                    print("=" * 60)
                    print("Found the following failed commit:")
                    print(f"    Author: {change.author.display_name}")
                    print(f"   Message: {change.message}")
                    print(f"Commit SHA: {change.id}")
                elif build.result == "canceled":
                    raise RuntimeError(f"Last launched build={run.id} was canceled.")
                break
            time.sleep(_polling_delay)

    return 0


if __name__ == "__main__":
    sys.exit(_main())
