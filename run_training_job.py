#!/usr/bin/env python3

import os

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

subscription_id = os.getenv(
    "AZURE_SUBSCRIPTION_ID", "485eb7ad-2bcd-4928-92b0-7902804178aa"
)
resource_group = os.getenv("AZURE_RESOURCE_GROUP", "rg-whisper-aml")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME", "mlw-whisper")

compute_name = "gpu-cluster-nc6"
environment_name = "whisper-finetune-env"
environment_version = "5"
conda_file_path = "./environment/conda_env.yml"

code_dir = "./src"
training_script_name = "train_whisper.py"

experiment_name = "whisper-malayalam-finetune"
job_display_name = "whisper-small-mal-finetune-run"


def main():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    try:
        custom_env = ml_client.environments.get(
            name=environment_name, version=environment_version
        )
        print(f"Found existing environment '{environment_name}:{environment_version}'.")
    except:
        print(
            f"Environment '{environment_name}:{environment_version}' not found. Creating it."
        )
        custom_env = Environment(
            name=environment_name,
            version=environment_version,
            description="Environment for Whisper fine-tuning",
            conda_file=conda_file_path,
            image="mcr.microsoft.com/azureml/curated/designer-pytorch-2.3-train:10",
        )
        custom_env = ml_client.environments.create_or_update(custom_env)
        print(f"Environment '{custom_env.name}:{custom_env.version}' created.")

    job_command = f"python {training_script_name}"
    job = command(
        code=code_dir,
        command=job_command,
        environment=f"{custom_env.name}:{custom_env.version}",
        compute=compute_name,
        experiment_name=experiment_name,
        display_name=job_display_name,
    )

    print(
        f"Submitting job '{job.display_name}' to experiment '{job.experiment_name}'..."
    )
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted. Job ID: {returned_job.name}")
    print(f"View in Azure ML Studio: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
