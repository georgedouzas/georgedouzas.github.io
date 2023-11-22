# %% [markdown]
# ---
# title: "Prefect"
# description: "Build, deploy and observe data workflows."
# author: "Georgios Douzas"
# date: "2023-06-022"
# categories: [Software Engineering, Open Source, Data Engineering]
# image: "featured.png"
# jupyter: python3
# ---

# %% [markdown]
"""
![](featured.png)

## Introduction

[Prefect](https://docs.prefect.io) is a workflow orchestration tool. It makes accessible the creation, scheduling, and monitoring
of complex data pipelines. The workflows are defined as Python code, while Prefect provides error handling, retry mechanisms, and
a user-friendly dashboard for monitoring. Prefect is based on the following concepts:

- **Tasks**: Functions that represent a discrete unit of work in a Prefect workflow.
- **Flows**: Containers for workflow logic and allow users to interact with and reason about the state of their workflows.
- **Results**: They represent the data returned by a flow or a task. 
- **Artifacts**: They are formatted outputs rendered in the Prefect UI, such as markdown, tables, or links.
- **States**: They represent the status of a particular task run or flow run.
- **Task Runners**: They allow selecting specific executors for Prefect tasks, such as concurrent, parallel, or distributed
execution of tasks.
- **Runtime Context**: It provides information about the current flow or task run that you can refer to in your code.
- **Profiles and Configuration**: They are settings that can be used to interact with Prefect Cloud and a Prefect server.
- **Blocks**: Prefect primitives that enable the storage of configuration and provide a UI interface.
- **Variables**: They are named, mutable string values, much like environment variables.
- **Deployments**: They are server-side concepts that encapsulate flows, allowing them to be scheduled and triggered via API.
- **Deployment Management**: A set of files that describe how to prepare one or more flow deployments.
- **Work Pools, Workers & Agents**	: They bridge the Prefect orchestration environment with the execution environment.
- **Storage**: It configures how flow code for deployments is persisted and retrieved by Prefect agents.
- **Filesystems**: They are blocks that allow to read and write data from paths.
- **Infrastructure**: They are blocks that specify infrastructure for flow runs created by the deployment.
- **Schedules**	: They define how to create new flow runs automatically on a specified cadence.
- **Logging**: They log useful information about flows and tasks runs on the server.

## Extracting the URLs

The goal is to create a data workflow that downloads soccer data from [Football-Data.co.uk](https://www.football-data.co.uk). The
URL of each of those main leagues has the form `'https://www.football-data.co.uk/mmz4281/{season}/{league_id}.csv'` where `season`
is the season of the league and `league_id` is the league ID. Let's define a few of those seasons and leagues:
"""

# %%
SEASONS = [
    '1213',
    '1314',
    '1516',
    '1617',
    '1718',
    '1819',
    '1920',
    '2021',
    '2122',
    '2223',
]
LEAGUES_MAPPING = {
    'E0': 'English',
    'SC0': 'Scotish',
    'D1': 'German',
    'I1': 'Italian',
    'SP1': 'Spanish',
    'F1': 'French',
    'N1': 'Dutch',
    'B1': 'Belgian',
    'P1': 'Portuguese',
    'T1': 'Turkish',
    'G1': 'Greek',
}

# %% [markdown]
"""
We can use the above seasons and leagues to construct a mapping of (URL, league name and season) pairs:
"""

# %%
URLS_MAPPING = {
    f'https://www.football-data.co.uk/mmz4281/{season}/{league_id}.csv': (
        league,
        '-'.join([season[0:2], season[2:]]),
    )
    for season in SEASONS
    for league_id, league in LEAGUES_MAPPING.items()
}
URLS_MAPPING

# %% [markdown]
"""
We will use the above URLs to download and extract the data into a single dataframe. Additionally, the following imports will be
required:
"""

# %%
from time import time
import httpx
import asyncio
import pandas as pd
from io import StringIO
from prefect import flow
from prefect import task
from prefect.logging import get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

# %% [markdown]
"""
## Just Python functions

The simplest approach to implement the data workflow is not to use Prefect and rely on Python functions. Let's start by defining
the three following functions:
"""


# %%
def request_csv_data(client, url, **kwargs):
    response = client.get(url=url)
    return response


def download_csvs_data(urls_mapping):
    with httpx.Client() as client:
        responses = [
            request_csv_data(client, url)
            for url, (league, season) in urls_mapping.items()
        ]
    csvs_data = [
        StringIO(str(response.content, encoding='windows-1254'))
        for response in responses
    ]
    return csvs_data


def download_data(urls_mapping):
    csvs_data = download_csvs_data(urls_mapping)
    data = [pd.read_csv(csv_data, encoding='windows-1254') for csv_data in csvs_data]
    data = pd.concat(data, ignore_index=True)
    return data


# %% [markdown]
"""
- `request_csv_data` will use the parameters `client` and `url` of a CSV to request the data.
- `download_csvs_data` will use the parameter `urls_mapping` and the `request_csv_data` function to download all CSVs data and
convert them to a list of `StringIO` objects that can be read from the `pd.read_csv` function as dataframes.
- `download_data` will use the parameter `urls_mapping` and the `download_csvs_data` function to download all CSVs data, read them
as dataframes and combine them into a single dataframe. 

Let's use the last function to run the data workflow:
"""

# %%
data = download_data(URLS_MAPPING)
data

# %% [markdown]
"""
The above code works perfectly fine but if you would like to have properties like scheduling, retries, logging, observability etc
then you would have to implement these features from scratch. 

## Using task and flows

Prefect offers all the above functionality. It also uses some sensible defaults but we can further customize the data workflow.
Based on the definitions of Prefect concepts, we can decorate the functions as follows:

- `request_csv_data` represents a discrete unit of work and will receive the `task` decorator.
- `download_csvs_data` contains the above tasks and will receive the `flow` decorator.
- `download_data` implements the full data workflow and will receive the `flow` decorator.

Therefore `request_csv_data` represents tasks, while `download_csvs_data` is a subflow of the `download_data` flow:
"""


# %%
@task(name='Request CSV data.', retries=5)
def request_csv_data(client: httpx.Client, url: str, **kwargs):
    logger = get_run_logger()
    start_time = time()
    response = client.get(url=url)
    logger.info(
        f'CSV data, {kwargs["league"]} league and {kwargs["season"]} season, response time: {time() - start_time}s'
    )
    return response


@flow(name='Download synchronously CSVs data.', validate_parameters=True)
def download_csvs_data(urls_mapping: dict[str, tuple[str, str]]):
    logger = get_run_logger()
    start_time = time()
    with httpx.Client() as client:
        responses = [
            request_csv_data(client, url, league=league, season=season)
            for url, (league, season) in urls_mapping.items()
        ]
    csvs_data = [
        StringIO(str(response.content, encoding='windows-1254'))
        for response in responses
    ]
    logger.info(f'CSVs data download time: {time() - start_time}s')
    return csvs_data


@flow(name='Download synchronously data.', validate_parameters=True)
def download_data(urls_mapping: dict[str, tuple[str, str]]):
    logger = get_run_logger()
    start_time = time()
    csvs_data = download_csvs_data(urls_mapping)
    data = [pd.read_csv(csv_data, encoding='windows-1254') for csv_data in csvs_data]
    data = pd.concat(data, ignore_index=True)
    logger.info(f'Data download time: {time() - start_time}s')
    return data


# %% [markdown]
"""
We run the updated flow:
"""

# %% [markdown]
data = download_data(URLS_MAPPING)
data

# %% [markdown]
"""
## Concurrent task runner

The above code executes the tasks in a sequence. This is not optimal for downloading a large number of files. Instead, using an
asynchronous httpx client will concurrently download the data. A current limitation of Prefect is that it does not allow passing
the asynchronous client from the flow to the tasks. Therefore we remove the `task` decorator from `request_csv_data`.
Nevertheless, we can still log the same message with the use of the `print` function and the `log_prints` parameter of the `flow`
decorator:
"""


# %%
async def request_csv_data(client: httpx.AsyncClient, url: str, **kwargs):
    start_time = time()
    response = await client.get(url=url)
    print(
        f'CSV data, {kwargs["league"]} league and {kwargs["season"]} season, response time: {time() - start_time}s'
    )
    return response


@flow(name='Download asynchronously CSVs data.', validate_parameters=True)
async def download_csvs_data(urls_mapping: dict[str, tuple[str, str]]):
    logger = get_run_logger()
    start_time = time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=30)) as client:
        requests = [
            request_csv_data(client, url, league=league, season=season)
            for url, (league, season) in urls_mapping.items()
        ]
        responses = await asyncio.gather(*requests)
    csvs_data = [
        StringIO(str(response.content, encoding='windows-1254'))
        for response in responses
    ]
    logger.info(f'CSVs data download time: {time() - start_time}s')
    return csvs_data


@flow(
    name='Download asynchronously the data.',
    validate_parameters=True,
    task_runner=ConcurrentTaskRunner(),
    log_prints=True,
)
async def download_data(urls_mapping: dict[str, tuple[str, str]]):
    logger = get_run_logger()
    start_time = time()
    csvs_data = await download_csvs_data(urls_mapping)
    data = [pd.read_csv(csv_data, encoding='windows-1254') for csv_data in csvs_data]
    data = pd.concat(data, ignore_index=True)
    logger.info(f'Data download time: {time() - start_time}s')
    return data


# %% [markdown]
"""
Running the flow speeds up the process significantly:
"""

# %%
data = await download_data(URLS_MAPPING)
data

# %% [markdown]
"""
## Prefect UI and deployments

You can spin up a local Prefect server UI with the `prefect server start` command in the shell and explore the characteristics of
the above Prefect flows we ran. The data are stored in the Prefect database which by default is a local SQLite database. To reset
it, you can run the command `prefect server database reset -y`.

Prefect also supports deployments i.e. packaging workflow code, settings, and infrastructure configuration so that the data
workflow can be managed via the Prefect API and run remotely by a Prefect agent.

You can read more at the official [Prefect documentation](https://docs.prefect.io).
"""
