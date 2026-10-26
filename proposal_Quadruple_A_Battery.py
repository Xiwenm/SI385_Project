import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SI385 Project Proposal
    ## Group Name: Quadruple A Battery
    Group Members: Faye Guan, Jessica Moon, Sean Park, Xiwen Mark
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from io import BytesIO
    import zipfile

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Question

    To what extent do time of day, day of week, and individual rides predict posted standby wait times at Walt Disney World and Universal Orlando between June and September 2026?
    """)
    return
    

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Dataset

    We will use the Chart the Magic open wait-time dataset (chartthemagic.com/open-data, CC BY 4.0). The data comes from ThemeParks.wiki and records posted wait times for major rides at Walt Disney World and Universal Orlando every 10 minutes.

    We will use four months of data from June to September 2026, including two types of files. The daily file includes information such as park, ride, average wait time, peak posted wait time, and number of readings for each day. We will use this to compare wait times across days of the week and different attractions. The hourly file shows average wait times for each ride by hour, averaged over each month. We will use this to look at how wait times change throughout the day.

    We can connect the two files using the ride ID. However, since the hourly data is averaged by month, we cannot directly compare time of day and day of week together. Instead, we will analyze them separately.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data 1
    """)
    return


@app.cell
def _(pd):
    # Data 1
    base_url = "https://chartthemagic.com/open-data"

    months = ["2026-06", "2026-07", "2026-08", "2026-09"]

    # Daily File
    daily_frames = []

    for month in months:
        url = f"{base_url}/orlando-wait-times-daily-{month}.csv"
        df = pd.read_csv(url)
        daily_frames.append(df)

    daily_wait = pd.concat(daily_frames, ignore_index=True)

    daily_wait["date"] = pd.to_datetime(daily_wait["date"])
    daily_wait["weekday"] = daily_wait["date"].dt.day_name()
    daily_wait["month"] = daily_wait["date"].dt.month

    # Hourly File
    hourly_frames = []

    for month in months:
        url = f"{base_url}/orlando-wait-times-hourly-{month}.csv"
        df = pd.read_csv(url)
        hourly_frames.append(df)

    hourly_wait = pd.concat(hourly_frames, ignore_index=True)

    return daily_wait, hourly_wait


@app.cell
def _(daily_wait):
    # Daily File initial look
    print("Rows:", len(daily_wait))
    print("Parks:", daily_wait["park"].nunique())
    print("Rides:", daily_wait["ride"].nunique())
    print("Date range:", daily_wait["date"].min(), "to", daily_wait["date"].max())

    daily_wait.head()
    return


@app.cell
def _(hourly_wait):
    # Hourly File initial look
    print("Rows:", len(hourly_wait))
    print("Columns:", hourly_wait.columns.tolist())

    hourly_wait.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data 2
    """)
    return


@app.cell
def _(pd):
    hf_url = (
        "https://huggingface.co/datasets/"
        "matanzig/Disney-Theme-Park-Queue-Dynamics/"
        "resolve/main/disney_cleaned_dataset.csv"
    )

    # Load only a sample
    hf_wait = pd.read_csv(hf_url, nrows=100_000)

    print("Rows:", len(hf_wait))
    print("Columns:", hf_wait.columns.tolist())

    hf_wait.head()
    return (hf_wait,)


@app.cell
def _(hf_wait, pd):
    #Data Prep
    hf_wait["DEB_TIME"] = pd.to_datetime(
        hf_wait["DEB_TIME"], errors="coerce"
    )

    hf_wait["hour"] = hf_wait["DEB_TIME"].dt.hour
    hf_wait["weekday"] = hf_wait["DEB_TIME"].dt.day_name()
    hf_wait["month"] = hf_wait["DEB_TIME"].dt.month

    hf_wait.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Motivation

    Theme park visitors often spend a significant amount of time waiting in lines at Walt Disney World and Universal Orlando, which can make it difficult to plan the day efficiently. Understanding how wait times vary by time of day, day of week, and attraction could help visitors decide when to visit the park, when to go on certain rides, and reduce unnecessary waiting. These findings could also help theme park operators better understand demand patterns and improve crowd management.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Division of Labour

    * **Jessica:** Analyze how wait times change throughout the day and identify which hours have the longest wait times.
    * **Faye:** Analyze how wait times differ across days of the week and compare weekdays and weekends.
    * **Sean:** Compare wait times across individual rides and identify which rides tend to have longer wait times.
    * **Xiwen:** Develop and evaluate regression models predicting posted standby wait times: one using day of week and attraction (daily file), and one using time of day and attraction (hourly file), and compare how much each factor explains.

    All team members will contribute to data cleaning, interpreting the results, and reviewing the final notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
