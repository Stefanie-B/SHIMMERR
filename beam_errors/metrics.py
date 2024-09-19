import csv
import numpy as np
import os
import pickle


def _parse_response(fname, shape):
    with open(fname) as csv_file:
        data = list(csv.DictReader(csv_file))
        response = np.array([row["value"] for row in data]).astype(complex)
        return response.reshape(shape)


def compute_realized_gains(
    directory_realized_responses,
    directory_ideal_responses,
    fname_realized_gains,
    patch_names=None,
):
    if patch_names is None:
        patch_names = [
            fname.rstrip(".csv") for fname in os.listdir(directory_ideal_responses)
        ]

    with open(f"{directory_ideal_responses}/{patch_names[0]}.csv") as csv_file:
        data = list(csv.DictReader(csv_file))
        metadata = {
            "frequencies": np.unique([float(row["frequency"]) for row in data]),
            "times": np.unique([row["time"] for row in data]),
            "stations": np.unique([row["station"] for row in data]),
        }
    metadata["directions"] = patch_names

    realized_gains = np.empty(
        [
            metadata["times"].size,
            metadata["frequencies"].size,
            metadata["stations"].size,
            len(patch_names),
        ]
    ).astype(complex)
    for i, patch_name in enumerate(patch_names):
        ideal_response = _parse_response(
            f"{directory_ideal_responses}/{patch_name}.csv", realized_gains.shape[:-1]
        )
        realized_response = _parse_response(
            f"{directory_realized_responses}/{patch_name}.csv",
            realized_gains.shape[:-1],
        )
        realized_gains[:, :, :, i] = realized_response / ideal_response

    output_gains = [
        {"gains": realized_gains[i, :, :, :]} for i in range(metadata["times"].size)
    ]
    os.makedirs("/".join(fname_realized_gains.split("/")[:-1]), exist_ok=True)
    with open(fname_realized_gains, "wb") as fp:
        pickle.dump(output_gains, fp)
    with open(f"{fname_realized_gains}_metadata", "wb") as fp:
        pickle.dump(metadata, fp)
