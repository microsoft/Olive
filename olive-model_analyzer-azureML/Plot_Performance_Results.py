import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def getParameters(fileName):
    parameters = {}

    with open(fileName) as f:
        lines = f.readlines()
    numLines = len(lines)
    parameterLines = [
        "instance_group {",
        "dynamic_batching {",
        "optimization {",
        "parameters {",
    ]
    keys = ["execution_mode", "inter_op_thread_count", "intra_op_thread_count"]
    for key in keys:
        parameters[key] = "NA"
    lineNum = 0
    parameters["count"] = "1"
    parameters["preferred_batch_size"] = []
    parameters["back_end"] = "CUDA"
    parameters["precision"] = "FP32"
    parameters["graph_level"] = "NA"
    while lineNum < numLines - 1:
        lineNum = lineNum + 1
        line = lines[lineNum].strip()
        if line in parameterLines:
            if line == "instance_group {":
                while line != "}":
                    lineNum = lineNum + 1
                    line = lines[lineNum].strip()
                    if len(line) > 5 and line[0:5] == "count":
                        parameters["count"] = line.split(" ")[1]
            if line == "dynamic_batching {":
                while line != "}":
                    lineNum = lineNum + 1
                    line = lines[lineNum].strip()
                    if len(line) > 20 and line[0:20] == "preferred_batch_size":
                        parameters["preferred_batch_size"].append(line.split(" ")[1])
            if line == "optimization {":
                while line != "}\n":
                    lineNum = lineNum + 1
                    line = lines[lineNum]
                    if len(line) > 10 and line[0:10] == "      name":
                        parameters["back_end"] = line.strip().split(" ")[1]
                        parameters["back_end"] = parameters["back_end"].strip('""')
                    if len(line) > 29 and line[0:29] == '        key: "precision_mode"':
                        lineNum = lineNum + 1
                        line = lines[lineNum].strip()
                        parameters["precision"] = line.split(" ")[1]
                        parameters["precision"] = parameters["precision"].strip('""')
                    if len(line) > 8 and line[0:9] == "  graph {":
                        lineNum = lineNum + 1
                        line = lines[lineNum]
                        if len(line) > 8 and line[0:9] == "    level":
                            parameters["graph_level"] = line.strip().split(" ")[1]
            if line == "parameters {":
                while line != "}\n":
                    lineNum = lineNum + 1
                    line = lines[lineNum]
                    if len(line) > 5 and line[0:6] == "  key:":
                        key = line.strip().split(" ")[1]
                        key = key.strip('""')
                        if key in keys:
                            lineNum = lineNum + 1
                            line = lines[lineNum]
                            if len(line) > 8 and line[0:9] == "  value {":
                                lineNum = lineNum + 1
                                line = lines[lineNum]
                                if len(line) > 16 and line[0:17] == "    string_value:":
                                    parameters[key] = line.strip().split(" ")[1]
                                    parameters[key] = parameters[key].strip('""')
    return parameters


def printGridParameters(parameters):
    grid_parameters_str = (
        "EM:"
        + parameters["execution_mode"]
        + "; ETC:"
        + parameters["inter_op_thread_count"]
        + "; ATC:"
        + parameters["intra_op_thread_count"]
        + "; GL:"
        + parameters["graph_level"]
        + "; MC:"
        + parameters["count"]
    )
    if len(parameters["preferred_batch_size"]) == 0:
        grid_parameters_str = grid_parameters_str + "; No DB"
    else:
        grid_parameters_str = (
            grid_parameters_str + "; DB:" + " ".join(parameters["preferred_batch_size"])
        )
    return grid_parameters_str


def extendDataframe(df, parameters_str, config_dict):
    for key in parameters_str:
        df[key] = "NA"
    df["grid_parameters"] = "NA"
    df["max"] = -1
    parameters_str.remove("preferred_batch_size")
    for config_path, parameters in config_dict.items():
        for key in parameters_str:
            df.loc[df["Model Config Path"] == config_path, key] = parameters[key]
        key = "preferred_batch_size"
        df.loc[df["Model Config Path"] == config_path, key] = " ".join(parameters[key])
        df.loc[
            df["Model Config Path"] == config_path, "grid_parameters"
        ] = printGridParameters(parameters)
        fg = df[df["Model Config Path"] == config_path][
            "Throughput (infer/sec)"
        ].sort_values()
        fglen = len(fg)
        if fglen > 2:
            df.loc[df["Model Config Path"] == config_path, "max"] = (
                fg.iloc[fglen - 1] + fg.iloc[fglen - 2] + fg.iloc[fglen - 3]
            ) / 3
        else:
            if fglen > 1:
                df.loc[df["Model Config Path"] == config_path, "max"] = (
                    fg.iloc[fglen - 1] + fg.iloc[fglen - 2]
                ) / 2
            else:
                df.loc[df["Model Config Path"] == config_path, "max"] = fg.iloc[
                    fglen - 1
                ]
    df.sort_values(
        ["back_end", "precision", "Model Config Path", "Concurrency"], inplace=True
    )
    return df


def removeEmpty(df):
    numEls = len(df)
    for i in range(0, numEls):
        max = df.iloc[i].max()
        df.iloc[i][df.iloc[i] == -1] = max
    return df


def getBaselineConfigPath(df):
    config_paths = df[(df["count"] == "1") & (df["preferred_batch_size"] == "")][
        "Model Config Path"
    ].unique()
    return config_paths[0]


def getMaxConfigPath(group_df):
    max_value = group_df["max"].max()
    max_row = group_df[group_df["max"] == max_value]
    config_file = max_row["Model Config Path"].unique()[0]
    return config_file


def getDefaultConfigPath(group_df):
    return [
        x for x in group_df["Model Config Path"].unique() if x[-14:] == "config_default"
    ][0]


def partitionbybackend(df):
    limits_df = pd.DataFrame(
        columns=["Backend_Precision", "Limit", 1, 8, 16, 32, 64, 128]
    )
    partition_members = {}
    back_end_precision_groups = df.groupby(["back_end", "precision"])
    Concurrency_values = [1, 2, 4, 8, 16, 32, 64, 128]
    keys = list(back_end_precision_groups.groups.keys())
    for key in keys:
        back_end_precision = "_".join(key)
        group_df = back_end_precision_groups.get_group(key)
        limits = {}
        limits["Baseline"] = getDefaultConfigPath(group_df)
        max_throughput_df = group_df.groupby("Model Config Path")[["max"]].max()
        max_throughput_df.reset_index(inplace=True)
        partition_members[back_end_precision + "_Baseline_GP"] = group_df[
            group_df["Model Config Path"] == limits["Baseline"]
        ]["grid_parameters"].unique()[0]
        members = list(group_df["Model Config Path"].unique())
        members.remove(limits["Baseline"])
        if len(members) > 1:
            limits["Maximum"] = getMaxConfigPath(max_throughput_df)
            best_performing_config_file = limits["Maximum"]
            partition_members[back_end_precision + "_Maximum_GP"] = group_df[
                group_df["Model Config Path"] == limits["Maximum"]
            ]["grid_parameters"].unique()[0]
            if limits["Maximum"] in members:
                members.remove(limits["Maximum"])
        partition_members[back_end_precision] = members
        for x in members:
            limits[x] = x
        for limit in limits.keys():
            dataframeentry = {"Backend_Precision": back_end_precision}
            dataframeentry["Limit"] = limit
            for concurrency in Concurrency_values:
                value = group_df[
                    (group_df["Model Config Path"] == limits[limit])
                    & (group_df["Concurrency"] == concurrency)
                ]["Throughput (infer/sec)"]
                if value.empty:
                    value = -1.0
                else:
                    value = float(value)
                dataframeentry[concurrency] = value
            limits_df = limits_df.append(dataframeentry, ignore_index=True)

    limits_df.set_index(["Backend_Precision", "Limit"], inplace=True)
    limits_df.columns = [int(x) for x in limits_df.columns]
    return limits_df, partition_members, best_performing_config_file


def update_axis_partition(
    ax,
    df,
    members,
    backend_precision_pair,
    pos,
    clr1=plt.cm.Purples(0.99),
    clr2=plt.cm.Purples(0.75),
    clr3=plt.cm.Purples(0.6),
    clr4=plt.cm.Purples(0.35),
):
    plotted = 0
    x = df.columns
    baselinePars = members[backend_precision_pair + "_Baseline_GP"]
    if backend_precision_pair + "_Maximum_GP" in members:
        maximumPars = members[backend_precision_pair + "_Maximum_GP"]
        y_m = df.loc[(slice(None), "Maximum"), :].iloc[0]
        ax.plot(
            x,
            y_m,
            label="OLive+Model Analyzer : ("
            + backend_precision_pair
            + " : "
            + str(y_m.max())
            + ") ["
            + maximumPars
            + "]",
            color=clr1,
        )
        y_b = df.loc[(slice(None), "Baseline"), :].iloc[0]
        ax.plot(
            x,
            y_b,
            label="OLive : ("
            + backend_precision_pair
            + " : "
            + str(y_b.max())
            + ") ["
            + baselinePars
            + "]",
            color=clr3,
        )
    else:
        y_b = df.loc[(slice(None), "Baseline"), :].iloc[0]
        ax.plot(
            x,
            y_b,
            label="Baseline : ("
            + backend_precision_pair
            + " : "
            + str(y_b.max())
            + ") ["
            + baselinePars
            + "]",
            color=clr3,
        )
    ax.set_ylabel("Throughput (infer/sec)", fontsize="medium")
    ax.set_xlabel("Request Concurrency", fontsize="medium")
    ax.tick_params(axis="both", labelsize="small")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(
        bbox_to_anchor=(0.9, pos), loc="upper right", frameon=False, fontsize="x-small"
    )


def WriteOptimalLocationFile(filename, location):
    with open(filename, "w") as f:
        f.write(location)


if __name__ == "__main__":

    ### PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_repository", default="")
    parser.add_argument("--inference_results_file", default="")
    parser.add_argument("--output_figure_file", default="Optimal_Results.png")
    parser.add_argument(
        "--optimal_location_file", default="Optimal_ConfigFile_Location.txt"
    )

    args, unparsed = parser.parse_known_args()

    output_repository = args.output_repository
    inference_results_file = args.inference_results_file
    output_figure_file = args.output_figure_file
    optimal_location_file = args.optimal_location_file

    if output_repository == "":
        raise Exception("Argument --output_repository has to be provided")

    if inference_results_file == "":
        raise Exception("Argument --inference_results_file has to be provided")

    allFiles = getListOfFiles(output_repository)
    allFiles = [x for x in allFiles if x[-5:] != ".onnx"]
    ini = len(output_repository) + 1
    config_paths = [x[ini:-13] for x in allFiles]
    parameters = [getParameters(x) for x in allFiles]
    config_dict = dict(zip(config_paths, parameters))
    config_files_locations = dict(zip(config_paths, allFiles))
    parameters_str = list(parameters[0].keys())
    results_df = pd.read_csv(inference_results_file)
    results_df
    extended_df = extendDataframe(results_df, parameters_str, config_dict)

    partition_df, partition_members, best_performing_config_file = partitionbybackend(
        extended_df
    )

    print("Best Performing Config File : ")
    print(config_files_locations[best_performing_config_file])
    WriteOptimalLocationFile(
        optimal_location_file, config_files_locations[best_performing_config_file]
    )

    partition_df = removeEmpty(partition_df)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.set_title(
        "OLive + Model Analyzer Profiling Throughput (infer/sec) Results",
        fontsize="x-large",
        fontweight="bold",
    )
    backend_precision_pair = "tensorrt_FP16"
    df_b = partition_df.loc[([backend_precision_pair]), :]
    pos = 0.6
    update_axis_partition(
        ax1,
        df_b,
        partition_members,
        backend_precision_pair,
        pos,
        clr1=plt.cm.Greens(0.9),
        clr2=plt.cm.Greens(0.75),
        clr3=plt.cm.Greens(0.6),
        clr4=plt.cm.Greens(0.15),
    )
    backend_precision_pair = "CUDA_FP32"
    df_b = partition_df.loc[([backend_precision_pair]), :]
    update_axis_partition(
        ax1,
        df_b,
        partition_members,
        backend_precision_pair,
        pos,
        clr1=plt.cm.Blues(0.9),
        clr2=plt.cm.Blues(0.75),
        clr3=plt.cm.Blues(0.6),
        clr4=plt.cm.Reds(0.15),
    )
    plt.savefig(output_figure_file)
