import os




def generate_and_plot_figs(folder_path,
                           savefile=False,
                           fit_starting_point=0.5,
                           fit_length=0.2,
                           range = [1,9e9],
                           num_peaks = 20,
                           sigma_filter=7,
                           prominence = 6,
                           threshold = 0.001
                           ):
    """
    Generates plots from .fig files in a given folder, with options for fitting.

    Args:
        folder_path (str): The path to the folder containing .fig files.
        savefile (bool, optional): Whether to save the generated plots as PDF files.
                                   Defaults to True.
        fit_starting_point (float, optional): The starting point for the linear fit
                                            (as a fraction of the data length).
                                            Only applies to files identified as 'fit' type.
                                            Defaults to 0.5.
        fit_length (float, optional): The fraction of the data length to use for the
                                    linear fit, starting from fit_starting_point.
                                    Only applies to files identified as 'fit' type.
                                    Defaults to 0.2.

    Returns:
        tuple: A tuple containing:
               - filename_vars (dict): Dictionary mapping generated variable names
                                       to original filenames.
               - plot_configs (list): List of tuples containing plot configuration
                                      details for each file.
               - results (dict): Dictionary mapping generated variable names
                                 to mat_to_py objects.
    """
    plot_configs = []
    filename_vars = {}
    results = {}

    phase_keywords = ["UNWRAPPED", "UNTANGLEDPHASE", "PHASE"]
    delay_keywords = ["GROUP", "GROUPDELAY"]
    peak_keywords = ["PEAK", "MAX"]
    fullrange_keywords = ["FULLFREQRANGE", "FULLRANGE","FULL", "FullRange"]

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".fig"):
            continue

        filename = os.path.join(folder_path, file)
        base = file.replace(".fig", "")
        base_upper = base.upper()

        if any(kw in base_upper for kw in phase_keywords):
            label_type = "phase"
            label = f"Phase {base}"
            plot_type = "fit"

        elif any(kw in base_upper for kw in delay_keywords):
            label_type = "groupdelay"
            label = f"Group Delay {base}"
            plot_type = "fit"
            # label = f"Group Delay {base[:3]}"
        elif any(kw in base_upper for kw in peak_keywords):
            label_type = "peak"
            label = f"Peak {base}"
            plot_type = "peak"
        elif any(kw in base_upper for kw in fullrange_keywords):
            label_type = "fullrange"
            label = f"Full Range {base}"
            plot_type = "fullrange"
        else:
            label_type = "dB"
            label = f"{base} [dB]"
            plot_type = "dB"
        print(label)
        port_id = base.upper()
        var_prefix = port_id + ("_" + label_type if label_type != "dB" else "_dB")

        filename_var = var_prefix + "_filename"
        filename_vars[filename_var] = filename

        plot_configs.append((var_prefix, filename, label, plot_type))

    for var_name, filename, ylabel, plot_type in plot_configs:
        obj = mat_to_py(filename, labels=["Frequency[GHz]", ylabel])
        globals()[var_name] = obj
        results[var_name] = obj

        if plot_type == "dB":
            obj.plot(square_root=False, fit=False, save=savefile)
        elif plot_type == "fit":
            obj.plot(fit=True, fitlength=fit_length, save=savefile, starting_point=fit_starting_point)
        elif plot_type == "peak":
            obj.lorentz_peak(save=savefile)
        elif plot_type == "fullrange":
            obj.peak_finder(num_peaks = num_peaks, range= range, save = savefile, sigma_filter=sigma_filter, threshold = threshold, prominence = prominence)

    return filename_vars, plot_configs, results
