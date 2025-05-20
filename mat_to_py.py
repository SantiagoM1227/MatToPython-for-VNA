from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
plt.rcParams['font.family'] = 'DejaVu Sans'




def scientific_notation(a,b):
  order_a = int(np.floor(np.log10(np.abs(a))))
  order_b = int(np.floor(np.log10(np.abs(b))))

  normalized_a = a / (10**order_a)
  normalized_b = b / (10**order_b)


  n = int(order_a - order_b)
  decimal_places = max(0, n + 1)
  format_string = f"{{:.{decimal_places}f}}"
  a_str = format_string.format(normalized_a)
  b_str = f"{normalized_b:.1f}"
  scientific_notation = f"({a_str} ± {b_str} × 10^({-n:.0f})) × 10^{order_a}"
  return scientific_notation



def plot_data(x_data, y_datas, x_label, y_labels, y_label, fit=False, starting_point=0.5, fitlength=0.2, square_root=False, save=False, filename="plot", num_ticks = 10):

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]

    largest_power = np.floor(np.log10(np.max(x_data)))
    x_data_normalized = x_data / (10**largest_power)

    if square_root:
        x_data_normalized = np.sqrt(x_data_normalized)
        x_label = f"$\sqrt{{{x_label}}}$"

    max_x = np.max(x_data_normalized)
    min_x = np.min(x_data_normalized)
    x_range = max_x - min_x

    # Convert single y_data to list for uniform processing
    if not isinstance(y_datas, (list, tuple)) or isinstance(y_datas[0], (int, float, np.number, np.ndarray)) and np.ndim(y_datas) == 1:
        y_datas = [y_datas]
        y_labels = [y_labels]

    plt.figure(figsize=(10, 6))

    for i, y_data in enumerate(y_datas):
        label = y_labels[i]

        if fit:
            labels_fit = ["p", "q"]
            start = int(len(x_data) * starting_point)
            end = int(len(x_data) * (starting_point + fitlength))

            y_largest_power = np.floor(np.log10(np.max(np.abs(y_data))))
            y_data_normalized = y_data / (10**y_largest_power)

            popt, pcov = curve_fit(linear, x_data_normalized[start:end], y_data_normalized[start:end])
            perr = np.sqrt(np.diag(pcov))
            popt_plot = popt * 10**y_largest_power

            popt_labels = popt * 10**y_largest_power
            perr_labels = perr * 10**y_largest_power

            labels_scientific = [
                scientific_notation(popt_labels[0], perr_labels[0]),
                scientific_notation(popt_labels[1], perr_labels[1])
            ]

            legend_str = f"{label} fit:\n"
            for label_f, label_scientific in zip(labels_fit, labels_scientific):
                legend_str += f"{label_f} = ({label_scientific}\n"

            xp = np.linspace(min_x, max_x, 100)
            yp = linear(xp, *popt_plot)
            plt.plot(xp, yp, label=legend_str.strip())

            plt.scatter(x_data_normalized, y_data, label=label, s=5, alpha=0.7)

        else:
            plt.plot(x_data_normalized, y_data, linewidth=1.6, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=label)


    plt.tight_layout()
    plt.legend(loc="best", fontsize="large")
    plt.xlabel(f"{x_label}", fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    if x_range < 1 and x_range > 0: # Apply new logic if range is less than 1 and greater than 0
        # num_ticks is now a parameter
      tick_values = np.linspace(min_x, max_x, num_ticks)
      tick_labels = []

      # Determine the order of magnitude of the range
      order_of_range = np.floor(np.log10(x_range))

      # Determine the required precision
      avg_diff = x_range / (num_ticks - 1) if num_ticks > 1 else 0.0
      if avg_diff > 0:
        required_precision = max(0, int(-np.floor(np.log10(avg_diff))))
      else:
        required_precision = 3 # Default precision for zero range


      # Standard formatting for small ranges
      display_precision = max(required_precision, 3) # Ensure at least 3 decimal places for small ranges
      tick_labels = [f"{val:.{display_precision}f}" for val in tick_values]
      plt.xlabel(x_label, fontsize=14)

    else: # Use previous logic for range >= 1 or range <= 0
              # Use integer/scientific notation ticks
      max_x_int = int(np.floor(max_x))
      if max_x - max_x_int >= 0.45:
        max_x_int += 1

      # Handle cases where min_x is very small or zero for log10
      if min_x > 0:
        min_order = int(np.floor(np.log10(min_x)))
        tick_start = round(min_x, -min_order)
      else:
        tick_start = np.ceil(min_x) # Start at the next integer if min_x <= 0

      tick_step = 1
        
      tick_values = np.arange(tick_start, max_x_int + tick_step, tick_step)

      # Ensure max_x is included if it's at or very close to the next integer tick
      if max_x >= tick_values[-1] - 1e-9 and max_x > tick_values[-1]:
        tick_values = np.append(tick_values, max_x)


      tick_labels = []
        
      for val in tick_values:
        if val >= 1000 or (val < 1 and val > 0): # Use scientific for large numbers or numbers between 0 and 1
          tick_labels.append(f"{val:.0e}")
        elif val <= 0: # Handle zero or negative numbers
          tick_labels.append(f"{int(val)}")
        else: # Use integer format for numbers >= 1
          tick_labels.append(f"{int(val)}")


      plt.xlabel(x_label, fontsize=14)

    

    plt.xticks(tick_values, labels=tick_labels)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save:
        dir_path = os.path.dirname(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        suffix = "_fit" if fit else ""
        pdf_filename = os.path.join(dir_path, f"{base_name}{suffix}.pdf")
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')

    plt.show()
    plt.close()



def linear(x, a, b):
    return a * x + b

def S21_dB_func(f,f0,QL):
  delta = (f/f0) - (f0/f)
  S21 = np.abs( (1+(QL*delta)*1j))
  return -20*np.log10(S21)

class mat_to_py():
    def __init__(self, filename, labels = None):
        self.power_to_prefix = {
          -12: 'p',   # pico
          -9 : 'n',   # nano
          -6 : 'μ',   # micro (mu)
          -3 : 'm',    # milli
          0  : '' ,     # base unit
          3  : 'k',    # kilo
          6  : 'M',    # mega
          9  : 'G',    # giga
          12 : 'T',    # tera
          15 : 'P',    # peta
          18 : 'E',    # exa
          21 : 'Z',    # zetta
          24 : 'Y',    # yotta
          }
        self.filename = filename
        fig_data = loadmat(filename)
        hgS_070000 = fig_data['hgS_070000']
        hgS_070000_children = hgS_070000['children'][0,0]
        hgS_070000_plot_elements = hgS_070000_children['children'][0,0]
        line_series = hgS_070000_plot_elements[0][0]
        self.x_data = line_series['properties'][0]['XData'][0][0]
        self.y_data = line_series['properties'][0]['YData'][0][0]

        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)

        if labels is not None:
          self.x_label = labels[0]
          self.y_label = labels[1]
        else:
          label_X_handler = hgS_070000_plot_elements[1][0]
          label_Y_handler = hgS_070000_plot_elements[2][0]
          self.x_label = label_X_handler[2][0,0]['String'][0]
          self.y_label = label_Y_handler[2][0,0]['String'][0]


    def __call__(self):
        return self.x_data, self.y_data
    
    def new_plot(self, square_root = False, save = False, fit = False, fitlength = 0.2, starting_point = 0.5, num_ticks=10):

      x_data = self.x_data
      y_data = self.y_data
      x_label = self.x_label
      y_label = self.y_label


      dir_path = os.path.dirname(self.filename)
      base_name = os.path.splitext(os.path.basename(self.filename))[0]
      pdf_filename = os.path.join(dir_path, f"{base_name}.pdf")
      if fit:
        pdf_filename = os.path.join(dir_path, f"{base_name}_fit.pdf")
      
      plot_data(x_data = x_data, 
                y_datas = y_data, 
                x_label = x_label, 
                y_labels = y_label, 
                y_label = y_label, 
                fit=fit, 
                starting_point=starting_point, 
                fitlength=fitlength, 
                square_root=square_root, 
                save=save, 
                filename= pdf_filename, 
                num_ticks = num_ticks)
      




    def plot(self, square_root = False, save = False, fit = False, fitlength = 0.2, starting_point = 0.5, num_ticks=10):

        x_data, y_data = self.x_data, self.y_data
        x_label, y_label = self.x_label, self.y_label



        largest_power = np.floor(np.log10(np.max(x_data)))

        x_data_normalized = x_data / (10**largest_power)

        if square_root:
            x_data_normalized = np.sqrt(x_data_normalized)
            x_label = f"$\sqrt{{{x_label}}}$"

        max_x = np.max(x_data_normalized)
        min_x = np.min(x_data_normalized)
        x_range = max_x - min_x

        plt.figure(figsize=(10, 6))



        if fit:
          labels = ["p", "q"]

          start = int(len(x_data)*(starting_point-fitlength/2.))
          end = int(len(x_data)*(starting_point+fitlength/2.))

          y_largest_power = np.floor(np.log10(np.max(np.abs(y_data))))
          y_data_normalized = y_data / (10**y_largest_power)

          #fit on normalized data to get better floating point resolution
          # i. e. deal with 1e-9 to 1e-6
          popt, pcov = curve_fit(linear, x_data_normalized[start:end], y_data_normalized[start:end])


          perr = np.sqrt(np.diag(pcov))

          #in order to plot normalized data against full y_data
          popt_plot = popt*10**y_largest_power
          perr_plot = perr*10**y_largest_power


          popt_labels = popt*10**(y_largest_power)
          perr_labels = perr*10**(y_largest_power)


          labels_scientific = [scientific_notation(popt_labels[0], perr_labels[0]),
                               scientific_notation(popt_labels[1], perr_labels[1])]
          print(labels_scientific)

          legend_str = f"linear fit:\n"
          for label, label_scientific in zip(labels, labels_scientific):
              legend_str += f"{label} = ({label_scientific})\n"

          xp = np.linspace(min_x, max_x, 100)
          yp = linear(xp, *popt_plot)
          plt.plot(xp, yp, 'r-', label=legend_str.strip() )
          plt.scatter(x_data_normalized, y_data, label="Data",marker = "o",s = 1, color = "blue")

          plt.title(f"LINEAR Fit with Parameters")

        else :
          plt.plot(x_data_normalized, y_data, 'r-', linewidth=1, label = "Data")

        plt.tight_layout()
        plt.legend(loc="best", fontsize="large")

        plt.xlabel(f"{x_label}", fontsize=14)
        plt.ylabel(y_label, fontsize=14)


        if x_range < 1 and x_range > 0: # Apply new logic if range is less than 1 and greater than 0
        # num_ticks is now a parameter
          tick_values = np.linspace(min_x, max_x, num_ticks)
          tick_labels = []

          # Determine the order of magnitude of the range
          order_of_range = np.floor(np.log10(x_range))

          # Determine the required precision
          avg_diff = x_range / (num_ticks - 1) if num_ticks > 1 else 0.0
          if avg_diff > 0:
            required_precision = max(0, int(-np.floor(np.log10(avg_diff))))
          else:
            required_precision = 3 # Default precision for zero range


          display_precision = max(required_precision, 3) # Ensure at least 3 decimal places for small ranges
          tick_labels = [f"{val:.{display_precision}f}" for val in tick_values]
          plt.xlabel(x_label, fontsize=14)

        else: # Use previous logic for range >= 1 or range <= 0
              # Use integer/scientific notation ticks
          max_x_int = int(np.floor(max_x))
          if max_x - max_x_int >= 0.45:
            max_x_int += 1

          # Handle cases where min_x is very small or zero for log10
          if min_x > 0:
            min_order = int(np.floor(np.log10(min_x)))
            tick_start = round(min_x, -min_order)
          else:
            tick_start = np.ceil(min_x) # Start at the next integer if min_x <= 0

          tick_step = 1
          tick_values = np.arange(tick_start, max_x_int + tick_step, tick_step)

          # Ensure max_x is included if it's at or very close to the next integer tick
          if max_x >= tick_values[-1] - 1e-9 and max_x > tick_values[-1]:
            tick_values = np.append(tick_values, max_x)


          tick_labels = []
          for val in tick_values:
            if val >= 1000 or (val < 1 and val > 0): # Use scientific for large numbers or numbers between 0 and 1
                 tick_labels.append(f"{val:.3e}")
            elif val <= 0: # Handle zero or negative numbers
                 tick_labels.append(f"{int(val)}")
            else: # Use integer format for numbers >= 1
                 tick_labels.append(f"{int(val)}")


          plt.xlabel(x_label, fontsize=14)

        plt.xticks(tick_values, labels=tick_labels)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save:
          dir_path = os.path.dirname(self.filename)
          base_name = os.path.splitext(os.path.basename(self.filename))[0]
          pdf_filename = os.path.join(dir_path, f"{base_name}.pdf")
          if fit:
            pdf_filename = os.path.join(dir_path, f"{base_name}_fit.pdf")
          plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    def peak_finder(self, range=[0.000001,1e+20], num_peaks = 2,save = False, sigma_filter = 2, threshold = 1e-10, prominence = 1e-10):
          from scipy.signal import find_peaks
          from scipy.ndimage import gaussian_filter1d
          from matplotlib.lines import Line2D


          x_data, y_data = self.x_data, self.y_data
          x_label, y_label = self.x_label, self.y_label

          good_indices = np.where((x_data >= range[0]) & (x_data <= range[1]))

          x_data = x_data[good_indices]
          y_data = y_data[good_indices]


          x_data = np.array(x_data)
          y_data = np.array(y_data)

          y_data = gaussian_filter1d(y_data, sigma=sigma_filter)


          peaks, _ = find_peaks(y_data, threshold= threshold, prominence = prominence)
          if len(peaks) < num_peaks:
            print(f"Found {len(peaks)} peaks.")





          proxy = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Detected Peaks')

          custom_line = Line2D([0], [0], marker = 'o', color='black', lw=2)



          plt.figure(figsize=(10, 6))
          plt.plot(x_data, y_data, 'b-', linewidth=1, label = "Data")
          handles, labels2 = plt.gca().get_legend_handles_labels()

          n = 1
          for peak in peaks[:num_peaks]:
              plt.axvline(x= x_data[peak], color='r', linestyle='--')
              print(f"Peak found at: {x_data[peak]}")
              labels2.append(f"f[{n}] = {int(x_data[peak])} [Hz] ")
              handles.append(proxy)
              n=n+1

          plt.legend(handles=handles,loc="best", fontsize="large", labels=labels2)

          plt.xlabel(f"{x_label}", fontsize=14)
          plt.ylabel(y_label, fontsize=14)
          plt.grid(True, which='both', linestyle='--', linewidth=0.5)
          if save:
            dir_path = os.path.dirname(self.filename)
            base_name = os.path.splitext(os.path.basename(self.filename))[0]
            pdf_filename = os.path.join(dir_path, f"{base_name}_found_Peaks.pdf")
            plt.savefig(pdf_filename, format='pdf')
          plt.show()
          plt.close()
          return peaks

    def lorentz_peak(self, save = False):

          frequencies = self.x_data
          min_x = np.min(frequencies)
          max_x = np.max(frequencies)


          S21dB_exp= self.y_data
          f0 = frequencies[S21dB_exp.argmax()]

          S21_f0 = S21dB_exp.max()
          S21dB_norm = S21dB_exp-S21_f0
          epsilon = 1

          popt, pcov = curve_fit(S21_dB_func, frequencies, S21dB_norm, p0=[f0,11000], bounds = ((f0-epsilon,-np.inf), (f0+epsilon,np.inf)))

          labels_fit = [f"$f_0$", f"$Q_L$"]
          S21dB_theo = S21_dB_func(frequencies, *popt)


          perr = np.sqrt(np.diag(pcov))

          popt_labels = popt
          perr_labels = perr

          labels_scientific = [scientific_notation(popt_labels[0], perr_labels[0]),
                               scientific_notation(popt_labels[1], perr_labels[1])]

          legend_str = "fit : \n"
          for label_f, label_scientific in zip(labels_fit, labels_scientific):
            legend_str += f"{label_f} = ({label_scientific}\n"

          dir_path = os.path.dirname(self.filename)
          base_name = os.path.splitext(os.path.basename(self.filename))[0]
          pdf_filename = os.path.join(dir_path, f"{base_name}_found_Peaks.pdf")

          plot_data(x_data = frequencies,
                    y_datas = [S21dB_norm,S21dB_theo],
                    x_label = "frequency[GHz]",
                    y_labels = [f"|S21/S21($f_0$)| exp [dB]",legend_str],
                    y_label = f"|S21/S21($f_0$)|[dB]",
                    save=save,
                    filename = pdf_filename
                    )








