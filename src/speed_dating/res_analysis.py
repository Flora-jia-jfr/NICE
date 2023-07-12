import os
import numpy as np
import pandas as pd
from pprint import pprint

# res_dir = '/drive2/florajia/NICE/res/exp2_no_envir_no_collider'
res_dir = '/drive2/florajia/NICE/res/exp2'

dat_dir = '/drive2/florajia/NICE/dat/SpeedDatingDat'
# train_types = ['all_train', 'train_test']
train_types = ['train_test']

# Create a multi-index for your DataFrame
index = pd.MultiIndex.from_product([["high", "med", "low"], ["Mod1", "Mod2", "Mod3", "Mod4"], ["dragon", "tarnet", "tarnet_single"]], names=["adjust", "mod", "net"])
# metrics = ["PEHE", "final_test_ate_error", "final_train_ate_error", "test_acc", "test_ate", "test_treatment_acc", "train_acc", "train_ate", "train_treatment_acc"]
metrics = ["PEHE", "test_acc", "test_ate", "test_treatment_acc", "train_acc", "train_ate", "train_treatment_acc"]

lower_is_better_metrics = ["PEHE", "final_test_ate_error", "final_train_ate_error"]
higher_is_better_metrics = ["test_acc", "test_treatment_acc", "train_acc", "train_treatment_acc"]

# Create an empty DataFrame with desired multi-index columns and rows
df = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([metrics, ["avg", "std"]]))
pd.set_option('display.float_format', '{:.2f}'.format)

for train_type in train_types:
    for risk in ["erm", "irm"]:
        # for net in ["dragon", "tarnet", "tarnet_single"]:
        for net in ["tarnet"]:
            for mod_num in range(1,5):
                for dim in ["high", "low", "med"]:
                    results = {metric: [] for metric in metrics}
                    for output_num in range(1,11):
                        for te in ["ate", "ite"]:
                            file_type = "csv" if te == "ate" else "npz"
                            file_path = os.path.join(res_dir, net, f"no_collider/{train_type}", f"Mod{mod_num}", dim, "{}_{}_output_{}.{}".format(risk, te, output_num, file_type))
                            if file_type == "csv": # ate
                                data = pd.read_csv(file_path)
                                for metric in metrics:
                                    results[metric].append(data[metric].values[0])
                            # else: # ite
                            #     continue
                            #     data = np.load(file_path)
                            #     numpy_array = data.files
                            #     print(numpy_array)
                            #     print(data['pred_ite'])
                    # Store the averages and std dev to the DataFrame
                    for metric in metrics:
                        avg = np.mean(results[metric])
                        std = np.std(results[metric])
                        df.loc[(dim, f'Mod{mod_num}', net), (metric, "avg")] = avg
                        df.loc[(dim, f'Mod{mod_num}', net), (metric, "std")] = std

        print(df)

        df.to_csv(f'exp2-res_analysis_{train_type}_{risk}.csv', index=True)

        # compare tarnet and tarnet_single
        better_settings = []

        # Iterate over settings
        for mod_num in range(1,5):
            for dim in ["high", "low", "med"]:
                for metric in metrics:
                    # Fetch metrics for "tarnet" and "tarnet_single"
                    tarnet_avg = df.loc[(dim, f'Mod{mod_num}', 'tarnet'), (metric, 'avg')]
                    tarnet_single_avg = df.loc[(dim, f'Mod{mod_num}', 'tarnet_single'), (metric, 'avg')]

                    # print("tarnet_avg: ", tarnet_avg, "tarnet_single_avg: ", tarnet_single_avg)
                    # If "tarnet_single" is performing better, save the setting
                    if metric in lower_is_better_metrics and tarnet_single_avg < tarnet_avg:
                        better_settings.append((dim, f'Mod{mod_num}', metric))
                    if metric in higher_is_better_metrics and tarnet_single_avg > tarnet_avg:
                        better_settings.append((dim, f'Mod{mod_num}', metric))

        # Print out the settings where "tarnet_single" performed better
        with open(f'better_settings_{train_type}_{risk}.txt', 'w') as f:
            for setting in better_settings:
                f.write(f"For Dimension: {setting[0]}, Model: {setting[1]}, and Metric: {setting[2]}, 'tarnet_single' performed better.\n")
        # for setting in better_settings:
        #     # TODO: change this to save it to some file
        #     print(f"For Dimension: {setting[0]}, Model: {setting[1]}, and Metric: {setting[2]}, 'tarnet_single' performed better.")

