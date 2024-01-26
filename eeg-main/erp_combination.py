import os
from typing import List
from datetime import datetime

import pandas as pd
import time

from src.task import combination_task
from src.analysis import AnalyzeEEG
from src.plot import PlotEEG
from src.recommendation import recommend_combination

def erp_combination(
    screen_width: int,
    screen_height: int,
    fs: int,
    channels: List,
    isi: int,
    top_image_path: str,
    clothes_type: str,
    image_folder: str,
    num_trials: int,
    num_images: int,
    event_save_path: str,
    result_dir: str,
    lowcut: float = 1.0,
    highcut: float = 30.0,
    tmin: float = -0.2,
    tmax: float = 1.0,
):
    result = []
    cul = 4
    today = str(datetime.now().date())
    if not os.path.exists(f"./data/{today}"):
        os.makedirs(f"./data/{today}")
    if not os.path.exists(f"./event/{today}"):
        os.makedirs(f"./event/{today}")

    labm = ["not","bottoms","shoes"]

    for i in range (1, 3):
        if not os.path.exists(f"./data/{today}/{i}"):
            os.makedirs(f"./data/{today}/{i}")
        if not os.path.exists(f"./event/{today}/{i}"):
            os.makedirs(f"./event/{today}/{i}")
        

        combination_task(
            screen_width=screen_width,
            screen_height=screen_height,
            isi=isi,
            top_image_path=top_image_path,
            image_folder=f"{args.image_path}/{labm[i]}",
            num_trials=num_trials,
            num_images=num_images,
            event_save_path=f"{event_save_path}/{today}/{i}",
            clothes_type=labm[i]
        )

        rawdata_folders = os.listdir("C:/MAVE_RawData")

        text_file_name = f"C:/MAVE_RawData/{rawdata_folders[-1]}/Rawdata.txt"
        data_df = pd.read_csv(text_file_name, delimiter="\t")

        record_start_time = data_df.iloc[0, 0]
        hour = str(record_start_time).split(":")[0]
        min = str(record_start_time).split(":")[1]
        sec = str(record_start_time).split(":")[2].split(".")[0]

        data_df = data_df[channels]
        data_file_path = f"./data/{today}/{i}/Rawdata_{hour}.{min}.{sec}.csv"
        data_df.to_csv(data_file_path, index=False)

        event_paths = os.listdir(f"./event/{today}/{i}")
        event_file = f"./event/{today}/{i}/{event_paths[-1]}"

        analyze_eeg = AnalyzeEEG(channels=channels, fs=fs)
        eeg, eeg_times, avg_evoked_list, times_list = analyze_eeg.analyze_erp(
            eeg_filename=data_file_path,
            event_filename=event_file,
            result_dir=result_dir,
            num_types=num_images,
            lowcut=lowcut,
            highcut=highcut,
            tmin=tmin,
            tmax=tmax,
        )

        plot_eeg = PlotEEG(
            channels=channels,
            result_dir=result_dir,
            is_show=False,
            is_save=True,
            eeg=eeg,
            eeg_times=eeg_times,
            eeg_filename="eeg_raw",
        )

        plot_eeg.plot_eeg()
        if labm[i] == "bottoms":
            for j in range(num_images):
                plot_eeg.plot_electrode(
                    avg_evoked_list[j],
                    times_list[j],
                    filename=f"bottoms_{j+1}_electrode",
                )
        elif labm[i] == "shoes":
            for k in range(num_images):
                plot_eeg.plot_electrode(
                    avg_evoked_list[k],
                    times_list[k],
                    filename=f"shoes_{k+1}_electrode",
                )
        else:
            raise ValueError("Invalid clothes type")

        result.append(recommend_combination(
                avg_evoked_list=avg_evoked_list,
                times_list=times_list,
                channels=channels,
                image_folder=image_folder,
                clothes_type=clothes_type,
        ))

        print("10초 뒤에 다음 테스크가 시작됩니다.")
        time.sleep(10)
        
    if 4 in result[0] and 4 in result[1]:
        print("당신이 범인일 확률은 100프로입니다.")
    elif 4 in result[0] or 4 in result[1]:
        print("당신이 범인일 확률은 50프로입니다")
    else:
        print("당신이 범인일 확률은 0프로 입니다.")


if __name__ == "__main__":
    import argparse
    import ast

    def parse_list(string: str):
        try:
            parsed_list = ast.literal_eval(string)
            if isinstance(parsed_list, list):
                return parsed_list
            else:
                raise argparse.ArgumentTypeError("Invalid list format")
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError("Invalid list format")

    parser = argparse.ArgumentParser(
        description="Insert arguments for function of erp combination"
    )
    parser.add_argument(
        "--screen_width",
        type=int,
        default=2880,
        help="Set screen width of combination task",
    )
    parser.add_argument(
        "--screen_height",
        type=int,
        default=1800,
        help="Set screen height of combination task",
    )
    parser.add_argument(
        "--fs", type=int, default=256, help="Get resolution of EEG device"
    )
    parser.add_argument(
        "--channels",
        type=parse_list,
        default="['EEG_Fp1', 'EEG_Fp2']",
        help="Get channels of EEG device",
    )
    parser.add_argument(
        "--isi",
        type=int,
        default=1000,
        help="Set inter-stimulus interval of combination task",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./images",
        help="Get image data path to use in the task",
    )
    parser.add_argument(
        "--tops_order",
        type=int,
        default=1,
        help="Set order of upper clothes to use in the task",
    )
    parser.add_argument(
        "--clothes_type",
        type=str,
        default="bottoms",
        help="Set type of clothes to use in the task",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="Set number of trials to use in the task",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Set number of clothes to use in the task",
    )
    parser.add_argument(
        "--event_save_path",
        type=str,
        default="./event",
        help="Set a record of events file saving path",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./plot",
        help="Set a EEG, ERP plots saving path",
    )
    parser.add_argument(
        "--result_dir_num",
        type=int,
        default=0,
        help="Set a EEG, ERP plots detailed saving path",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default= 1, # 1
        help="Set butter filter lowcut to get ERP",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=30.0, # 30
        help="Set butter filter highcut to get ERP",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=-0.2,
        help="Set epoch tmin to get ERP",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=1.0,
        help="Set epoch tmax to get ERP",
    )
    args = parser.parse_args()

    erp_combination(
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        fs=args.fs,
        channels=args.channels,
        isi=args.isi,
        top_image_path=f"{args.image_path}/tops/T{args.tops_order}.jpg",
        image_folder=f"{args.image_path}/{args.clothes_type}",
        clothes_type=f"{args.clothes_type}",
        num_trials=args.num_trials,
        num_images=args.num_images,
        event_save_path=f"{args.event_save_path}",
        result_dir=f"{args.result_dir}/{args.clothes_type}/{args.result_dir_num}",
        lowcut=args.lowcut,
        highcut=args.highcut,
        tmin=args.tmin,
        tmax=args.tmax,
    )
