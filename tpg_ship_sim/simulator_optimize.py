import os
from datetime import datetime, timedelta, timezone

import polars as pl
import pytz
from dateutil import tz
from tqdm import tqdm

from tpg_ship_sim.model import forecaster, tpg_ship


def get_TY_start_time(typhoon_data_path):
    """
    ############################## def get_TY_start_time ##############################

    [ 説明 ]

    この関数は台風の発生時刻を取得するための関数です。

    本来は発生したごとに逐次記録すれば良いのですが、そのプログラムを作っても嵩張るだけだと思ったので、

    予報期間に関係なく発生時間は取得できるものとして辞書化することにしました。台風番号をキーに、最初の発生時間を値を引けます。


    ##############################################################################

    引数 :
        typhoon_data_path (dataflame) : 過去の台風のデータのパス

    戻り値 :
        TY_occurrence_time (list) : 各台風の発生時刻のリスト

    #############################################################################
    """

    # CSVファイルの読み込み
    df = pl.read_csv(typhoon_data_path)

    # 発生時間（ユニックスタイム）でソート
    df = df.sort("unixtime")

    # 台風番号をキーに、最初の発生時間を値とする辞書を作成
    typhoon_start_times = df.groupby("TYPHOON NUMBER").agg(pl.col("unixtime").min())

    # 辞書に変換
    typhoon_start_times_dict = {
        row["TYPHOON NUMBER"]: row["unixtime"]
        for row in typhoon_start_times.iter_rows(named=True)
    }

    return typhoon_start_times_dict


############################################################################################


def simulate(
    simulation_start_time,
    simulation_end_time,
    tpg_ship_1,  # TPG ship
    typhoon_path_forecaster,  # Forecaster
    st_base,  # Storage base
    sp_base,  # Supply base
    support_ship_1,  # Support ship 1
    support_ship_2,  # Support ship 2
    typhoon_data_path,
    output_folder_path,
) -> None:

    # タイムステップ
    time_step = 6

    # UTCタイムゾーンの設定
    UTC = timezone(timedelta(hours=+0), "UTC")

    # 開始日時の生成
    datetime_1_1 = datetime.strptime(
        simulation_start_time, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=pytz.utc)

    # 終了日時の生成
    datetime_12_31 = datetime.strptime(
        simulation_end_time, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=pytz.utc)
    unixtime_12_31 = int(datetime_12_31.timestamp())

    # タイムスタンプから現在の年月を取得
    current_time = int(datetime_1_1.timestamp())
    year = datetime.fromtimestamp(current_time, UTC).year
    month = datetime.fromtimestamp(current_time, UTC).month

    # unixtimeでの時間幅
    time_step_unix = 3600 * time_step

    # 繰り返しの回数
    record_count = int((unixtime_12_31 - current_time) / (time_step_unix) + 1)

    # 台風データ設定
    typhoon_path_forecaster.year = year
    # typhoon_data = pl.read_csv(
    #     "data/" + "typhoon_data_"
    #     # + str(int(time_step))
    #     # + "hour_intervals_verpl/table"
    #     + str(year) + "_" + str(int(time_step)) + "_interval.csv",
    #     # encoding="shift-jis",
    # )
    typhoon_data = pl.read_csv(typhoon_data_path)
    typhoon_path_forecaster.original_data = typhoon_data

    # 風データ設定
    wind_data = pl.read_csv(
        "data/wind_datas/era5_testdata_E180W90S0W90_"
        + str(int(year))
        + "_"
        + str(int(month))
        + ".csv"
    )

    # 発電船パラメータ設定
    tpg_ship_1.forecast_time = typhoon_path_forecaster.forecast_time

    # 拠点位置に関する設定
    # 発電船拠点位置
    tpg_ship_1.base_lat = st_base.locate[0]
    tpg_ship_1.base_lon = st_base.locate[1]

    tpg_ship_1.TY_start_time_list = get_TY_start_time(typhoon_data_path)

    tpg_ship_1.set_initial_states()

    #####################################  出力用の設定  ############################################
    unix = []
    date = []

    ####################### TPG ship ##########################
    tpg_ship_1.set_outputs()

    ####################### Storage base ##########################
    st_base.set_outputs()

    ####################### Supply base ##########################
    sp_base.set_outputs()

    ####################### Support ship ##########################
    support_ship_1.set_outputs()

    support_ship_2.set_outputs()

    #######################################  出力用リストへ入力  ###########################################
    unix.append(current_time)
    date.append(datetime.fromtimestamp(unix[-1], UTC))

    tpg_ship_1.outputs_append()
    GS_data = tpg_ship_1.get_outputs(unix, date)

    ####################### Storage base ##########################
    st_base.outputs_append()
    stBASE_data = st_base.get_outputs(unix, date)

    ####################### Supply base ##########################
    sp_base.outputs_append()
    spBASE_data = sp_base.get_outputs(unix, date)

    ####################### Support ship ##########################
    support_ship_1.outputs_append()
    support_ship_2.outputs_append()

    spSHIP1_data = support_ship_1.get_outputs(unix, date)
    spSHIP2_data = support_ship_2.get_outputs(unix, date)

    # for data_num in tqdm(range(record_count), desc="Simulating..."): # tqdmを使うと進捗が表示される
    for data_num in range(record_count):

        year = datetime.fromtimestamp(current_time, UTC).year

        # 月毎の風データの取得
        if month != datetime.fromtimestamp(current_time, UTC).month:
            month = datetime.fromtimestamp(current_time, UTC).month
            wind_data = pl.read_csv(
                "data/wind_datas/era5_testdata_E180W90S0W90_"
                + str(int(year))
                + "_"
                + str(int(month))
                + ".csv"
            )

        # 予報データ取得
        tpg_ship_1.forecast_data = typhoon_path_forecaster.create_forecast(
            time_step, current_time
        )

        # timestep後の発電船の状態を取得
        tpg_ship_1.get_next_ship_state(
            year, current_time, time_step, wind_data, st_base
        )

        # timestep後の中継貯蔵拠点と運搬船の状態を取得
        st_base.operation_base(
            tpg_ship_1, support_ship_1, support_ship_2, year, current_time, time_step
        )

        # timestep後の供給拠点の状態を取得
        sp_base.operation_base(
            tpg_ship_1, support_ship_1, support_ship_2, year, current_time, time_step
        )

        # timestep後の時刻の取得
        current_time = current_time + time_step_unix

        #######################################  出力用リストへ入力  ###########################################
        unix.append(current_time)
        date.append(datetime.fromtimestamp(unix[-1], UTC))

        tpg_ship_1.outputs_append()
        GS_data = tpg_ship_1.get_outputs(unix, date)

        ####################### storageBASE ##########################
        st_base.outputs_append()
        stBASE_data = st_base.get_outputs(unix, date)

        ####################### supplyBASE ##########################
        sp_base.outputs_append()
        spBASE_data = sp_base.get_outputs(unix, date)

        ####################### supportSHIP ##########################
        support_ship_1.outputs_append()
        support_ship_2.outputs_append()

        spSHIP1_data = support_ship_1.get_outputs(unix, date)
        spSHIP2_data = support_ship_2.get_outputs(unix, date)


############################################################################################
