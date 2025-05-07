import os
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scienceplots
from PIL import Image
from tqdm import tqdm


def get_concat_h_resize(
    im1: Image.Image,
    im2: Image.Image,
    resample: int = Image.BICUBIC,
    resize_big_image: bool = True,
) -> Image.Image:
    if im1.height == im2.height:
        _im1, _im2 = im1, im2
    elif ((im1.height > im2.height) and resize_big_image) or (
        (im1.height < im2.height) and not resize_big_image
    ):
        _im1 = im1.resize(
            (int(im1.width * im2.height / im1.height), im2.height), resample=resample
        )
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize(
            (int(im2.width * im1.height / im2.height), im1.height), resample=resample
        )

    dst = Image.new("RGB", (_im1.width + _im2.width, _im1.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (_im1.width, 0))
    return dst


def get_concat_v_resize(
    im1: Image.Image,
    im2: Image.Image,
    resample: int = Image.BICUBIC,
    resize_big_image: bool = True,
) -> Image.Image:
    if im1.width == im2.width:
        _im1, _im2 = im1, im2
    elif ((im1.width > im2.width) and resize_big_image) or (
        (im1.width < im2.width) and not resize_big_image
    ):
        _im1 = im1.resize(
            (im2.width, int(im1.height * im2.width / im1.width)), resample=resample
        )
        _im2 = im2
    else:
        _im1 = im1
        _im2 = im2.resize(
            (im1.width, int(im2.height * im1.width / im2.width)), resample=resample
        )

    dst = Image.new("RGB", (_im1.width, _im1.height + _im2.height))
    dst.paste(_im1, (0, 0))
    dst.paste(_im2, (0, _im1.height))
    return dst


def merge_map_graph(
    png_id_length, map_folder_path, graph_folder_path, output_folder_path
):

    # グラフ保存用のフォルダがなければ作成
    os.makedirs(output_folder_path, exist_ok=True)

    for j in tqdm(range(png_id_length), desc="Merging map and graph"):
        img1 = Image.open(map_folder_path + "/draw" + str(j) + ".png")
        img2 = Image.open(graph_folder_path + "/draw" + str(j) + ".png")
        get_concat_h_resize(img1, img2).save(
            output_folder_path + "/draw" + str(j) + ".png"
        )


def create_movie(images_folder, output_folder_path, fps=24):

    # グラフ保存用のフォルダがなければ作成
    os.makedirs(output_folder_path, exist_ok=True)

    # 画像のファイル名を取得し、番号部分を抽出して数値順にソートする
    image_files = [
        os.path.join(images_folder, file)
        for file in sorted(
            os.listdir(images_folder),
            key=lambda x: int(x.split(".")[0].split("draw")[1]),
        )
        if file.startswith("draw") and file.endswith(".png")
    ]

    # 画像サイズを取得
    img = cv2.imread(image_files[0])
    height, width, _ = img.shape

    # 動画フォーマットを指定して動画ライターを作成
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Macで動作するMP4形式を指定
    out = cv2.VideoWriter(
        output_folder_path + "/output_video.mp4", fourcc, fps, (width, height)
    )

    # 画像を動画に書き込む
    for image_file in tqdm(image_files, desc="Creating movie"):
        img = cv2.imread(image_file)
        out.write(img)

    # 動画ライターを解放
    out.release()


def draw_map(
    typhoon_data_path,
    tpg_ship_result_path,
    storage_base_result_path,
    support_ship_1_result_path,
    support_ship_2_result_path,
    output_folder_path,
    stbase_position,
    spbase_position,
):

    # データの読み込み
    typhoon_data = pl.read_csv(typhoon_data_path)
    ship_typhoon_route_data = pl.read_csv(tpg_ship_result_path)
    stBASE_data = pl.read_csv(storage_base_result_path)
    spSHIP1_data = pl.read_csv(support_ship_1_result_path)
    spSHIP2_data = pl.read_csv(support_ship_2_result_path)

    UTC = timezone(timedelta(hours=+0), "UTC")

    # グラフ保存用のフォルダがなければ作成
    os.makedirs(output_folder_path, exist_ok=True)

    month = 0

    for j in tqdm(range(len(ship_typhoon_route_data)), desc="Drawing map"):

        current_time = ship_typhoon_route_data[j, "unixtime"]
        year = datetime.fromtimestamp(current_time, UTC).year

        if month != datetime.fromtimestamp(current_time, UTC).month:
            month = datetime.fromtimestamp(current_time, UTC).month
            wind_data = pl.read_csv(
                "data/wind_datas/era5_testdata_E180W90S0W90_"
                + str(int(year))
                + "_"
                + str(int(month))
                + ".csv",
                encoding="shift-jis",
            )
            # LONとLATが偶数の時のみのデータを抽出
            wind_data = wind_data.filter(wind_data["LON"] % 2 == 0)
            wind_data = wind_data.filter(wind_data["LAT"] % 2 == 0)
            wind_lon = wind_data[:]["LON"]  # 経度データ読み込み
            wind_lat = wind_data[:]["LAT"]  # 緯度データ読み込み
            wind_u = wind_data[:]["U10_E+_W-[m/s]"]
            wind_v = wind_data[:]["V10_N+_S-[m/s]"]

        # 地図の作成
        fig = plt.figure(figsize=(12, 16))  # プロット領域の作成（matplotlib）
        ax = fig.add_subplot(
            1, 1, 1, projection=ccrs.PlateCarree()
        )  # サブプロット作成時にcartopy呼び出し

        ax.set_facecolor("paleturquoise")
        land_h = cfeature.NaturalEarthFeature("physical", "land", "50m")
        ax.add_feature(land_h, color="g")
        ax.set_extent([120, 180, 0, 70], ccrs.Geodetic())

        for i in range(len(wind_lat)):
            # 風矢印表示
            size = 12
            vec_size = np.sqrt((wind_u[i]) ** 2 + (wind_v[i]) ** 2)
            Q_wind = ax.quiver(
                wind_lon[i],
                wind_lat[i],
                wind_u[i] / vec_size * size,
                wind_v[i] / vec_size * size,
                vec_size,
                cmap="YlOrRd",
                clim=(0, 15),
                width=0.003,
                scale=400.0,
            )

        # 中継貯蔵拠点&待機位置
        stbase_lat = stbase_position[0]
        stbase_lon = stbase_position[1]
        ax.plot(stbase_lon, stbase_lat, "crimson", markersize=15, marker="d")

        # 供給拠点
        spbase_lat = spbase_position[0]
        spbase_lon = spbase_position[1]
        ax.plot(spbase_lon, spbase_lat, "crimson", markersize=20, marker="*")

        # 陸地境界の設定
        # nonaggression_line_lat = [0,13,13,15,15,24,24,26,26,28,28,32.2,32.2,34,34,41.2,41.2,44,44,50,50]
        # nonaggression_line_lon = [127.5,127.5,125,125,123,123,126,126,130.1,130.1,132.4,132.4,137.2,137.2,143,143,149,149,156,156,180]
        # for i in range(len(nonaggression_line_lat)-1):
        #    ax.plot([nonaggression_line_lon[i],nonaggression_line_lon[i+1]],[nonaggression_line_lat[i],nonaggression_line_lat[i+1]],'red',linewidth=4)

        # 台風の作成
        unixtime = ship_typhoon_route_data[j, "unixtime"]

        TYdata = typhoon_data.filter((pl.col("unixtime") == unixtime))
        typhoon_lon = TYdata[:]["LON"]
        typhoon_lat = TYdata[:]["LAT"]
        if len(typhoon_lon) != 0:
            for k in range(len(typhoon_lon)):
                ax.plot(
                    typhoon_lon[k],
                    typhoon_lat[k],
                    "grey",
                    markersize=50,
                    marker="o",
                    alpha=0.8,
                )

        # 凡例の作成
        view_lon = 170
        text_lon = 173
        d_view_lat = 3
        view_ship_lon = 150
        view_ship_lat = 60
        view_lat_1 = 58
        lon_width = 2

        ship_text_lon = 154
        ship_text_lat = 63.5

        # 目標名の設定
        target = ship_typhoon_route_data[j, "TARGET LOCATION"]
        if ship_typhoon_route_data[j, "TARGET TYPHOON"] != 0:
            target = "Typhoon " + str(target)
        else:
            target = str(target)

        # 船速取得
        speed = str(format(ship_typhoon_route_data[j, "SHIP SPEED[kt]"], ".1f"))

        # 状態表示　移動・発電・待機
        if (
            (ship_typhoon_route_data[j, "SHIP SPEED[kt]"] == 0)
            & (ship_typhoon_route_data[j, "TIMESTEP POWER GENERATION[Wh]"] == 0)
            & (ship_typhoon_route_data[j, "TIMESTEP POWER CONSUMPTION[Wh]"] == 0)
        ):
            state = "Standby"
            iro = "lime"
        elif ship_typhoon_route_data[j, "TIMESTEP POWER GENERATION[Wh]"] > 0:
            state = "Power Generation"
            speed = speed + "(prov)"
            iro = "yellow"
        elif (ship_typhoon_route_data[j, "TIMESTEP POWER CONSUMPTION[Wh]"] > 0) or (
            ship_typhoon_route_data[j, "SHIP SPEED[kt]"] > 0
        ):
            state = "Moving"
            iro = "red"

        else:
            print("Error")

        # 船内蓄電量
        storage_per = str(
            format(ship_typhoon_route_data[j, "ONBOARD POWER STORAGE PER[%]"], ".1f")
        )

        # TPGship関連情報表示
        r = patches.Rectangle(
            xy=(145, 58), width=40, height=13, ec="k", fc="w", zorder=3
        )
        ax.add_patch(r)

        ax.quiver(
            view_ship_lon,
            view_ship_lat,
            0,
            55,
            color=iro,
            edgecolor="k",
            linewidth=1.0,
            headwidth=30,
            headlength=50,
            headaxislength=50,
            width=0.02,
            scale=400.0,
            zorder=4,
        )
        ax.text(
            ship_text_lon,
            ship_text_lat + 5.2,
            "TPGship prototype",
            size=25,
            color="black",
            zorder=4,
        )
        ax.text(
            ship_text_lon,
            ship_text_lat + 2.5,
            "Target          : " + target,
            size=18,
            color="black",
            zorder=4,
        )
        ax.text(
            ship_text_lon,
            ship_text_lat - 0.0,
            "States          : " + state,
            size=18,
            color="black",
            zorder=4,
        )
        ax.text(
            ship_text_lon,
            ship_text_lat - 2.5,
            "Speed[kt]    : " + speed,
            size=18,
            color="black",
            zorder=4,
        )
        ax.text(
            ship_text_lon,
            ship_text_lat - 5.0,
            "Storage[%]  : " + storage_per,
            size=18,
            color="black",
            zorder=4,
        )

        # 日数記録
        date = str(datetime.fromtimestamp(unixtime, UTC))
        r = patches.Rectangle(
            xy=(145, 54), width=24, height=4, ec="k", fc="w", zorder=3
        )
        ax.add_patch(r)
        ax.text(
            145.5,
            view_lat_1 - 0.8 * d_view_lat - 0.4,
            date,
            size=16.5,
            color="black",
            zorder=4,
        )

        # 台風の凡例表示
        r = patches.Rectangle(
            xy=(167, 54), width=20, height=4, ec="k", fc="w", zorder=3
        )
        ax.add_patch(r)

        ax.plot(
            view_lon,
            view_lat_1 - 0.7 * d_view_lat,
            "grey",
            markersize=30,
            marker="o",
            zorder=4,
        )
        ax.text(
            text_lon,
            view_lat_1 - 0.7 * d_view_lat - 0.5,
            "Typhoon",
            size=15,
            color="black",
            zorder=4,
        )

        # 中継貯蔵拠点の凡例表示
        r = patches.Rectangle(
            xy=(145, 46), width=20, height=8, ec="k", fc="w", zorder=3
        )
        stbase_state = str(stBASE_data[j, "BRANCH CONDITION"])
        stbase_storage = str(format(stBASE_data[j, "STORAGE PER[%]"], ".1f"))
        ax.add_patch(r)
        ax.plot(
            147.0,
            view_lat_1 - 2.4 * d_view_lat,
            "crimson",
            markersize=30,
            marker="d",
            zorder=4,
        )
        ax.text(
            149.0,
            view_lat_1 - 1.8 * d_view_lat - 0.5,
            "Storage Base",
            size=20,
            color="black",
            zorder=4,
        )
        ax.text(
            149.0,
            view_lat_1 - 2.3 * d_view_lat - 0.5,
            "States : " + stbase_state,
            size=14,
            color="black",
            zorder=4,
        )
        ax.text(
            149.0,
            view_lat_1 - 2.8 * d_view_lat - 0.5,
            "Storage[%]  : " + stbase_storage,
            size=14,
            color="black",
            zorder=4,
        )

        # 供給拠点の凡例表示
        ax.plot(
            147.0,
            view_lat_1 - 3.6 * d_view_lat,
            "crimson",
            markersize=20,
            marker="*",
            zorder=4,
        )
        ax.text(
            149.0,
            view_lat_1 - 3.6 * d_view_lat - 0.5,
            "Supply Base",
            size=15,
            color="black",
            zorder=4,
        )

        # 運搬船の凡例表示
        r = patches.Rectangle(
            xy=(165, 46), width=20, height=8, ec="k", fc="w", zorder=3
        )
        ax.add_patch(r)
        spship1_storage = str(format(spSHIP1_data[j, "STORAGE PER[%]"], ".1f"))
        spship2_storage = str(format(spSHIP2_data[j, "STORAGE PER[%]"], ".1f"))
        ax.quiver(
            166.5,
            51,
            0,
            15,
            color="navy",
            edgecolor="k",
            linewidth=0.25,
            headwidth=6,
            headlength=10,
            headaxislength=10,
            width=0.02,
            scale=400.0,
            zorder=4,
        )
        ax.text(
            168.0,
            view_lat_1 - 1.7 * d_view_lat - 0.5,
            "support_ship_1",
            size=15,
            color="black",
            zorder=4,
        )
        ax.text(
            168.0,
            view_lat_1 - 2.2 * d_view_lat - 0.5,
            "Storage[%]  : " + spship1_storage,
            size=13,
            color="black",
            zorder=4,
        )
        ax.quiver(
            166.5,
            46.7,
            0,
            15,
            color="purple",
            edgecolor="k",
            linewidth=0.25,
            headwidth=6,
            headlength=10,
            headaxislength=10,
            width=0.02,
            scale=400.0,
            zorder=4,
        )
        ax.text(
            168.0,
            view_lat_1 - 3.1 * d_view_lat - 0.5,
            "support_ship_2",
            size=15,
            color="black",
            zorder=4,
        )
        ax.text(
            168.0,
            view_lat_1 - 3.6 * d_view_lat - 0.5,
            "Storage[%]  : " + spship2_storage,
            size=13,
            color="black",
            zorder=4,
        )

        # 運搬船1の作図
        spship1_lat = spSHIP1_data[j, "LAT"]
        spship1_lon = spSHIP1_data[j, "LON"]
        if j == 0:
            u = 0
            v = 1
        else:
            spship1_lat_2 = spSHIP1_data[j - 1, "LAT"]
            spship1_lon_2 = spSHIP1_data[j - 1, "LON"]
            u = spship1_lon - spship1_lon_2
            v = spship1_lat - spship1_lat_2
            if (u == 0) & (v == 0):
                u = 0
                v = 1

        size = 10

        vec_size = np.sqrt(u**2 + v**2)
        ax.quiver(
            spship1_lon,
            spship1_lat,
            u / vec_size * size,
            v / vec_size * size,
            color="navy",
            edgecolor="k",
            linewidth=0.25,
            headlength=6,
            headaxislength=6,
            width=0.004,
            scale=400.0,
            zorder=4.5,
        )

        # 運搬船2の作図
        spship2_lat = spSHIP2_data[j, "LAT"]
        spship2_lon = spSHIP2_data[j, "LON"]
        if j == 0:
            u = 0
            v = 1
        else:
            spship2_lat_2 = spSHIP2_data[j - 1, "LAT"]
            spship2_lon_2 = spSHIP2_data[j - 1, "LON"]
            u = spship2_lon - spship2_lon_2
            v = spship2_lat - spship2_lat_2
            if (u == 0) & (v == 0):
                u = 0
                v = 1

        size = 10

        vec_size = np.sqrt(u**2 + v**2)
        ax.quiver(
            spship2_lon,
            spship2_lat,
            u / vec_size * size,
            v / vec_size * size,
            color="purple",
            edgecolor="k",
            linewidth=0.25,
            headlength=6,
            headaxislength=6,
            width=0.004,
            scale=400.0,
            zorder=4.5,
        )

        # TPGshipの作図
        ship_lat = ship_typhoon_route_data[j, "TPGSHIP LAT"]
        ship_lon = ship_typhoon_route_data[j, "TPGSHIP LON"]
        if j == 0:
            u = 0
            v = 1
        else:
            ship_lat_2 = ship_typhoon_route_data[j - 1, "TPGSHIP LAT"]
            ship_lon_2 = ship_typhoon_route_data[j - 1, "TPGSHIP LON"]
            u = ship_lon - ship_lon_2
            v = ship_lat - ship_lat_2
            if (u == 0) & (v == 0):
                u = 0
                v = 1

        size = 10

        vec_size = np.sqrt(u**2 + v**2)
        ax.quiver(
            ship_lon,
            ship_lat,
            u / vec_size * size,
            v / vec_size * size,
            color=iro,
            edgecolor="k",
            linewidth=0.25,
            headlength=6,
            headaxislength=6,
            width=0.004,
            scale=400.0,
            zorder=5,
        )

        # j = j_ori

        plt.savefig(output_folder_path + "/draw" + str(j) + ".png")
        plt.close(fig)

        im = Image.open(output_folder_path + "/draw" + str(j) + ".png")

        # 環境ごとに調整
        im_crop = im.crop((150, 250, 1080, 1370))
        im_crop.save(output_folder_path + "/draw" + str(j) + ".png", quality=100)


def draw_graph(
    tpg_ship_result_path,
    storage_base_result_path,
    supply_base_result_path,
    # support_ship_1_result_path,
    # support_ship_2_result_path,
    output_folder_path,
):

    # データの読み込み
    TPGship_data = pl.read_csv(tpg_ship_result_path)
    stBASE_data = pl.read_csv(storage_base_result_path)
    spBASE_data = pl.read_csv(supply_base_result_path)

    # グラフ保存用のフォルダがなければ作成
    os.makedirs(output_folder_path, exist_ok=True)

    UTC = timezone(timedelta(hours=+0), "UTC")

    # データの整理
    totalgene = spBASE_data["TOTAL SUPPLY[Wh]"]
    tg = []
    for i in range(len(totalgene)):
        tg.append(totalgene[i] / 10**9)

    onboardene = TPGship_data["ONBOARD ENERGY STORAGE[Wh]"]
    obe = []
    for i in range(len(onboardene)):
        obe.append(onboardene[i] / 10**9)

    basestorage = stBASE_data["STORAGE[Wh]"]
    base_data = []
    for i in range(len(basestorage)):
        base_data.append(basestorage[i] / 10**9)

    day = TPGship_data["unixtime"]
    daylist = []
    for i in range(len(day)):
        daylist.append((day[i] - day[0]) / 86400)

    # TPGship_dataから運用日数を取得
    start_time = TPGship_data["unixtime"][0]
    end_time = TPGship_data["unixtime"][len(TPGship_data) - 1]
    start_date = datetime.fromtimestamp(start_time, UTC)
    end_date = datetime.fromtimestamp(end_time, UTC)
    operation_days = (end_date - start_date).days

    # TPGship_dataのONBOARD POWER STORAGE PER[%]の0でない値と対応するONBOARD ENERGY STORAGE[Wh]の値からONBOARD POWER STORAGE PER[%]が100％の時のONBOARD ENERGY STORAGE[Wh]を計算する
    onboardene = TPGship_data["ONBOARD ENERGY STORAGE[Wh]"]
    onboardper = TPGship_data["ONBOARD POWER STORAGE PER[%]"]
    for i in range(len(onboardper)):
        if onboardper[i] != 0:
            onboardene_100 = onboardene[i] / (onboardper[i] / 100)
            break

    # "TOTAL POWER GENERATION[Wh]"の最大値を取得し、GWhに変換して50GWh刻みに切り上げる
    max_gene = max(totalgene)
    max_gene = int(max_gene / 10**9)
    max_gene = (max_gene // 50 + 1) * 50

    # stBASE_dataのSTORAGE PER[%]の0でない値と対応するSTORAGE[Wh]の値からSTORAGE PER[%]が100％の時のSTORAGE[Wh]を計算する
    basestorage = stBASE_data["STORAGE[Wh]"]
    basestorageper = stBASE_data["STORAGE PER[%]"]
    for i in range(len(basestorageper)):
        if basestorageper[i] != 0:
            basestorage_100 = basestorage[i] / (basestorageper[i] / 100)
            break

    # グラフの表示
    plt.style.use(["science", "no-latex", "high-vis", "grid"])  # latexなしで動くように
    plt.rcParams["font.size"] = 20

    fig = plt.figure(figsize=(10, 16))  # プロット領域の作成（matplotlib）

    ax1 = fig.add_subplot(3, 1, 1)
    ax1_xmin, ax1_xmax = 0, operation_days
    ax1_ymin, ax1_ymax = 0, max_gene
    ax1.set_xlim(xmin=ax1_xmin, xmax=ax1_xmax)  # x軸の範囲を指定
    ax1.set_ylim(ymin=ax1_ymin, ymax=ax1_ymax)  # y軸の範囲を指定
    ax1.set(xlabel="Operation Time[Day]")  # x軸のラベル
    ax1.set(ylabel="Total EC Supply[GWh]")  # y軸のラベル
    ax1.plot(daylist, tg, label="TOTAL", linewidth=3)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2_xmin, ax2_xmax = 0, operation_days
    ax2_ymin, ax2_ymax = 0, onboardene_100 / 10**9 + 5
    ax2.set_xlim(xmin=ax2_xmin, xmax=ax2_xmax)  # x軸の範囲を指定
    ax2.set_ylim(ymin=ax2_ymin, ymax=ax2_ymax)  # y軸の範囲を指定
    ax2.set(xlabel="Operation Time[Day]")  # x軸のラベル
    ax2.set(ylabel="Onboard Power Storage[GWh]")  # y軸のラベル
    ax2.plot(daylist, obe, label="ONBOARD", linewidth=3)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3_xmin, ax3_xmax = 0, operation_days
    ax3_ymin, ax3_ymax = 0, basestorage_100 / 10**9 + 5
    ax3.set_xlim(xmin=ax3_xmin, xmax=ax3_xmax)  # x軸の範囲を指定
    ax3.set_ylim(ymin=ax3_ymin, ymax=ax3_ymax)  # y軸の範囲を指定
    ax3.set(xlabel="Operation Time[Day]")  # x軸のラベル
    ax3.set(ylabel="EC at Storage Base[GWh]")  # y軸のラベル
    ax3.plot(daylist, base_data, label="BASE OPERATION", linewidth=3)

    for j in tqdm(range(len(TPGship_data)), desc="Drawing graph"):
        line1 = ax1.vlines(daylist[j], ax1_ymin, ax1_ymax)
        line2 = ax2.vlines(daylist[j], ax2_ymin, ax2_ymax)
        line3 = ax3.vlines(daylist[j], ax3_ymin, ax3_ymax)

        plt.savefig(output_folder_path + "/draw" + str(j) + ".png")

        line1.remove()
        line2.remove()
        line3.remove()
