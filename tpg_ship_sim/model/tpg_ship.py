import math

import numpy as np
import polars as pl
from geographiclib.geodesic import Geodesic
from geopy.distance import geodesic
from geopy.point import Point


class TPG_ship:
    """
    ############################### class TPGship ###############################

    [ 説明 ]

    このクラスは台風発電船を作成するクラスです。台風発電船の行動とその行動をするための条件を定義します。

    台風発電船自体の能力や状態量もここで定義されることになります。

    ##############################################################################

    引数 :
        year (int) : シミュレーションを行う年
        time_step (int) : シミュレーションにおける時間の進み幅[hours]
        current_time (int) : シミュレーション上の現在時刻(unixtime)

    属性 :
        max_storage (float) : 台風発電船の蓄電容量の上限値
        generator_rated_output_w (float) : 台風発電船の定格出力
        max_speed_power (float) : 付加物のない船体を最大船速で進めるのに必要な出力


        storage (float) : 台風発電船のその時刻での蓄電量
        storage_percentage (float) : 台風発電船のその時刻での蓄電量の割合
        gene_elect (float) : その時刻での発電量
        gene_carrier (float) : その時刻でのエネルギーキャリア生成量
        loss_elect (float) : その時刻での消費電力量
        ship_state (int) : 台風発電船の状態。通常航行、待機=0,発電状態=1,台風追従=2,台風低速追従=2.5,拠点回航=3,待機位置回航=4。
        total_gene_elect (float) : その時刻までの合計発電量
        total_gene_carrier (float) : その時刻まで電力を用いて生成したエネルギーキャリアの合計量
        total_loss_elect (float) : その時刻までの合計消費電力量
        total_gene_time (int) : その時刻までの合計発電時間
        total_loss_time (int) : その時刻までの合計電力消費時間

        speed_kt (float) : その時刻での台風発電船の船速(kt)
        target_name (str) : 目標地点の名前。台風の場合は番号の文字列入力。
        base_lat (float) : 拠点の緯度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        base_lon (float) : 拠点の経度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        standby_lat (float) : 待機位置の緯度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        standby_lon (float) : 待機位置の経度　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        ship_lat (float) : その時刻での台風発電船の緯度
        ship_lon (float) : その時刻での台風発電船の経度
        target_lat (float) : 目標地点の緯度
        target_lon (float) : 目標地点の経度
        target_distance (float) : 台風発電船から目標地点までの距離(km)
        target_TY (int) : 追従対象の台風の番号の整数入力。ない場合は0が入る。
        go_base (int) : 1の時その時の蓄電容量によらず拠点に帰る。
        next_TY_lat (float) : time_step後の目標台風の緯度。ない場合は経度と共に0
        next_TY_lon (float) : time_step後の目標台風の経度。ない場合は緯度と共に0
        next_ship_TY_dis (float) : time_step後の目標台風と台風発電船の距離(km)。ない場合はNaN。
        brance_condition (str) : 台風発電船が行動分岐のどの分岐になったかを示す

        distance_judge_hours (int) : 追従判断基準時間。発電船にとって台風が遠いか近いかを判断する基準。　※本プログラムでは使用しない
        judge_energy_storage_per (int) : 発電船が帰港判断をする蓄電割合。
        typhoon_effective_range (float) : 発電船が台風下での航行となる台風中心からの距離[km]
        govia_base_judge_energy_storage_per (int) : 発電船が拠点経由で目的地に向かう判断をする蓄電割合。
        judge_direction (float) : 発電船が2つの目的地の方位差から行動を判断する時の基準値[度]
        standby_via_base (int) : 待機位置へ拠点を経由して向かう場合のフラグ
        judge_time_times (float) : 台風の補足地点に発電船が最大船速で到着する時間に対し台風が到着する時間が「何倍」である時追うと判断するのかの基準値

        normal_ave_speed (float) : 平常時の平均船速(kt)
        max_speed (float) : 最大船速(kt)
        TY_tracking_speed (float) : 台風を追いかける時の船速(kt)
        speed_kt (float) : その時の船速(kt)

        forecast_data (dataflame) : 各時刻の台風の予想座標がわかるデータ。台風番号、時刻、座標を持つ　※Forecasterからもらう必要がある。
        TY_start_time_list (list) : 全ての台風の発生時刻のリスト　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。
        forecast_weight (float) : 台風を評価する際の式で各項につける重みの数値。他の項は(100-forecast_weight)。　※現段階では外部から入力が必要。調査で適当な値が求まったらそれを初期代入する予定。



    #############################################################################
    """

    # コスト関連
    building_cost = 0
    maintenance_cost = 0
    carrier_cost = 0

    ####################################  パラメータ  ######################################

    def __init__(
        self,
        initial_position,
        hull_num,
        storage_method,
        max_storage_wh,
        electric_propulsion_max_storage_wh,
        trust_efficiency,
        carrier_to_elect_efficiency,
        elect_to_carrier_efficiency,
        generator_turbine_radius,
        generator_efficiency,
        generator_drag_coefficient,
        generator_pillar_chord,
        generator_pillar_max_tickness,
        generator_pillar_width,
        generator_num,
        sail_area,
        sail_space,
        sail_num,
        sail_steps,
        ship_return_speed_kt,
        ship_max_speed_kt,
        forecast_weight,
        typhoon_effective_range,
        govia_base_judge_energy_storage_per,
        judge_time_times,
        operational_reserve_percentage,
        standby_position,
    ) -> None:
        self.ship_lat = initial_position[0]
        self.ship_lon = initial_position[1]
        self.hull_num = hull_num
        self.storage_method = storage_method
        self.max_storage = max_storage_wh

        self.electric_propulsion_max_storage_wh = electric_propulsion_max_storage_wh
        self.trust_efficiency = trust_efficiency
        self.carrier_to_elect_efficiency = carrier_to_elect_efficiency
        self.elect_to_carrier_efficiency = elect_to_carrier_efficiency
        self.generator_turbine_radius = generator_turbine_radius
        self.generator_efficiency = generator_efficiency
        self.generator_drag_coefficient = generator_drag_coefficient
        self.generator_pillar_chord = generator_pillar_chord
        self.generator_pillar_max_tickness = generator_pillar_max_tickness
        self.generator_pillar_width = generator_pillar_width
        self.generator_num = generator_num
        self.sail_area = sail_area
        self.sail_space = sail_space
        self.sail_steps = sail_steps
        self.sail_num = sail_num
        self.sail_weight = 120 * (self.sail_area / 880)
        self.nomal_ave_speed = ship_return_speed_kt
        self.max_speed = ship_max_speed_kt

        self.forecast_weight = forecast_weight
        self.typhoon_effective_range = typhoon_effective_range
        self.govia_base_judge_energy_storage_per = govia_base_judge_energy_storage_per
        self.judge_time_times = judge_time_times
        self.operational_reserve = self.max_storage * (
            operational_reserve_percentage / 100
        )
        self.operational_reserve_percentage = operational_reserve_percentage
        self.standby_lat = standby_position[0]
        self.standby_lon = standby_position[1]

    # __init__で定義されたパラメータをデータフレームとして記録する関数
    def get_outputs_for_evaluation(self):
        """
        ############################ def get_outputs_for_evaluation ############################

        [ 説明 ]

        台風発電船のパラメータをデータフレームとして記録する関数です。

        後述するset_initial_states関数の実行後に実行する必要があります。

        ##############################################################################

        """

        data = pl.DataFrame(
            {
                "Base_lat": [self.base_lat],
                "Base_lon": [self.base_lon],
                "Standby_lat": [self.standby_lat],
                "Standby_lon": [self.standby_lon],
                "hull_num": [self.hull_num],
                "hull_L_oa": [self.hull_L_oa],
                "hull_B": [self.hull_B],
                "storage_method": [self.storage_method],
                "max_storage": [self.max_storage],
                "electric_propulsion_max_storage_wh": [
                    self.electric_propulsion_max_storage_wh
                ],
                "trust_efficiency": [self.trust_efficiency],
                "carrier_to_elect_efficiency": [self.carrier_to_elect_efficiency],
                "elect_to_carrier_efficiency": [self.elect_to_carrier_efficiency],
                "generator_turbine_radius": [self.generator_turbine_radius],
                "generator_efficiency": [self.generator_efficiency],
                "generator_drag_coefficient": [self.generator_drag_coefficient],
                "generator_pillar_chord": [self.generator_pillar_chord],
                "generator_pillar_max_tickness": [self.generator_pillar_max_tickness],
                "generator_pillar_width": [self.generator_pillar_width],
                "generator_num": [self.generator_num],
                "generator_rated_output_w": [self.generator_rated_output_w],
                "sail_num": [self.sail_num],
                "sail_width": [self.sail_width],
                "sail_area": [self.sail_area],
                "sail_space": [self.sail_space],
                "sail_steps": [self.sail_steps],
                "sail_weight": [self.sail_weight],
                "num_sails_per_row": [self.num_sails_per_row],
                "num_sails_rows": [self.num_sails_rows],
                "nomal_ave_speed": [self.nomal_ave_speed],
                "max_speed": [self.max_speed],
                "generating_speed_kt": [self.generating_speed_kt],
                "forecast_weight": [self.forecast_weight],
                "typhoon_effective_range": [self.typhoon_effective_range],
                "govia_base_judge_energy_storage_per": [
                    self.govia_base_judge_energy_storage_per
                ],
                "judge_time_times": [self.judge_time_times],
                "sail_penalty": [self.sail_penalty],
                "operational_reserve_percentage": self.operational_reserve_percentage,
                "total_gene_elect": self.total_gene_elect_list[-1],
                "total_gene_carrier": self.total_gene_carrier_list[-1],
                "total_loss_elect": self.total_loss_elect_list[-1],
                "sum_supply_elect": self.sum_supply_elect_list[-1],
                "minus_storage_penalty": self.minus_storage_penalty_list[-1],
            }
        )

        data = data.with_columns(
            [
                pl.col("Base_lat").cast(pl.Float64),
                pl.col("Base_lon").cast(pl.Float64),
                pl.col("Standby_lat").cast(pl.Float64),
                pl.col("Standby_lon").cast(pl.Float64),
                pl.col("hull_num").cast(pl.Int64),
                pl.col("hull_L_oa").cast(pl.Float64),
                pl.col("hull_B").cast(pl.Float64),
                pl.col("storage_method").cast(pl.Int64),
                pl.col("max_storage").cast(pl.Float64),
                pl.col("electric_propulsion_max_storage_wh").cast(pl.Float64),
                pl.col("trust_efficiency").cast(pl.Float64),
                pl.col("carrier_to_elect_efficiency").cast(pl.Float64),
                pl.col("elect_to_carrier_efficiency").cast(pl.Float64),
                pl.col("generator_turbine_radius").cast(pl.Float64),
                pl.col("generator_efficiency").cast(pl.Float64),
                pl.col("generator_drag_coefficient").cast(pl.Float64),
                pl.col("generator_pillar_chord").cast(pl.Float64),
                pl.col("generator_pillar_max_tickness").cast(pl.Float64),
                pl.col("generator_pillar_width").cast(pl.Float64),
                pl.col("generator_num").cast(pl.Int64),
                pl.col("generator_rated_output_w").cast(pl.Float64),
                pl.col("sail_num").cast(pl.Int64),
                pl.col("sail_width").cast(pl.Float64),
                pl.col("sail_area").cast(pl.Float64),
                pl.col("sail_space").cast(pl.Float64),
                pl.col("sail_steps").cast(pl.Int64),
                pl.col("sail_weight").cast(pl.Float64),
                pl.col("num_sails_per_row").cast(pl.Int64),
                pl.col("num_sails_rows").cast(pl.Int64),
                pl.col("nomal_ave_speed").cast(pl.Float64),
                pl.col("max_speed").cast(pl.Float64),
                pl.col("generating_speed_kt").cast(pl.Float64),
                pl.col("forecast_weight").cast(pl.Float64),
                pl.col("typhoon_effective_range").cast(pl.Float64),
                pl.col("govia_base_judge_energy_storage_per").cast(pl.Float64),
                pl.col("judge_time_times").cast(pl.Float64),
                pl.col("sail_penalty").cast(pl.Float64),
                pl.col("operational_reserve_percentage").cast(pl.Float64),
                pl.col("total_gene_elect").cast(pl.Float64),
                pl.col("total_gene_carrier").cast(pl.Float64),
                pl.col("total_loss_elect").cast(pl.Float64),
                pl.col("sum_supply_elect").cast(pl.Float64),
                pl.col("minus_storage_penalty").cast(pl.Int64),
            ]
        )

        return data

    ####################################  状態量  ######################################

    # 状態量と__init__で定義されたパラメータから求められる従属量の初期値を設定する関数

    def cal_dwt(self, storage_method, storage):
        """
        ############################ def cal_dwt ############################

        [ 説明 ]

        載貨重量トンを算出する関数です。

        ##############################################################################

        引数 :
            storage_method (int) : 貯蔵方法の種類。1=電気貯蔵,2=水素貯蔵
            storage (float) : 貯蔵容量[Wh]

        戻り値 :
            dwt (float) : 載貨重量トン

        #############################################################################
        """
        # 載貨重量トンを算出する。単位はt。
        # storageの容量は中の物質を完全燃焼させた場合のエネルギー量として考える

        if storage_method == 1:  # 電気貯蔵
            # 重量エネルギー密度1000Wh/kgの電池を使うこととする。
            dwt = storage / 1000 / 1000

        elif storage_method == 2:  # MCH貯蔵
            # 有機ハイドライドで水素を貯蔵することとする。
            dwt = storage / 5000 * 0.0898 / 47.4

        elif storage_method == 3:  # メタン貯蔵
            # 物性より計算　メタン1molの完全燃焼で802kJ=802/3600kWh
            # mol数の計算
            mol = storage / ((802 / 3600) * 1000)
            # メタンの分子量16.04g/molを用いてtに変換
            dwt = mol * 16.04 / 10**6

        elif storage_method == 4:  # メタノール貯蔵
            # 物性より計算　メタノール1molの完全燃焼で726.2kJ=726.2/3600kWh
            # mol数の計算
            mol = storage / ((726.2 / 3600) * 1000)
            # メタノールの分子量32.04g/molを用いてtに変換
            dwt = mol * 32.04 / 10**6

        elif storage_method == 5:  # e-ガソリン貯蔵
            # 代表の分子としてC8H18（オクタン）を用いる
            # オクタン1molの完全燃焼で5500kJ=5500/3600kWh
            # mol数の計算
            mol = storage / ((5500 / 3600) * 1000)
            # オクタンの分子量114.23g/molを用いてtに変換
            dwt = mol * 114.23 / 10**6

        else:
            print("cannot cal")

        return dwt

    def cal_maxspeedpower(
        self,
        max_speed,
        sail_num,
        sail_weight,
        storage1,
        storage1_method,
        storage2,
        storage2_method,
        body_num,
    ):
        """
        ############################ def cal_maxspeedpower ############################

        [ 説明 ]

        最大船速時に船体を進めるのに必要な出力を算出する関数です。

        ##############################################################################

        引数 :
            max_speed (float) : 最大船速(kt)
            sail_num (int) : 帆の本数
            sail_weight (float) : 帆の重量(ton)
            storage1 (float) : メインストレージの貯蔵容量1[Wh]
            storage1_method (int) : メインストレージの貯蔵方法の種類。1=電気貯蔵,2=水素貯蔵
            storage2 (float) : 電気推進用の貯蔵容量[Wh]
            storage2_method (int) : 電気推進用の貯蔵方法の種類。1=電気貯蔵,2=水素貯蔵
            body_num (int) : 船体の数

        戻り値 :
            power (float) : 船体を進めるのに必要な出力

        #############################################################################
        """

        main_storage_dwt = self.cal_dwt(storage1_method, storage1)
        electric_propulsion_storage_dwt = self.cal_dwt(storage2_method, storage2)
        sail_weight_sum = sail_weight * sail_num

        self.main_storage_weight = main_storage_dwt
        self.ep_storage_weight = electric_propulsion_storage_dwt
        self.sails_weight = sail_weight_sum

        sum_dwt_t = main_storage_dwt + electric_propulsion_storage_dwt + sail_weight_sum
        c_f = self.calculate_interference_coefficient(sail_num)

        if storage1_method == 1:  # 電気貯蔵
            # バルカー型
            k = 1.7
            power = (
                k
                * ((sum_dwt_t / body_num) ** (2 / 3))
                * (max_speed**3)
                * body_num
                * c_f
            )

        elif storage1_method >= 2:  # MCH・メタン・メタノール・e-ガソリン貯蔵
            # タンカー型
            k = 2.2
            power = (
                k
                * ((sum_dwt_t / body_num) ** (2 / 3))
                * (max_speed**3)
                * body_num
                * c_f
            )

        else:
            print("cannot cal")

        return power

    def calculate_interference_coefficient(self, sail_num):
        """
        ############################ def calculate_interference_coefficient ############################

        [ 説明 ]

        船体の干渉係数を算出する関数です。self.hull_Bがわかった状態で使うようにしてください。

        具体的には、calculate_max_sail_num関数、

        ##############################################################################

        戻り値 :
            interference_coefficient (float) : 船体の干渉係数

        #############################################################################
        """

        # 寸法計算
        self.calculate_hull_size(sail_num)

        # 船体の干渉係数を算出する。単位はなし。
        # 船体の干渉係数は船体の形状によって異なるため、ここでは簡易的に算出する。

        # 船体の干渉係数を算出

        if self.hull_num == 1:  # 船体が1つの場合

            interference_coefficient = 1.0

        elif self.hull_num == 2:  # 船体が2つの場合

            # 船体の載貨重量トンを計算
            hull_dwt = self.cal_dwt(self.storage_method, self.max_storage)
            # バッテリーの重量トンを計算
            battery_weight_ton = self.cal_dwt(
                1, self.electric_propulsion_max_storage_wh
            )
            total_ship_weight = (
                hull_dwt + battery_weight_ton + (sail_num * self.sail_weight)
            )
            total_ship_weight_per_body = total_ship_weight / self.hull_num

            # 「統計解析による船舶諸元に関する研究」よりDWTとL_oa,Bの値を算出する
            # 船体の寸法を計算
            if self.storage_method == 1:  # 電気貯蔵 = コンテナ型
                L_oa, B = self.calculate_LB_container(total_ship_weight_per_body)

            elif (
                (self.storage_method == 2)
                or (self.storage_method == 4)
                or (self.storage_method == 5)
            ):  # MCH・メタノール・e-ガソリン貯蔵 = タンカー型
                L_oa, B = self.calculate_LB_tanker(total_ship_weight_per_body)

            elif self.storage_method == 3:  # メタン貯蔵 = LNG船型
                L_oa, B = self.calculate_LB_lng(total_ship_weight_per_body)

            # 船側間距離を計算
            d = self.hull_B - 2 * B
            # 船体の干渉係数を算出
            interference_coefficient = 1.0 + 1.0 / (d / B)

        else:
            print("cannot cal")

        return interference_coefficient

    # コンテナ型の船体寸法計算
    def calculate_LB_container(self, total_ship_weight_per_body):
        """
        ############################ def calculate_LB_container ############################

        [ 説明 ]

        コンテナ型の船体の寸法を算出する関数です。

        船舶の主要諸元に関する解析(https://www.ysk.nilim.go.jp/kenkyuseika/pdf/ks0991.pdf)より計算を行います。

        ##############################################################################

        引数 :
            total_ship_weight_per_body (float) : 1つの船体の載貨重量トン[t]

        戻り値 :
            L_oa (float) : 船体の全長
            B (float) : 船体の幅

        #############################################################################
        """

        if total_ship_weight_per_body < 35000:
            L_oa = 6.0564 * (total_ship_weight_per_body**0.3398)
            B = 1.4257 * (total_ship_weight_per_body**0.2883)

        elif 35000 <= total_ship_weight_per_body < 45000:
            L_oa = 228.3
            B = 31.8

        elif 45000 <= total_ship_weight_per_body < 55000:
            L_oa = 268.8
            B = 33.7

        elif 55000 <= total_ship_weight_per_body < 65000:
            L_oa = 284.5
            B = 35.5

        elif 65000 <= total_ship_weight_per_body < 75000:
            L_oa = 291.0
            B = 39.2

        elif 75000 <= total_ship_weight_per_body < 85000:
            L_oa = 304.8
            B = 42.0

        elif 85000 <= total_ship_weight_per_body < 95000:
            L_oa = 310.9
            B = 44.1

        elif 95000 <= total_ship_weight_per_body < 105000:
            L_oa = 338.0
            B = 45.3

        elif 105000 <= total_ship_weight_per_body < 135000:
            L_oa = 343.1
            if total_ship_weight_per_body < 115000:
                B = 47.3
            elif 115000 <= total_ship_weight_per_body < 125000:
                B = 48.0
            else:
                B = 48.5

        elif 135000 <= total_ship_weight_per_body < 155000:
            L_oa = 367.5
            if total_ship_weight_per_body < 145000:
                B = 48.5
            else:
                B = 52.0

        elif 155000 <= total_ship_weight_per_body < 175000:
            L_oa = 378.3
            B = 52.0

        else:
            L_oa = 399.7
            B = 59.4

        return L_oa, B

    # タンカー型の船体寸法計算
    def calculate_LB_tanker(self, total_ship_weight_per_body):
        """
        ############################ def calculate_LB_tanker ############################

        [ 説明 ]

        タンカー型の船体の寸法を算出する関数です。

        船舶の主要諸元に関する解析(https://www.ysk.nilim.go.jp/kenkyuseika/pdf/ks0991.pdf)より計算を行います。

        ##############################################################################

        引数 :
            total_ship_weight_per_body (float) : 1つの船体の載貨重量トン[t]

        戻り値 :
            L_oa (float) : 船体の全長
            B (float) : 船体の幅

        #############################################################################
        """

        if total_ship_weight_per_body < 20000:
            L_oa = 5.4061 * (total_ship_weight_per_body**0.3500)
            B = 1.4070 * (total_ship_weight_per_body**0.2864)

        elif 20000 <= total_ship_weight_per_body < 280000:
            L_oa = 10.8063 * (total_ship_weight_per_body**0.2713)
            if total_ship_weight_per_body < 40000:
                B = 1.4070 * (total_ship_weight_per_body**0.2864)
            elif 40000 <= total_ship_weight_per_body < 80000:
                B = 32.9
            elif 80000 <= total_ship_weight_per_body < 120000:
                B = 43.5
            elif 120000 <= total_ship_weight_per_body < 200000:
                B = 48.9
            else:
                B = 60.2

        else:
            L_oa = 333.7
            B = 60.2

        return L_oa, B

    # LNG船型の船体寸法計算
    def calculate_LB_lng(self, total_ship_weight_per_body):
        """
        ############################ def calculate_LB_lng ############################

        [ 説明 ]

        LNG船型の船体の寸法を算出する関数です。

        船舶の主要諸元に関する解析(https://www.ysk.nilim.go.jp/kenkyuseika/pdf/ks0991.pdf)より計算を行います。

        ##############################################################################

        引数 :
            total_ship_weight_per_body (float) : 1つの船体の載貨重量トン[t]

        戻り値 :
            L_oa (float) : 船体の全長
            B (float) : 船体の幅

        #############################################################################
        """
        # 横軸がGT（総トン数）で記載されていたので、DWTをGTに変換する。一般にDWTがGTの0.6〜0.8倍程度とのことを利用する。
        GT = total_ship_weight_per_body / 0.7

        # GTに対する船体の全長、船体の幅を算出
        if GT < 150000:
            L_oa = 6.1272 * (GT**0.3343)
            B = 1.1239 * (GT**0.3204)
        else:
            L_oa = 345.2
            B = 54.6

        return L_oa, B

    # 船幅、船長計算
    def calculate_hull_size(self, sail_num):
        """
        ############################ def calculate_hull_size ############################

        [ 説明 ]

        船体の寸法を算出する関数です。

        ##############################################################################

        戻り値 :
            hull_L_oa (float) : 船体の全長
            hull_B (float) : 船体の幅

        #############################################################################
        """
        # 甲板面積を計算
        # 「統計解析による船舶諸元に関する研究」よりDWTとL_oa,Bの値を算出する

        # 船体の載貨重量トンを計算
        hull_dwt = self.cal_dwt(self.storage_method, self.max_storage)
        # バッテリーの重量トンを計算
        battery_weight_ton = self.cal_dwt(1, self.electric_propulsion_max_storage_wh)
        total_ship_weight = (
            hull_dwt + battery_weight_ton + (sail_num * self.sail_weight)
        )
        total_ship_weight_per_body = total_ship_weight / self.hull_num

        # DWTの記録
        self.ship_dwt = total_ship_weight

        # 船体の寸法を計算
        if self.storage_method == 1:  # 電気貯蔵 = コンテナ型
            L_oa, B = self.calculate_LB_container(total_ship_weight_per_body)

        elif (
            (self.storage_method == 2)
            or (self.storage_method == 4)
            or (self.storage_method == 5)
        ):  # MCH・メタノール・e-ガソリン貯蔵 = タンカー型
            L_oa, B = self.calculate_LB_tanker(total_ship_weight_per_body)

        elif self.storage_method == 3:  # メタン貯蔵 = LNG船型
            L_oa, B = self.calculate_LB_lng(total_ship_weight_per_body)

        # L_oa,Bの記録
        self.hull_L_oa = L_oa
        self.hull_B = B

        # 双胴船の場合の補正
        if self.hull_num == 2:
            # 船体が2つの場合、Bは3.5倍とする
            B = B * 3.5
            self.hull_B = B

    # 搭載性能の計算
    def calculate_max_sail_num(self):
        """
        ############################ def calculate_max_sail_num ############################

        [ 説明 ]

        台風発電船が搭載可能な帆の本数を算出する関数です。

        ##############################################################################

        戻り値 :
            max_sail_num (int) : 台風発電船が搭載可能な帆の本数

        #############################################################################
        """

        # ウインドチャレンジャーの帆を基準とする
        base_sail_area = 880  # 基準帆面積 [m^2]
        base_sail_width = 15  # 基準帆幅 [m]
        base_sail_height = 50  # 基準帆高さ [m]
        assumed_num_sails = 100  # 帆の仮想本数

        # 1. 帆の本数を仮定して、重量から船の寸法を計算する
        # 2. 計算した船の寸法から、甲板面積を算出
        # 3. 甲板面積と帆の幅から搭載可能な最大帆数を算出
        # 4. 仮の帆の本数と搭載可能な最大帆数を比較する
        # 5. 仮の帆の本数を更新し、帆の本数が等しくなるまで繰り返す

        while True:

            # 1. 帆の本数を仮定して、重量から船の寸法を計算する
            # 2. 計算した船の寸法から、甲板面積を算出
            self.calculate_hull_size(assumed_num_sails)
            L_oa = self.hull_L_oa  # 船体の全長 [m]
            B = self.hull_B  # 船体の幅 [m]

            deck_area = L_oa * B  # 簡易甲板面積 [m^2]

            # 3. 甲板面積と帆の幅から搭載可能な最大帆数を算出

            # 帆の寸法を基準帆から算出
            scale_factor = (self.sail_area / base_sail_area) ** 0.5
            sail_width = base_sail_width * scale_factor
            self.sail_width = sail_width
            self.sail_height = base_sail_height * scale_factor
            # 帆の搭載間隔を指定
            sail_space_per_sail = self.sail_width * self.sail_space

            if B < sail_space_per_sail:
                # 甲板幅が帆幅より狭い場合、船長に合わせて帆の本数を算出
                max_sails_by_deck_area = L_oa / sail_space_per_sail
                # 本数を四捨五入
                max_sails_by_deck_area = round(max_sails_by_deck_area)
            else:
                # 甲板面積から搭載できる最大帆数を算出
                max_sails_by_deck_area_L = L_oa / sail_space_per_sail
                max_sails_by_deck_area_B = B / sail_space_per_sail
                max_sails_by_deck_area = round(max_sails_by_deck_area_L) * round(
                    max_sails_by_deck_area_B
                )

            # 4. 仮の帆の本数と搭載可能な最大帆数を比較する
            # 5. 仮の帆の本数を更新し、帆の本数が等しくなるまで繰り返す

            if assumed_num_sails == max_sails_by_deck_area:
                break
            else:
                assumed_num_sails = max_sails_by_deck_area

        max_sail_num = max_sails_by_deck_area

        # デバック用出力
        # L_oa,B,sail_width,max_sails_by_deck_areaの出力
        # print(
        #     f"L_oa: {L_oa}",
        #     f"B: {B}",
        #     f"sail_width: {sail_width}",
        #     f"max_sails_by_deck_area: {max_sails_by_deck_area}",
        # )

        return max_sail_num

    # 発電時の推力性能（発電時の船速）の計算を抗力性能の計算とともに行う
    def cal_generating_ship_speed(self, sail_num):
        """
        ############################ def cal_generating_ship_speed ############################

        [ 説明 ]

        台風発電船の発電時の推力性能（発電時の船速）を計算する関数です。

        ##############################################################################

        戻り値 :
            generating_ship_speed_kt (float) : 発電時の船速(kt)

        #############################################################################
        """

        # 風速から発電船の最大速度を計算する。
        generating_ship_speed_mps = 0

        # 風で得られる推力の計算。
        wind_speed = self.generationg_wind_speed_mps  # 風速 [m/s]

        # 60~120度の風向きの場合、風で得られる最大の推力を計算する。
        max_wind_force = 0
        wind_directions = np.arange(60, 120, 1)  # [degrees]
        for wind_direction in wind_directions:
            # 横風
            lift_coefficient = 1.8
            drag_coefficient = 0.4
            lift = (
                0.5
                * self.air_density
                * wind_speed**2
                * self.sail_area
                * lift_coefficient
                * sail_num
            )
            drag = (
                0.5
                * self.air_density
                * wind_speed**2
                * self.sail_area
                * drag_coefficient
                * sail_num
            )
            # 推力は船の進行方向が正、横力は進行方向右向きが正
            force_angle = np.radians(wind_direction)
            # 風で得られる推力
            wind_force = lift * np.sin(force_angle) + drag * np.cos(force_angle)
            # 最大の推力を取得
            if wind_force > max_wind_force:
                max_wind_force = wind_force
                self.max_wind_force_direction = wind_direction
                self.sails_lift = lift
                self.sails_drag = drag

        # 帆の密度に対するペナルティを計算
        self.calculate_sail_penalty(sail_num)

        max_wind_force = max_wind_force * self.sail_penalty

        # 動いている水中タービンの抵抗係数の計算(船速の2乗をかけると抵抗が計算できる係数)
        # タービン抵抗係数(こちらは船速が決まっている時に使われる係数)
        turbine_drag_coefficient_1 = self.generator_drag_coefficient
        # 定格出力からタービン回転面積の逆算
        turbine_area = self.generator_turbine_radius**2 * np.pi
        # タービン抵抗係数の計算
        turbine_drag_coefficient = (
            self.generator_num
            * 0.5
            * self.sea_density
            * turbine_area
            * turbine_drag_coefficient_1
        )

        # 最大船速時の船体の抵抗（仕事率）の計算
        self.max_speed_power = self.cal_maxspeedpower(
            self.max_speed,
            sail_num,
            self.sail_weight,
            self.max_storage,
            self.storage_method,
            self.electric_propulsion_max_storage_wh,
            1,
            self.hull_num,
        )
        # 船体の抵抗係数(船速の2乗をかけると抵抗が計算できる係数)
        hull_drag_coefficient = self.max_speed_power / self.max_speed**3
        # hull_drag_coefficientがkt基準なので、m/sに変換
        hull_drag_coefficient = hull_drag_coefficient * 1.94384**3

        # 船体の抵抗とタービンの抵抗の和と風で得られる最大推力が釣り合う船速を計算
        generating_ship_speed_mps = np.sqrt(
            (max_wind_force) / (turbine_drag_coefficient + hull_drag_coefficient)
        )
        # ktに変換
        generating_ship_speed_kt = generating_ship_speed_mps * 1.94384

        self.tpgship_generating_lift = max_wind_force
        self.tpgship_generating_drag = (
            turbine_drag_coefficient + hull_drag_coefficient
        ) * generating_ship_speed_mps**2
        self.tpgship_turbine_drag = (
            turbine_drag_coefficient * generating_ship_speed_mps**2
        )
        self.tpgship_hull_drag = hull_drag_coefficient * generating_ship_speed_mps**2

        # 最終結果の反映
        return generating_ship_speed_kt

    # 帆の密度（間隔）からペナルティを計算
    def calculate_sail_penalty(self, sail_num):
        """
        ############################ def calculate_sail_penalty ############################

        [ 説明 ]

        任意の帆の数と船の横幅、全長を用いて、帆の等間隔配置を計算し、

        台風発電船の帆の密度（間隔）からペナルティを計算する関数です。

        ##############################################################################

        戻り値:
        spacing_penalty (float): 帆の間隔によるペナルティ
        message (str): 完全に等間隔に並べられない場合の理由と改善策（本プログラム内では出力しないデバック用出力）
        """
        # 船体の寸法を計算
        self.calculate_hull_size(sail_num)

        sail_width = self.sail_width

        max_sail_num = self.calculate_max_sail_num()

        B = self.hull_B
        L_oa = self.hull_L_oa

        if self.sail_num == max_sail_num:
            sail_space = self.sail_space
        else:
            # B*L_oaの長方形内に幅sail_widthの帆をsail_num本、等間隔で並べた時の帆の間隔を計算
            sail_space = np.sqrt(B * L_oa / sail_num) / sail_width

        spacing_penalty = 0
        if B > sail_width * sail_space:
            optimal_num_sails_per_row = round(B / (sail_width * sail_space))
        else:
            optimal_num_sails_per_row = 1

        optimal_num_rows = round(L_oa / (sail_width * sail_space))

        # 帆の密度に対するペナルティを計算
        if sail_space >= 2:
            spacing_penalty = 1
        elif sail_space <= 1:
            spacing_penalty = 0.6
        else:
            spacing_penalty = 0.6 - (0.4 * (1 - sail_space))

        self.num_sails_per_row = optimal_num_sails_per_row  # 1行に配置する帆の本数
        self.num_sails_rows = optimal_num_rows  # 帆の行数
        self.sail_penalty = spacing_penalty  # 帆の間隔によるペナルティ

    # 発電性能（水中タービンの定格出力）を計算
    def calculate_generater_rated_output(self):
        """
        ############################ def calculate_generater_rated_output ############################

        [ 説明 ]

        台風発電船の水中タービンの定格出力を計算する関数です。

        ##############################################################################

        戻り値 :
            rated_output_w (float) : 水中タービンの定格出力(W)

        #############################################################################
        """

        # 発電時の船速をktからm/sに変換
        self.generating_speed_mps = self.generating_speed_kt * 0.514444

        # タービン回転面積
        turbine_area = self.generator_turbine_radius**2 * np.pi
        # 定格出力の計算
        rated_output_w = (
            self.generator_num
            * 0.5
            * self.sea_density
            * self.generating_speed_mps**3
            * turbine_area
            * self.generator_efficiency
        )

        return rated_output_w

    # 帆の本数、発電時船速、定格出力を計算して、反映する関数
    def calculate_sail_num_and_generating_ship_speed_and_generater_rated_output(self):
        """
        ############################ def calculate_sail_num_and_generating_ship_speed_and_generater_rated_output ############################

        [ 説明 ]

        台風発電船の帆の本数、発電時船速、定格出力を計算して、反映する関数です。

        副次的に、船体の諸元が求められるので、必ず使用する必要があります。

        ##############################################################################

        """
        # 帆の本数のチェック
        max_sail_num = self.calculate_max_sail_num()
        if self.sail_num > max_sail_num:
            self.sail_num = max_sail_num

        # 発電時の船速の計算
        self.generating_speed_kt = self.cal_generating_ship_speed(self.sail_num)
        if self.generating_speed_kt > self.limit_ship_speed_kt:
            self.generating_speed_kt = self.limit_ship_speed_kt

        # 定格出力を計算
        self.generator_rated_output_w = self.calculate_generater_rated_output()

    # 状態量の初期値入力と従属量の入力
    def set_initial_states(self):
        """
        ############################ def set_initial_states ############################

        [ 説明 ]

        台風発電船の各種状態量に初期値を与える関数です。

        max_storage , base_lat , base_lon , standby_lat , standby_lonの数値の定義が少なくとも先に必要です。

        ##############################################################################

        """
        # 運航環境の状態量
        self.air_density = 1.225  # kg/m^3
        self.sea_density = 1025.0  # kg/m^3
        self.kinematic_viscosity = 1.139 * 10**-6  # m^2/s

        # 台風発電船の受ける風パラメータ
        self.wind_u = 0
        self.wind_v = 0
        self.wind_state = str("no")

        # 船内電気関係の状態量
        self.storage = self.operational_reserve
        self.storage_percentage = (self.storage / self.max_storage) * 100
        self.supply_elect = 0
        self.sum_supply_elect = 0
        self.gene_elect = 0
        self.gene_carrier = 0
        self.loss_elect = 0
        self.ship_state = 0
        self.total_gene_elect = 0
        self.total_gene_carrier = 0
        self.total_loss_elect = 0
        self.total_gene_time = 0
        self.total_loss_time = 0

        # 発電船の行動に関する状態量(現状のクラス定義では外部入力不可(更新が内部関数のため))
        self.hull_drag_work = 0
        self.wind_trust_work = 0
        self.generator_drag_work = 0
        self.speed_kt = 0
        self.ship_lat_before = self.ship_lat
        self.ship_lon_before = self.ship_lon
        self.target_name = "base station"
        self.target_lat = self.base_lat
        self.target_lon = self.base_lon
        self.target_distance = 0
        self.target_TY = 0
        self.go_base = 0
        self.TY_and_base_action = 0
        self.next_TY_lat = 0
        self.next_TY_lon = 0
        self.next_ship_TY_dis = np.nan
        self.brance_condition = "start forecast"
        self.GS_gene_judge = 0
        self.electric_propulsion_storage_wh = self.electric_propulsion_max_storage_wh
        self.electric_propulsion_storage_state = str("no action")  # 電気推進機の状態
        self.wind_speed = 0
        self.wind_direction = 0
        self.minus_storage_penalty = 0

        # 発電船自律判断システム設定
        self.judge_energy_storage_per = 100
        self.judge_direction = 10
        self.standby_via_base = 0

        # 発電船発電時の状態量
        self.limit_ship_speed_kt = (
            # 52  # 発電時の船速の上限値(kt) 双胴フェリー船の最高速度を参照
            32  # 発電時の船速の上限値(kt) 一般的なコンテナ船の上限速度を参照
            # 22  # 発電時の船速の上限値(kt) 一般的な貨物船の上限速度を参照
        )
        self.generationg_wind_speed_mps = 25
        self.generationg_wind_dirrection_deg = 80.0
        self.calculate_sail_num_and_generating_ship_speed_and_generater_rated_output()  # 帆の本数と発電時の推力性能（発電時の船速）の計算を行う

    def set_outputs(self):
        """
        ############################ def set_outputs ############################

        [ 説明 ]

        台風発電船の各種出力を記録するリストを作成する関数です。

        ##############################################################################

        """

        # 発電船の行動詳細
        self.branch_condition_list = []

        # 台風の番号
        self.target_typhoon_num_list = []

        # 目標地点
        self.target_name_list = []
        self.target_lat_list = []
        self.target_lon_list = []
        self.target_dis_list = []

        # 台風座標
        self.TY_lat_list = []
        self.TY_lon_list = []

        # 発電船台風間距離
        self.GS_TY_dis_list = []

        # 発電船の座標
        self.GS_lat_list = []
        self.GS_lon_list = []

        # 発電船の状態
        self.GS_state_list = []  # 発電船の行動状態(描画用数値)
        self.GS_speed_list = []
        self.wind_speed_list = []
        self.wind_direction_list = []

        ############################# 発電指数 ###############################
        self.GS_elect_storage_percentage_list = []  # 船内蓄電割合
        self.GS_storage_state_list = []
        self.gene_elect_time_list = []  # 発電時間
        self.total_gene_elect_list = []  # 総発電量
        self.total_gene_carrier_list = []  # 総生産エネルギーキャリア
        self.loss_elect_time_list = []  # 電力消費時間（航行時間）
        self.total_loss_elect_list = []  # 総消費電力
        self.balance_gene_elect_list = []  # 発電収支（船内蓄電量）
        self.per_timestep_gene_elect_list = []  # 時間幅あたりの発電量
        self.per_timestep_gene_carrier_list = []  # 時間幅あたりの生産エネルギーキャリア
        self.per_timestep_loss_elect_list = []  # 時間幅あたりの消費電力
        # self.year_round_balance_gene_elect_list = []  # 通年発電収支
        self.sum_supply_elect_list = []  # 総供給電力量
        self.minus_storage_penalty_list = []  # 蓄電量が0以下の場合のペナルティ

        # 発電状態チェック用
        self.GS_gene_judge_list = []

        # 船体抵抗チェック用
        self.hull_drag_work_list = []

        # 風状態チェック用
        self.wind_trust_work_list = []
        self.wind_u_list = []
        self.wind_v_list = []
        self.wind_state_list = []

        # 電気推進用蓄電池チェック用
        self.electric_propulsion_storage_state_list = []
        self.trust_power_storage_list = []

        # 発電機抵抗チェック用
        self.generator_drag_work_list = []

    def outputs_append(self):
        """
        ############################ def outputs_append ############################

        [ 説明 ]

        set_outputs関数で作成したリストに出力を記録する関数です。

        ##############################################################################

        """
        self.branch_condition_list.append(self.brance_condition)

        self.target_name_list.append(self.target_name)
        self.target_lat_list.append(float(self.target_lat))
        self.target_lon_list.append(float(self.target_lon))
        self.target_dis_list.append(float(self.target_distance))

        self.target_typhoon_num_list.append(self.target_TY)
        self.TY_lat_list.append(float(self.next_TY_lat))
        self.TY_lon_list.append(float(self.next_TY_lon))
        self.GS_TY_dis_list.append(float(self.next_ship_TY_dis))

        self.GS_lat_list.append(float(self.ship_lat))
        self.GS_lon_list.append(float(self.ship_lon))
        self.GS_state_list.append(self.ship_state)
        self.GS_speed_list.append(float(self.speed_kt))
        self.wind_speed_list.append(float(self.wind_speed))
        self.wind_direction_list.append(float(self.wind_direction))

        self.per_timestep_gene_elect_list.append(
            float(self.gene_elect)
        )  # 時間幅あたりの発電量[Wh]
        self.per_timestep_gene_carrier_list.append(
            float(self.gene_carrier)
        )  # 時間幅あたりの生産エネルギーキャリア[Wh]
        self.gene_elect_time_list.append(float(self.total_gene_time))  # 発電時間[hour]
        self.total_gene_elect_list.append(float(self.total_gene_elect))  # 総発電量[Wh]
        self.total_gene_carrier_list.append(
            float(self.total_gene_carrier)
        )  # 総生産エネルギーキャリア[Wh]
        self.per_timestep_loss_elect_list.append(
            float(self.loss_elect)
        )  # 時間幅あたりの消費電力[Wh]
        self.loss_elect_time_list.append(
            float(self.total_loss_time)
        )  # 電力消費時間（航行時間）[hour]
        self.total_loss_elect_list.append(
            float(self.total_loss_elect)
        )  # 総消費電力[Wh]

        storage_percentage = (self.storage / self.max_storage) * 100
        # 蓄電量が20％以下
        if storage_percentage < 0:
            self.minus_storage_penalty = self.minus_storage_penalty + 100
            storage_state = 0

        elif storage_percentage <= 20:

            storage_state = 1
        # 蓄電量が100％以上
        elif storage_percentage >= 100:

            storage_state = 4
        # 蓄電量が80％以上
        elif storage_percentage >= 80:

            storage_state = 3
        # 蓄電量が20％より多く、80％より少ない
        else:

            storage_state = 2

        self.GS_elect_storage_percentage_list.append(
            float(storage_percentage)
        )  # 船内蓄電割合[%]
        self.GS_storage_state_list.append(int(storage_state))

        self.balance_gene_elect_list.append(
            float(self.storage)
        )  # 発電収支（船内蓄電量）[Wh]

        self.sum_supply_elect_list.append(
            float(self.sum_supply_elect)
        )  # 総供給電力量[Wh]

        self.minus_storage_penalty_list.append(int(self.minus_storage_penalty))

        # 発電状態チェック用
        self.GS_gene_judge_list.append(self.GS_gene_judge)

        # 船体抵抗チェック用
        self.hull_drag_work_list.append(float(self.hull_drag_work))

        # 風状態チェック用
        self.wind_trust_work_list.append(float(self.wind_trust_work))
        self.wind_u_list.append(float(self.wind_u))
        self.wind_v_list.append(float(self.wind_v))
        self.wind_state_list.append(self.wind_state)

        # 発電機抵抗チェック用
        self.generator_drag_work_list.append(float(self.generator_drag_work))

        # 電気推進用蓄電池チェック用
        self.electric_propulsion_storage_state_list.append(
            self.electric_propulsion_storage_state
        )
        self.trust_power_storage_list.append(float(self.electric_propulsion_storage_wh))

    def get_outputs(self, unix_list, date_list):
        """
        ############################ def get_outputs ############################

        [ 説明 ]

        台風発電船の各種出力を記録したリストをデータフレームに変換する関数です。

        ##############################################################################

        引数 :
            unix_list : unix時間のリスト
            date_list : 日時のリスト

        戻り値 :
            data : 台風発電船の各種出力を記録したデータフレーム

        #############################################################################
        """

        data = pl.DataFrame(
            {
                "unixtime": unix_list,
                "datetime": date_list,
                "TARGET LOCATION": self.target_name_list,
                "TARGET LAT": self.target_lat_list,
                "TARGET LON": self.target_lon_list,
                "TARGET DISTANCE[km]": self.target_dis_list,
                "TARGET TYPHOON": self.target_typhoon_num_list,
                "TARGET TY LAT": self.TY_lat_list,
                "TARGET TY LON": self.TY_lon_list,
                "TPGSHIP LAT": self.GS_lat_list,
                "TPGSHIP LON": self.GS_lon_list,
                "TPG_TY DISTANCE[km]": self.GS_TY_dis_list,
                "BRANCH CONDITION": self.branch_condition_list,
                "TPGSHIP STATUS": self.GS_state_list,
                "SHIP SPEED[kt]": self.GS_speed_list,
                "TIMESTEP POWER GENERATION[Wh]": self.per_timestep_gene_elect_list,
                "TIMESTEP CARRIER GENERATION[Wh]": self.per_timestep_gene_carrier_list,
                "TOTAL GENE TIME[h]": self.gene_elect_time_list,
                "TOTAL POWER GENERATION[Wh]": self.total_gene_elect_list,
                "TOTAL GENE CARRIER[Wh]": self.total_gene_carrier_list,
                "TIMESTEP POWER CONSUMPTION[Wh]": self.per_timestep_loss_elect_list,
                "TOTAL CONS TIME[h]": self.loss_elect_time_list,
                "TOTAL POWER CONSUMPTION[Wh]": self.total_loss_elect_list,
                "WIND TRUST WORK[Wh]": self.wind_trust_work_list,
                "WIND U[m/s]": self.wind_u_list,
                "WIND V[m/s]": self.wind_v_list,
                "WIND STATE": self.wind_state_list,
                "ONBOARD POWER STORAGE PER[%]": self.GS_elect_storage_percentage_list,
                "ONBOARD POWER STORAGE STATUS": self.GS_storage_state_list,
                "ONBOARD ENERGY STORAGE[Wh]": self.balance_gene_elect_list,
                "ONBOARD ELECTRIC PROPULSION STORAGE[Wh] ": self.trust_power_storage_list,
                "ONBOARD ELECTRIC PROPULSION STORAGE STATUS": self.electric_propulsion_storage_state_list,
                "TOTAL SUPPLY ELECTRICITY[Wh]": self.sum_supply_elect_list,
                "MINUS STORAGE PENALTY": self.minus_storage_penalty_list,
            }
        )

        return data

    ####################################  メソッド  ######################################

    # 船の機能としては発電量計算、消費電力量計算、予報データから台風の目標地点の決定、timestep後の時刻における追従対象台風の座標取得のみ？
    # 状態量を更新するような関数はメソッドではない？

    # とりあえず状態量の計算をしている関数がわかるように　#状態量計算　をつけておく
    def find_nearest_wind_point(self, wind_data):
        """
        ############################ def find_nearest_wind_point ############################

        [ 説明 ]

        その時刻における台風発電船の位置の風の状態を取得する関数です。

        風速情報と台風発電船の移動軌跡に最も近い地点を見つける関数です。

        ##############################################################################

        引数 :
            wind_data : ERA5の風速データ


        戻り値 :
            u (float) : x方向（経度方向東正）の風速[m/s]
            v (float) : y方向（緯度方向北正）の風速[m/s]

        #############################################################################
        """

        # 最も近い風速情報を取得
        nearest_point = (
            wind_data.with_columns(
                (pl.lit(self.ship_lat) - pl.col("LAT")).abs().alias("lat_diff"),
                (pl.lit(self.ship_lon) - pl.col("LON")).abs().alias("lon_diff"),
            )
            .with_columns((pl.col("lat_diff") + pl.col("lon_diff")).alias("total_diff"))
            .sort("total_diff")
            .head(1)
        )

        # 最も近い風速情報を取得
        u = nearest_point["U10_E+_W-[m/s]"][0]
        v = nearest_point["V10_N+_S-[m/s]"][0]

        self.wind_u = u
        self.wind_v = v

        # print("wind座標",nearest_point["LAT"][0],nearest_point["LON"][0])
        # print("現在地から1番近いポイント＝",nearest_point)

        return u, v

    #######################  硬翼帆の場合の風力推進機の出力計算  #######################

    # 硬翼帆の揚力を推進力に利用する場合の計算用関数(26~167,193~333度の横風領域)(167~193度の向かい風)
    def calculate_lift(self, wind_speed, lift_coefficient):
        """
        ############################ def calculate_lift ############################

        [ 説明 ]

        硬翼帆が風から得る揚力を計算する関数です。

        ##############################################################################

        引数 :
            wind_speed (float) : 風速[m/s]
            lift_coefficient (float) : 揚力係数

        戻り値 :
            lift (float) : 揚力[N]


        #############################################################################
        """

        lift = (
            0.5 * self.air_density * wind_speed**2 * self.sail_area * lift_coefficient
        )

        return lift

    def calculate_drag(self, wind_speed, drag_coefficient):
        """
        ############################ def calculate_drag ############################

        [ 説明 ]

        硬翼帆が風から得る抗力を計算する関数です。

        ##############################################################################

        引数 :
            wind_speed (float) : 風速[m/s]
            drag_coefficient (float) : 抗力係数

        戻り値 :
            drag (float) : 抗力[N]


        #############################################################################
        """

        drag = (
            0.5 * self.air_density * wind_speed**2 * self.sail_area * drag_coefficient
        )

        return drag

    def calculate_force(self, wind_direction, lift, drag):
        """
        ############################ def calculate_force ############################

        [ 説明 ]

        硬翼帆が風から得る推進力と横力を計算する関数です。

        推進力は船の進行方向が正、横力は進行方向右向きが正とします。

        ##############################################################################

        引数 :
            wind_direction (float) : 台風発電船の進行方向と風向の角度差[deg]
            lift (float) : 揚力[N]
            drag (float) : 抗力[N]

        戻り値 :
            force_trust (float) : 台風発電船が得る推進力[N]
            force_side (float) : 台風発電船が得る横力[N]


        #############################################################################
        """
        # 推力は船の進行方向が正、横力は進行方向右向きが正
        force_angle = np.radians(wind_direction)
        # force_side =  -lift * np.cos(force_angle) + drag * np.sin(force_angle)

        if wind_direction <= 180:
            force_trust = lift * np.sin(force_angle) + drag * np.cos(force_angle)
        else:
            # 硬翼帆は一回転させず進行方向に対して0から180度、0から-180度に動かすため推力は鏡写しで計算する必要がある
            force_angle = np.radians(360 - wind_direction)
            force_trust = lift * np.sin(force_angle) + drag * np.cos(force_angle)

        return force_trust  # , force_side

    # 硬翼帆の抗力を推進力に利用する場合の計算用関数(0~26,333~360度の追い風領域)
    def calculate_plate_drag(self, wind_direction, drag):
        """
        ############################ def calculate_plate_drag ############################

        [ 説明 ]

        硬翼帆が追い風から得る抗力による推進力と横力を計算する関数です。

        推進力は船の進行方向が正、横力は進行方向右向きが正とします。

        ##############################################################################

        引数 :
            wind_direction (float) : 台風発電船の進行方向と風向の角度差[deg]
            drag (float) : 抗力[N]

        戻り値 :
            force_trust (float) : 台風発電船が得る推進力[N]
            force_side (float) : 台風発電船が得る横力[N]


        #############################################################################
        """
        force = drag
        force_trust = force * np.cos(np.radians(wind_direction)) * self.sail_penalty
        # force_side = force * np.sin(np.radians(wind_direction))
        return force_trust  # , force_side

    # 風向360度対応仕事量計算用関数
    def calculate_work(self, wind_direction, time_interval):
        """
        ############################ def calculate_plate_drag ############################

        [ 説明 ]

        硬翼帆によって台風発電船が得られる仕事量の計算用関数です。

        ##############################################################################

        引数 :
            wind_direction (float) : 台風発電船の進行方向と風向の角度差[deg]
            time_interval (int) : エネルギー計算における時間の進み幅[hours]

        戻り値 :
            wind_work (float) : time_intervalあたりに風から得られる仕事量[Wh]


        #############################################################################
        """
        # 船速の単位をktからm/sに変換
        ship_speed_mps = self.speed_kt * 0.514444  # 船の速度 [m/s]
        # 風速の計算
        wind_speed = math.sqrt(self.wind_u**2 + self.wind_v**2)

        if self.GS_gene_judge == 1:  # 台風下にいる場合
            ship_speed_mps = self.generating_speed_mps
            wind_speed = self.generationg_wind_speed_mps
            wind_direction = self.generationg_wind_dirrection_deg

        # 記録
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

        lift = 0
        drag = 0

        # 風向きによって計算方法を分岐
        # 横風
        if 26 <= wind_direction <= 167 or 193 <= wind_direction <= 333:
            self.wind_state = str("Cross wind")
            # 迎角20度時の係数を使用
            lift_coefficient = 1.8
            drag_coefficient = 0.4
            lift = self.calculate_lift(wind_speed, lift_coefficient)
            drag = self.calculate_drag(wind_speed, drag_coefficient)
            force_trust = self.calculate_force(wind_direction, lift, drag)

        # 追い風
        elif 0 <= wind_direction <= 26 or 333 <= wind_direction <= 360:
            self.wind_state = str("Tail wind")
            # 迎角90度時の係数を使用
            drag_coefficient = 1.3
            drag = self.calculate_drag(wind_speed, drag_coefficient)
            lift = 0
            force_trust = self.calculate_plate_drag(wind_direction, drag)

        # 向かい風
        else:
            self.wind_state = str("Head wind")
            # 迎角0度時の係数を使用
            lift_coefficient = 0.5  # 揚力係数
            drag_coefficient = 0.1  # 抗力係数
            lift = self.calculate_lift(wind_speed, lift_coefficient) / self.sail_steps
            drag = self.calculate_drag(wind_speed, drag_coefficient) / self.sail_steps
            force_trust = self.calculate_force(wind_direction, lift, drag)

        # 今回は横力(force_side)を無視して推進力のみを計算

        # 仕事量の計算[Wh]
        wind_work = (
            (force_trust)
            * self.sail_penalty
            * ship_speed_mps
            * time_interval
            * self.sail_num
        )

        # self.ship_speed_sub_list.append(ship_speed_mps)
        # self.wind_speed_sub_list.append(wind_speed)
        # self.wind_direction_sub_list.append(wind_direction)
        # self.wind_state_sub_list.append(self.wind_state)
        # self.wind_work_list.append(wind_work)
        # self.wind_trust_list.append(force_trust)
        # self.sail_lift_list.append(lift*self.sail_num)
        # self.sail_drag_list.append(drag*self.sail_num)

        return wind_work

    def calculate_initial_bearing(self):
        """
        ############################ def calculate_initial_bearing ############################

        [ 説明 ]

        台風発電船の航路である大圏航路上の移動を再現するための方位角の取得関数です。

        取得した方位角を用いて始点から終点間のtime_interval毎の座標を取得します。

        ##############################################################################

        引数 :


        戻り値 :
            initial_bearing (float) : 台風発電船の進行方向の初期方位角[deg]


        #############################################################################
        """

        # 始点と終点の緯度と経度
        start_coord = (self.ship_lat_before, self.ship_lon_before)
        end_coord = (self.ship_lat, self.ship_lon)

        # Geodesicクラスのインスタンスを作成
        geod = Geodesic.WGS84

        # 始点と終点の緯度と経度
        lat1, lon1 = start_coord[0], start_coord[1]
        lat2, lon2 = end_coord[0], end_coord[1]

        # geod.Inverse()を使用して始点から終点までの地理的な情報を取得
        # 結果として辞書が返され、辞書のキー 'azi1' は初期方位角を含む
        result = geod.Inverse(lat1, lon1, lat2, lon2)

        # 初期方位角を取得（北を0度とし、時計回りに測定）
        initial_bearing = result["azi1"]

        return initial_bearing

    def calculate_trajectory_energy(self, wind_data, timestep):
        """
        ############################ def calculate_trajectory_energy ############################

        [ 説明 ]

        台風発電船の移動軌跡と風速情報から風力推進機（硬翼帆）で得られるエネルギーを計算する関数です。

        ##############################################################################

        引数 :
            wind_data : ERA5の風速データ
            timestep (int) : シミュレーションにおける時間の進み幅[hours]


        戻り値 :
            total_wind_work (float) : time_stepで風から風力推進機が得られる仕事量[Wh]

        #############################################################################
        """

        # 初期方位角を計算
        initial_bearing = self.calculate_initial_bearing()

        # 始点と終点の緯度と経度
        start_coord = (self.ship_lat_before, self.ship_lon_before)
        end_coord = (self.ship_lat, self.ship_lon)

        # 移動しない場合の計算を省く場合の分岐
        # if start_coord == end_coord:
        #    return 0

        # else:

        # 始点と終点を Point オブジェクトに変換
        start_point = Point(start_coord)
        end_point = Point(end_coord)

        # 大圏距離を計算
        geo_dist = geodesic(start_point, end_point)
        total_distance = geo_dist.kilometers

        # 時間間隔の設定
        time_interval = timestep / (2 * timestep)

        # 初期化
        # current_position = start_point
        current_bearing = initial_bearing
        total_wind_work = 0

        # 時間間隔ごとの移動
        roop_num = 2 * timestep

        for i in range(roop_num):

            # 現在の時刻
            time = i * time_interval

            # 指定された時間での移動距離を計算
            distance_travelled = (total_distance / timestep) * time

            # 現在の位置を計算
            # current_position = geodesic(kilometers=distance_travelled).destination(point=start_point, bearing=current_bearing)

            # 現在の方位角を更新
            current_bearing = self.calculate_initial_bearing()

            # 現在の位置の緯度経度
            # current_lat, current_lon = (current_position.latitude,current_position.longitude)

            # 風速データを取得
            # ここでは風速データを取得するための関数 find_nearest_wind_point を使用します。
            # この関数は data から緯度経度に最も近い風速データを取得するものです。
            u, v = self.find_nearest_wind_point(wind_data)

            # 風向角度を計算（北を 0 度として時計回りに増加）
            wind_angle = math.degrees(math.atan2(v, u))
            # 北を 0 度として時計回りに測定するように調整
            wind_angle = (360 - wind_angle + 90) % 360

            # 船の方位角と風向の角度差を計算
            if current_bearing < 0:
                current_bearing += 360

            # 船の方位から見て風がどの方向に向かって吹いているのかを計算している。（必要に応じて吹いて来ている方向に直す）
            wind_direction = (wind_angle - current_bearing + 360) % 360

            # エネルギーを計算
            wind_work = self.calculate_work(wind_direction, time_interval)

            # エネルギーを累積
            total_wind_work += wind_work

            # print("ship座標",current_lat,current_lon)
            # print("風速",u,v)
            # print("進路",current_bearing)
            # print("風向",wind_angle)
            # print("角度差",angle_difference)
            # print("エネルギー",energy)

        return total_wind_work

    #######################################################################################

    ###########################  発電機に関する設定  #####################################
    def calculate_stopped_generator_drag_work(self):
        """
        ############################ def calculate_generater_drag_work ############################

        [ 説明 ]

        台風発電船の発電機(実際は発電プロペラ後方の流線型物体）による非台風下における抵抗の仕事量を計算する関数です。

        停止状態のプロペラの抵抗を計算するのが面倒だったため、発電プロペラ後方の流線型物体の抵抗を計算する関数を作成しました。

        そのため、流線型物体の船体から飛び出ている長さwはプロペラの直径より大きいものとして考えています。

        ##############################################################################

        引数 (今回はselfで定義しているので注意):
            ship_speed_kt (float) : 船速（kt）
            c (float) : 発電機（発電プロペラ後方の流線型物体）のコード長（m）
            t (float) : 発電機（発電プロペラ後方の流線型物体）の最大厚さ（m）
            w (float) : 発電機（発電プロペラ後方の流線型物体）の幅方向の長さ（m）

        戻り値 :
            Da (float) : 発電機（発電プロペラ後方の流線型物体）の抵抗力による仕事量（W）

        #############################################################################

        """
        # 海水の密度と動粘度
        rho = self.sea_density  # kg/m^3
        nu = self.kinematic_viscosity  # m^2/s

        # 船速（物体周りの流速）のm/sへの変換
        ship_speed_mps = self.speed_kt * 0.514444

        if ship_speed_mps != 0:
            # 発電プロペラ後方の流線型物体のレイノルズ数
            re = ship_speed_mps * self.generator_pillar_chord / nu

            # 平板の摩擦抵抗係数をプラントル・シュリヒティングの公式から計算
            cf = 0.455 / (math.log10(re) ** 2.58)

            # 発電プロペラ後方にある2次元柱体の流線型物体（以下、支柱）の粘性抵抗係数をヘルナーの実験式から計算
            cd = (
                2
                * (
                    self.generator_pillar_chord / self.generator_pillar_max_tickness
                    + 2
                    + 60
                    + (self.generator_pillar_max_tickness / self.generator_pillar_chord)
                    ** 3
                )
                * cf
            )

            # 単位幅あたりの抵抗力
            d = 0.5 * cd * rho * ship_speed_mps**2 * self.generator_pillar_max_tickness

            # 発電機の支柱の抵抗力
            da = d * self.generator_pillar_width

            # 発電機1つの支柱の抵抗力による仕事量（W）
            da = da * ship_speed_mps
        else:
            da = 0

        if self.GS_gene_judge == 1:
            # 発電機のタービン回転面積
            s_pg = self.generator_turbine_radius**2 * math.pi
            # 発電機のタービン1機あたりの抵抗
            da = (
                0.5
                * rho
                * self.generating_speed_mps**2
                * s_pg
                * self.generator_drag_coefficient
            )
            # 発電機のタービン1機あたりの抵抗力による仕事量（W）
            da = da * self.generating_speed_mps

        return da

    #######################################################################################

    def calculate_power_consumption(self, wind_data, time_step):
        """
        ############################ def calculate_power_consumption ############################

        [ 説明 ]

        time_stepごとの台風発電船の消費電力量(Wh)を計算する関数です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        戻り値 :
            energy_loss (float) : time_stepで消費される電力量[Wh]

        #############################################################################
        """

        self.wind_trust_work = self.calculate_trajectory_energy(
            wind_data, time_step
        )  # [Wh]
        self.generator_drag_work = (
            self.generator_num
            * self.calculate_stopped_generator_drag_work()
            * time_step
        )  # [Wh]

        # 船体抵抗による仕事量
        if self.GS_gene_judge == 1:
            self.speed_kt = self.generating_speed_kt
            self.hull_drag_work = (
                (self.max_speed_power)
                * ((self.generating_speed_kt / self.max_speed) ** 3)
                * time_step
            )
        else:
            self.hull_drag_work = (
                (self.max_speed_power)
                * ((self.speed_kt / self.max_speed) ** 3)
                * time_step
            )

        # 台風追従に必要な出力
        typhoon_tracking_power = (
            self.hull_drag_work + self.generator_drag_work - self.wind_trust_work
        )

        if typhoon_tracking_power < 0:
            typhoon_tracking_power = 0

        # 電気から動力への変換効率を考慮
        energy_loss = typhoon_tracking_power / self.trust_efficiency

        return energy_loss

    # 状態量計算
    def get_distance(self, target_position):
        """
        ############################ def get_distance ############################

        [ 説明 ]

        台風発電船から目標地点への距離を計算する関数です。

        ##############################################################################

        引数 :
            target_position (taple) : 目標地点の座標(緯度,経度)


        戻り値 :
            distance (float) : 台風発電船から目標地点への距離(km)

        #############################################################################
        """

        A_position = (self.ship_lat, self.ship_lon)

        # AーB間距離
        distance = geodesic(A_position, target_position).km

        return distance

    def get_direction(self, target_position):
        """
        ############################ def get_distance ############################

        [ 説明 ]

        台風発電船から目標地点への方位を計算する関数です。

        反時計回り(左回り)を正として角度（度数法）を返します。

        ##############################################################################

        引数 :
            target_position (taple) : 目標地点の座標(緯度,経度)


        戻り値 :
            direction (float) : 台風発電船から目標地点への方位(度)

        #############################################################################
        """
        # 北を基準に角度を定義する
        x1 = 10 + self.ship_lat  # 北緯(10+船の緯度)度
        y1 = 0 + self.ship_lon  # 東経(0+船の経度)度

        # 外積計算　正なら左回り、負なら右回り
        # 船の座標 (回転中心)
        x2 = self.ship_lat
        y2 = self.ship_lon

        # 目標地点の座標
        x3 = target_position[0]
        y3 = target_position[1]

        gaiseki = (x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2)
        naiseki = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
        size12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        size32 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

        if gaiseki == 0:  # 直線上

            if naiseki < 0:
                direction = np.pi
            else:
                direction = 0

        elif gaiseki < 0:  # 右回り
            direction = -np.arccos((naiseki) / (size12 * size32))

        elif gaiseki > 0:  # 左回り
            direction = np.arccos((naiseki) / (size12 * size32))

        else:
            print("direction error")

        direction = np.rad2deg(direction)

        return direction

    # 状態量計算
    def change_kt_kmh(self):
        """
        ############################ def change_kt_kmh ############################

        [ 説明 ]

        ktをkm/hに変換する関数です

        ##############################################################################


        戻り値 :
            speed_kmh (float) : km/hに変換された船速

        #############################################################################
        """

        speed_kmh = self.speed_kt * 1.852

        return speed_kmh

    # 予報データから台風の目標地点の決定
    def get_target_data(self, year, current_time, time_step):
        """
        ############################ def get_target_data ############################

        [ 説明 ]

        「予報データ」(forecast_data)から目標とする台風を決める関数です。

        予想発電時間は台風発電船が台風に追いついてから台風が消滅するまで追った場合の時間です。

        消滅時間がわからない場合は発生から5日後に台風が消滅するものとして考えます。5日以上存在する場合は予報期間の最後の時刻に消滅すると仮定します。

        台風補足時間は台風発電船が予想される台風の座標に追いつくまでにかかる時間です。

        以上二つの数値を用いて評価用の数値を以下のように計算します。

        評価数値　＝　予想発電時間＊(forecast_weight) - 台風補足時間＊(100 - forecast_weight)

        これを予報データ内の全データで計算して最も評価数値が大きかったものを選びそれを返します。

        2023/05/24追記

        補足時間について、台風発電船の最大船速で到着するのにかかる時間のX倍の時間をかけなければ台風の想定到着時間に目的地に到着できない場合、

        選択肢から除外するものとする。

        Xは判断の基準値として設定されるものとする。

        ##############################################################################

        引数 :
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        戻り値 :
            target_typhoon_data (dataflame) : 追従目標に選ばれた台風のデータ。予報データから1行分のみ取り出した状態。

        #############################################################################
        """

        # 台風の平均存続時間
        # 今回は5日ととりあえずしてある
        TY_mean_time_to_live = 24 * 5
        TY_mean_time_to_live_unix = TY_mean_time_to_live * 3600

        ship_speed_kmh = self.change_kt_kmh()

        # unixtimeでの時間幅
        forecast_time_unix = 3600 * self.forecast_time
        last_forecast_time = int(current_time + forecast_time_unix)
        start_forecast_time = int(current_time + 3600 * time_step)

        # 該当時刻内のデータの抜き出し
        typhoon_data_forecast = self.forecast_data

        # 陸地認識フェーズ　陸地内に入っているデータの消去
        typhoon_data_forecast = typhoon_data_forecast.filter(
            (
                ((pl.col("FORE_LAT") >= 0) & (pl.col("FORE_LAT") <= 13))
                & (pl.col("FORE_LON") >= 127.5)
            )  # p1 ~ p2
            | (
                ((pl.col("FORE_LAT") >= 13) & (pl.col("FORE_LAT") <= 15))
                & (pl.col("FORE_LON") >= 125)
            )  # p25 ~ p255
            | (
                ((pl.col("FORE_LAT") >= 15) & (pl.col("FORE_LAT") <= 24))
                & (pl.col("FORE_LON") >= 123)
            )  # p3 ~ p4
            | (
                ((pl.col("FORE_LAT") >= 24) & (pl.col("FORE_LAT") <= 26))
                & (pl.col("FORE_LON") >= 126)
            )  # p5 ~ p55
            | (
                ((pl.col("FORE_LAT") >= 26) & (pl.col("FORE_LAT") <= 28))
                & (pl.col("FORE_LON") >= 130.1)
            )  # p555 ~ p6
            | (
                ((pl.col("FORE_LAT") >= 28) & (pl.col("FORE_LAT") <= 32.2))
                & (pl.col("FORE_LON") >= 132.4)
            )  # p7 ~ p8
            | (
                ((pl.col("FORE_LAT") >= 32.2) & (pl.col("FORE_LAT") <= 34))
                & (pl.col("FORE_LON") >= 137.2)
            )  # p9 ~ p10
            | (
                ((pl.col("FORE_LAT") >= 34) & (pl.col("FORE_LAT") <= 41.2))
                & (pl.col("FORE_LON") >= 143)
            )  # p11 ~ p12
            | (
                ((pl.col("FORE_LAT") >= 41.2) & (pl.col("FORE_LAT") <= 44))
                & (pl.col("FORE_LON") >= 149)
            )  # p13 ~ p14
            | (
                ((pl.col("FORE_LAT") >= 44) & (pl.col("FORE_LAT") <= 50))
                & (pl.col("FORE_LON") >= 156)
            )  # p15 ~ p16
            | ((pl.col("FORE_LAT") >= 50))  # p16 ~
        )

        # 台風番号順に並び替え
        typhoon_data_forecast = typhoon_data_forecast.select(
            pl.col("*").sort_by("TYPHOON NUMBER")
        )

        if len(typhoon_data_forecast) != 0:
            # 予報における一番若い番号の台風の取得
            TY_start_bangou = typhoon_data_forecast[0, "TYPHOON NUMBER"]
            TY_end_bangou = typhoon_data_forecast[
                len(typhoon_data_forecast) - 1, "TYPHOON NUMBER"
            ]

            # 台風発生時刻の取得番号
            occurrence_time_acquisition_num = TY_start_bangou - (year - 2000) * 100
            # 台風番号より台風の個数を調べる
            TY_num_forecast = typhoon_data_forecast.n_unique("TYPHOON NUMBER")

            # 予報期間内の台風がどの時刻まで予報されているのかを記録するリスト
            TY_forecast_end_time = []
            # 欠落した番号がいた場合のリスト
            missing_num_list = []
            # 各台風番号での予測終了時刻の取得
            TY_bangou = TY_start_bangou

            for i in range(TY_num_forecast):

                # 番号が後なのに先に発生しているケースがあったのでその応急処置
                # if (i == TY_num_forecast -1) and (error_num == 1):
                # print("ERROR",TY_bangou,TY_end_bangou,"unixtime",current_time,"~",last_forecast_time)
                typhoon_data_by_num = typhoon_data_forecast.filter(
                    pl.col("TYPHOON NUMBER") == TY_bangou
                )
                while len(typhoon_data_by_num) == 0:
                    if len(typhoon_data_by_num) == 0:
                        missing_num_list.append(TY_bangou)
                        TY_bangou = TY_bangou + 1
                        typhoon_data_by_num = typhoon_data_forecast.filter(
                            pl.col("TYPHOON NUMBER") == TY_bangou
                        )

                typhoon_data_by_num = typhoon_data_forecast.filter(
                    pl.col("TYPHOON NUMBER") == TY_bangou
                )
                typhoon_data_by_num = typhoon_data_by_num.select(
                    pl.col("*").sort_by("unixtime", descending=True)
                )
                TY_forecast_end_time.append(typhoon_data_by_num[0, "unixtime"])

                TY_bangou = TY_bangou + 1

            # 現在地から予測される台風の位置までの距離
            distance_list = []
            # 現在地から予測される台風の位置に到着する時刻
            ship_catch_time_list = []
            # 現在時刻から目的地に台風が到着するのにかかる時間
            arrival_time_list = []
            # 上記二つの時間の倍率
            time_times_list = []
            # 到着時から追従した場合に予測される発電量
            projected_elect_gene_time = []
            # 現在地から台風の位置に到着するのに実際必要な時刻
            true_ship_catch_time_list = []

            # 台風番号順に並び替えて当該時刻に発電船が到着した場合に最後まで追従できる発電時間を項目として作る
            # last_forecast_time(予報内の最終台風存続確認時刻)と最後の時刻が等しい場合には平均の存続時間で発電量を推定する
            # 今回は良い方法が思いつかなかったので全データから台風発生時刻を取得する。本来は発生時刻を記録しておきたい。

            # 台風発生時刻の取得
            # 各台風番号で開始時刻の取得

            data_num = len(typhoon_data_forecast)

            # データごとに予測発電時間を入力する
            for i in range(data_num):
                # 仮の発電開始時間
                gene_start_time = typhoon_data_forecast[i, "unixtime"]
                # 考える台風番号
                TY_predict_bangou = typhoon_data_forecast[i, "TYPHOON NUMBER"]

                adjustment_num = 0
                for j in range(len(missing_num_list)):
                    if TY_predict_bangou >= missing_num_list[j]:
                        adjustment_num = adjustment_num + 1

                # データ参照用の番号
                data_reference_num = (
                    TY_predict_bangou - TY_start_bangou - adjustment_num
                )

                # 当該台風の予報内での終了時刻
                end_time_forecast_TY = TY_forecast_end_time[data_reference_num]
                # 当該台風の発生時刻
                start_time_forecast_TY = self.TY_start_time_list.get(
                    TY_predict_bangou, None
                )

                # 台風最終予想時刻による場合分け。予報期間終了時刻と同じ場合はその後も台風が続くものとして、平均存続時間を用いる。
                # 平均存続時間よりも長く続いている台風の場合は最終予想時刻までを発電するものと仮定する。
                if (end_time_forecast_TY == last_forecast_time) and (
                    (end_time_forecast_TY - start_time_forecast_TY)
                    < TY_mean_time_to_live_unix
                ):

                    # 予想される発電時間[hour]を出す
                    forecast_gene_time = (
                        start_time_forecast_TY
                        + TY_mean_time_to_live_unix
                        - gene_start_time
                    ) / 3600
                    # end_time_list.append(start_time_forecast_TY + TY_mean_time_to_live_unix)
                    # shori.append(1)

                else:

                    # 予想期間内で発電時間[hour]を出す
                    forecast_gene_time = (end_time_forecast_TY - gene_start_time) / 3600
                    # end_time_list.append(end_time_forecast_TY)
                    # shori.append(2)

                projected_elect_gene_time.append(forecast_gene_time)

            # データフレームに予想発電時間の項目を追加する
            # typhoon_data_forecast["処理"] = shori
            # typhoon_data_forecast["予想発電開始時間"] = start_time_list
            # typhoon_data_forecast["予想発電終了時間"] = end_time_list
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(projected_elect_gene_time).alias("FORE_GENE_TIME")
            )

            # 距離の判別させる
            for i in range(data_num):

                typhoon_posi_future = (
                    typhoon_data_forecast[i, "FORE_LAT"],
                    typhoon_data_forecast[i, "FORE_LON"],
                )
                ship_typhoon_dis = self.get_distance(typhoon_posi_future)

                # 座標間の距離から到着時刻を計算する
                if ship_speed_kmh == 0:
                    ship_speed_kmh = self.max_speed * 1.852
                ship_catch_time = math.ceil(ship_typhoon_dis / ship_speed_kmh)

                # 現時刻から台風がその地点に到達するまでにかかる時間を出す
                typhoon_arrival_time = int(
                    (typhoon_data_forecast[i, "unixtime"] - current_time) / 3600
                )

                # arrival_time_list.append(typhoon_arrival_time)

                # ship_catch_time_list.append(ship_catch_time)

                time_times_list.append(ship_catch_time / typhoon_arrival_time)

                # 台風の到着予定時刻と船の到着予定時刻の内遅い方をとる
                if typhoon_arrival_time > ship_catch_time:
                    true_ship_catch_time_list.append(typhoon_arrival_time)
                else:
                    true_ship_catch_time_list.append(ship_catch_time)

                distance_list.append(ship_typhoon_dis)

                # print(ship_typhoon_dis)
                # print(typhoon_data_forecast.loc[i,"distance"])

            # 台風の距離を一応書いておく
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(distance_list).alias("distance")
            )
            # データフレームに距離の項目を追加する
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(true_ship_catch_time_list).alias("TY_CATCH_TIME")
            )
            # データフレームに距離の項目を追加する
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(time_times_list).alias("JUDGE_TIME_TIMES")
            )

            # 予想発電時間と台風補足時間の差を出す
            time_difference = []
            for i in range(len(typhoon_data_forecast)):
                time_difference.append(
                    typhoon_data_forecast[i, "FORE_GENE_TIME"] * self.forecast_weight
                    - typhoon_data_forecast[i, "TY_CATCH_TIME"]
                    * (100 - self.forecast_weight)
                )

            # 予想発電時間と台風補足時間の差をデータに追加。名前は時間対効果
            typhoon_data_forecast = typhoon_data_forecast.with_columns(
                pl.Series(time_difference).alias("TIME_EFFECT")
            )

            # 基準倍数以下の時間で到達できる台風のみをのこす。[実際の到達時間] ≦ (台風の到着時間) が実際の判定基準
            typhoon_data_forecast = typhoon_data_forecast.filter(
                pl.col("JUDGE_TIME_TIMES") <= self.judge_time_times
            )

            # データを時間対効果が大きい順に並び替える
            typhoon_data_forecast = typhoon_data_forecast.select(
                pl.col("*").sort_by("TIME_EFFECT", descending=True)
            )

            if len(typhoon_data_forecast) != 0:
                # 出力データフレーム
                time_effect = typhoon_data_forecast[0, "TIME_EFFECT"]

                typhoon_data_forecast = typhoon_data_forecast.filter(
                    pl.col("TIME_EFFECT") == time_effect
                )

                if len(typhoon_data_forecast) > 1:
                    # データを発電時間が長い順に並び替える
                    typhoon_data_forecast = typhoon_data_forecast.select(
                        pl.col("*").sort_by("FORE_GENE_TIME", descending=True)
                    )

                    gene_time_max = typhoon_data_forecast[0, "FORE_GENE_TIME"]
                    typhoon_data_forecast = typhoon_data_forecast.filter(
                        pl.col("FORE_GENE_TIME") == gene_time_max
                    )

                    if len(typhoon_data_forecast) > 1:
                        # データを台風補足時間が短い順に並び替える
                        typhoon_data_forecast = typhoon_data_forecast.select(
                            pl.col("*").sort_by("TY_CATCH_TIME"), ascending=True
                        )

                        gene_time_max = typhoon_data_forecast[0, "TY_CATCH_TIME"]
                        typhoon_data_forecast = typhoon_data_forecast.filter(
                            pl.col("TY_CATCH_TIME") == gene_time_max
                        )

        return typhoon_data_forecast

    # timestep後の時刻における追従対象台風の座標取得
    def get_next_time_target_TY_data(self, time_step, current_time):
        """
        ############################ def get_next_time_target_TY_data ############################

        [ 説明 ]

        get_target_dataで選ばれ、追従対象となった台風のcurrent_time + time_stepの時刻での座標を取得する関数です。

        存在しない場合は空のデータフレームが返ります。

        ##############################################################################

        引数 :
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        戻り値 :
            next_time_target_data (dataflame) : 追従目標に選ばれた台風の次の時刻でのデータ

        #############################################################################
        """

        forecast_data = self.forecast_data
        next_time = int(current_time + time_step * 3600)
        target_TY = int(self.target_name)

        next_time_target_data = forecast_data.filter(
            (pl.col("unixtime") == next_time) & (pl.col("TYPHOON NUMBER") == target_TY)
        )

        return next_time_target_data

    # 状態量計算
    # 次の時刻での船の座標
    def get_next_position(self, time_step):
        """
        ############################ def get_next_position ############################

        [ 説明 ]

        台風発電船の次の時刻での座標を計算するための関数です。

        現在地から目標地点まで直線に進んだ場合にいる座標を計算して返します。

        台風発電船が次の時刻で目的地に到着できる場合は座標は目的地のものになります。

        状態量が更新されるのみなのでreturnでの戻り値はありません。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        target_position = (self.target_lat, self.target_lon)

        # 目的地と現在地の距離
        Goal_now_distance = self.get_distance(target_position)  # [km]

        # 船がtime_step時間で進める距離
        advance_distance = self.change_kt_kmh() * time_step

        # 緯度の差
        g_lat = self.target_lat
        n_lat = self.ship_lat

        lat_difference = g_lat - n_lat

        # 経度の差
        g_lon = self.target_lon
        n_lon = self.ship_lon

        lon_difference = g_lon - n_lon

        # 進める距離と目的地までの距離の比を出す
        if Goal_now_distance != 0:
            distance_ratio = advance_distance / Goal_now_distance
        else:
            distance_ratio = 0

        # 念の為の分岐
        # 距離の比が1を超える場合目的地に到着できることになるので座標を目的地へ、そうでないなら当該距離進める

        if distance_ratio < 1 and distance_ratio > 0:

            # 次の時間にいるであろう緯度
            next_lat = lat_difference * distance_ratio + n_lat

            # 次の時間にいるであろう経度
            next_lon = lon_difference * distance_ratio + n_lon

        else:

            # 次の時間にいるであろう緯度
            next_lat = g_lat

            # 次の時間にいるであろう経度
            next_lon = g_lon

        next_position = (next_lat, next_lon)

        # 移動前の座標の記録（消費電力計算に使用）
        self.ship_lat_before = self.ship_lat
        self.ship_lon_before = self.ship_lon

        self.ship_lat = next_lat
        self.ship_lon = next_lon

    def return_base_action(self, time_step, Storage_base):
        """
        ############################ def return_base_action ############################

        [ 説明 ]

        台風発電船が拠点に帰港する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        self.target_lat = self.base_lat
        self.target_lon = self.base_lon

        self.go_base = 1
        self.brance_condition = "battery capacity exceeded specified ratio"

        # 帰港での船速の入力
        self.speed_kt = self.nomal_ave_speed

        self.target_name = "base station"

        base_ship_dis_time = (
            self.get_distance((self.base_lat, self.base_lon)) / self.change_kt_kmh()
        )
        # timestep後に拠点に船がついている場合
        if base_ship_dis_time <= time_step:
            self.brance_condition = "arrival at base station"
            self.go_base = 0
            self.TY_and_base_action = 0

            self.speed_kt = 0

            # 電気/MCHの積み下ろし
            if self.storage >= self.operational_reserve:
                # 拠点の貯蔵容量を超える場合は拠点の容量に合わせる
                Storage_base_capacity_dif = (
                    Storage_base.max_storage - Storage_base.storage
                )
                tpg_ship_supply_elect = self.storage - self.operational_reserve
                if tpg_ship_supply_elect > Storage_base_capacity_dif:
                    self.supply_elect = Storage_base_capacity_dif
                    self.storage = self.storage - Storage_base_capacity_dif
                else:
                    self.supply_elect = self.storage - self.operational_reserve
                    self.storage = self.operational_reserve
            else:
                self.supply_elect = 0
                # self.storageはそのまま

            self.sum_supply_elect = self.sum_supply_elect + self.supply_elect

            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 0  # 0なら消費していない、1なら消費

            # 発電船状態入力
            self.ship_state = 0  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        else:
            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 1  # 0なら消費していない、1なら消費

            # 発電船状態入力
            self.ship_state = 4  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

    def return_standby_action(self, time_step, Storage_base):
        """
        ############################ def return_standby_action ############################

        [ 説明 ]

        台風発電船が待機位置に向かう場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]
            Storage_base (class) : 貯蔵拠点の情報を持つクラス


        #############################################################################
        """

        self.brance_condition = "returning to standby position as no typhoon"

        self.target_lat = self.standby_lat
        self.target_lon = self.standby_lon

        self.target_name = "Standby position"

        self.speed_kt = self.nomal_ave_speed
        standby_ship_dis_time = (
            self.get_distance((self.standby_lat, self.standby_lon))
            / self.change_kt_kmh()
        )

        if standby_ship_dis_time <= time_step:
            self.brance_condition = "arrival at standby position"

            self.speed_kt = 0

            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 0  # 0なら消費していない、1なら消費

            if self.standby_lat == self.base_lat and self.standby_lon == self.base_lon:

                # 電気/MCHの積み下ろし
                if self.storage >= self.operational_reserve:

                    # 拠点の貯蔵容量を超える場合は拠点の容量に合わせる
                    Storage_base_capacity_dif = (
                        Storage_base.max_storage - Storage_base.storage
                    )
                    tpg_ship_supply_elect = self.storage - self.operational_reserve
                    if tpg_ship_supply_elect > Storage_base_capacity_dif:
                        self.supply_elect = Storage_base_capacity_dif
                        self.storage = self.storage - Storage_base_capacity_dif
                    else:
                        self.supply_elect = self.storage - self.operational_reserve
                        self.storage = self.operational_reserve
                else:
                    self.supply_elect = 0
                    # self.storageはそのまま

                self.sum_supply_elect = self.sum_supply_elect + self.supply_elect

                # 電気推進機用電力の供給
                self.electric_propulsion_storage_wh = (
                    self.electric_propulsion_max_storage_wh
                )
                self.electric_propulsion_storage_state = str("charge in Standby")

            # 発電船状態入力
            self.ship_state = 0  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        else:
            # 発電の有無の判断
            self.GS_gene_judge = 0  # 0なら発電していない、1なら発電
            # 電力消費の有無の判断
            self.GS_loss_judge = 1  # 0なら消費していない、1なら消費

            # 発電船状態入力
            self.ship_state = 4  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

    def typhoon_chase_action(self, time_step, Storage_base):
        """
        ############################ def typhoon_chase_action ############################

        [ 説明 ]

        台風発電船が台風を追従する場合の基本的な行動をまとめた関数です。

        行っていることは、目的地の設定、行動の記録、船速の決定、到着の判断です。

        追加で、拠点を経由するのかの判断も行います。

        ##############################################################################

        引数 :
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        self.speed_kt = self.max_speed

        max_speed_kmh = self.change_kt_kmh()

        # GS_dis_judge = TY_tracking_speed_kmh*self.distance_judge_hours

        TY_tracking_speed = (self.target_TY_data[0, "distance"]) / (
            self.target_TY_data[0, "TY_CATCH_TIME"]
        )  # その場から台風へ時間ぴったりに着くように移動する場合の船速

        # 算出したTY_tracking_speedが最大船速を超えないか判断。超える場合は最大船速に置き換え
        if TY_tracking_speed > max_speed_kmh:
            self.speed_kt = self.max_speed
        else:
            # km/hをktに変換
            self.speed_kt = TY_tracking_speed / 1.852

        # 追従対象の台風までの距離
        GS_TY_dis = self.target_TY_data[0, "distance"]

        self.brance_condition = "tracking typhoon at maximum ship speed started"

        self.target_lat = self.target_TY_data[0, "FORE_LAT"]
        self.target_lon = self.target_TY_data[0, "FORE_LON"]

        if self.target_TY_data[0, "TY_CATCH_TIME"] <= time_step:
            self.brance_condition = "arrived at typhoon"
            self.speed_kt = self.generating_speed_kt

            self.GS_gene_judge = 1

            self.GS_loss_judge = 0

            # 発電船状態入力
            self.ship_state = 1  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        else:

            self.brance_condition = "tracking typhoon"

            self.GS_gene_judge = 0

            self.GS_loss_judge = 1

            # 発電船状態入力
            self.ship_state = 2  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

            # 座標間距離を用いた発電の有無のチェック用数値
            self.distance_check = 1  # 1ならチェック必要

        # 拠点を経由できるか、するかの判断フェーズ
        direction_to_TY = self.get_direction((self.target_lat, self.target_lon))
        direction_to_base = self.get_direction((self.base_lat, self.base_lon))
        direction_difference = np.abs(direction_to_TY - direction_to_base)
        targetTY_base_dis = geodesic(
            (self.target_lat, self.target_lon), (self.base_lat, self.base_lon)
        ).km
        need_distance = (
            self.get_distance((self.base_lat, self.base_lon)) + targetTY_base_dis
        )
        max_speed_kmh = self.max_speed * 1.852
        need_time_hours = need_distance / max_speed_kmh
        TY_catch_time = self.target_TY_data[0, "TY_CATCH_TIME"]

        TY_distance = self.get_distance((self.target_lat, self.target_lon))
        base_distance = self.get_distance((self.base_lat, self.base_lon))
        distance_dis = TY_distance - base_distance

        self.TY_and_base_action = 0

        if self.storage_percentage >= self.govia_base_judge_energy_storage_per:
            if need_time_hours <= TY_catch_time:
                # 元の目的地に問題なくつけるのであれば即実行
                self.speed_kt = self.max_speed
                self.TY_and_base_action = (
                    1  # 台風に向かいながら拠点に帰港する行動のフラグ
                )

                self.return_base_action(time_step, Storage_base)

                self.brance_condition = "tracking typhoon via base"

            else:
                if direction_difference < self.judge_direction and distance_dis > 0:
                    # 拠点の方が近くて、方位に大きな差がなければとりあえず経由する
                    self.speed_kt = self.max_speed
                    self.TY_and_base_action = (
                        1  # 台風に向かいながら拠点に帰港する行動のフラグ
                    )

                    self.return_base_action(time_step, Storage_base)

                    self.brance_condition = "tracking typhoon via base"

    # 状態量計算
    # 行動判定も入っているので機能の要素も入っている？
    # 全てのパラメータを次の時刻のものに変える処理
    def get_next_ship_state(
        self, year, current_time, time_step, wind_data, Storage_base
    ):
        """
        ############################ def get_next_ship_state ############################

        [ 説明 ]

        台風発電船というエージェントの行動規則そのものの設定であるとともに各分岐条件での状態量の更新を行う関数です。

        行動規則は次のように設定されています。

        1.その時刻での蓄電量の割合がX％以上なら拠点へ帰還

        2.台風がない場合待機位置へ帰還

        3.台風が存在し、追いついている場合発電

        4.台風が存在し、追従中でかつそれが近い場合最大船速で追従

        5.台風が存在し、追従中でかつそれが遠い場合低速で追従

        以上の順番で台風発電船が置かれている状況を判断し、対応した行動を台風発電船がとるように設定している。

        そして、各行動に対応した状態量の更新を行なっている。

        ##############################################################################

        引数 :
            year (int) : シミュレーションを行う年
            current_time (int) : シミュレーション上の現在時刻[unixtime]
            time_step (int) : シミュレーションにおける時間の進み幅[hours]


        #############################################################################
        """

        self.distance_check = 0
        self.electric_propulsion_storage_state = str("no action")
        self.standby_via_base = 0

        # 蓄電量X％以上の場合
        if (
            self.storage_percentage >= self.judge_energy_storage_per
            or self.go_base == 1
        ):
            # if self.go_base == 1:
            self.speed_kt = self.nomal_ave_speed

            self.return_base_action(time_step, Storage_base)

            if self.standby_via_base == 1:
                self.brance_condition = "return standby via base"

            ############  ここでデータ取得から判断させるよりも台風発電の選択肢に行った時にフラグを立てる方が良いかも？  ###############

            # 追従対象の台風が存在するか判別
            self.target_TY_data = self.get_target_data(year, current_time, time_step)
            typhoon_num = len(self.target_TY_data)

            #############  近くに寄った場合に帰るという選択肢の追加  #####################

            # base_ship_dis = self.get_distance((self.base_lat,self.base_lon))

            if (
                typhoon_num == 0
                or self.storage_percentage >= self.judge_energy_storage_per
            ):  # 台風がないまたは蓄電容量規定値超え

                # 追従対象の台風がないことにする

                self.target_TY = 0

                self.next_TY_lat = 0
                self.next_TY_lon = 0
                self.next_ship_TY_dis = np.nan

            elif (
                self.storage_percentage >= self.govia_base_judge_energy_storage_per
            ):  # 少量の蓄電でも戻る場合の基準値を利用した場合

                if typhoon_num == 0:

                    # 追従対象の台風がないことにする

                    self.target_TY = 0

                    self.next_TY_lat = 0
                    self.next_TY_lon = 0
                    self.next_ship_TY_dis = np.nan

                elif self.TY_and_base_action == 1:

                    # 台風が来ているけど途中でよる場合の処理
                    self.brance_condition = "tracking typhoon via base"

                    # 最大船速でとっとと戻る
                    self.speed_kt = self.max_speed

                    self.target_name = str(self.target_TY_data[0, "TYPHOON NUMBER"])
                    self.target_TY = self.target_TY_data[0, "TYPHOON NUMBER"]

                    comparison_lat = self.target_TY_data[0, "FORE_LAT"]
                    comparison_lon = self.target_TY_data[0, "FORE_LON"]

                    next_time_TY_data = self.get_next_time_target_TY_data(
                        time_step, current_time
                    )

                    if len(next_time_TY_data) != 0:
                        self.next_TY_lat = next_time_TY_data[0, "FORE_LAT"]
                        self.next_TY_lon = next_time_TY_data[0, "FORE_LON"]
                        next_TY_locate = (self.next_TY_lat, self.next_TY_lon)

                        self.next_ship_TY_dis = self.get_distance(next_TY_locate)

                    else:
                        # 追従対象の台風がないことにする
                        self.next_TY_lat = 0
                        self.next_TY_lon = 0
                        self.next_ship_TY_dis = np.nan

                    if (
                        self.target_TY_lat != comparison_lat
                        or self.target_TY_lon != comparison_lon
                    ):

                        # 目標地点が変わりそうなら台風追従行動の方で再検討
                        self.typhoon_chase_action(time_step, Storage_base)

        # 蓄電量90％未満の場合
        else:

            self.next_TY_lat = 0
            self.next_TY_lon = 0
            self.next_ship_TY_dis = np.nan

            self.speed_kt = self.max_speed
            # 追従対象の台風が存在するか判別
            self.target_TY_data = self.get_target_data(year, current_time, time_step)
            typhoon_num = len(self.target_TY_data)

            # 待機位置へ帰還
            if typhoon_num == 0:

                if self.storage_percentage >= self.govia_base_judge_energy_storage_per:
                    self.return_base_action(time_step, Storage_base)
                    self.brance_condition = "return standby via base"
                    self.standby_via_base = 1
                    self.target_TY = 0
                else:
                    self.return_standby_action(time_step, Storage_base)
                    self.target_TY = 0

            # 追従対象の台風が存在する場合
            elif typhoon_num >= 1:

                self.target_name = str(self.target_TY_data[0, "TYPHOON NUMBER"])
                self.target_TY = self.target_TY_data[0, "TYPHOON NUMBER"]

                next_time_TY_data = self.get_next_time_target_TY_data(
                    time_step, current_time
                )

                if len(next_time_TY_data) != 0:
                    self.next_TY_lat = next_time_TY_data[0, "FORE_LAT"]
                    self.next_TY_lon = next_time_TY_data[0, "FORE_LON"]
                    next_TY_locate = (self.next_TY_lat, self.next_TY_lon)

                    self.next_ship_TY_dis = self.get_distance(next_TY_locate)

                self.typhoon_chase_action(time_step, Storage_base)

                self.target_TY_lat = self.target_TY_data[0, "FORE_LAT"]
                self.target_TY_lon = self.target_TY_data[0, "FORE_LAT"]

                ####

                ########### 低速追従は考えないものとする ##############

                # elif target_TY_data[0,"TY_CATCH_TIME"] > judge_time and GS_TY_dis > GS_dis_judge:
                #    self.brance_condition = "typhoon is at a distance"

                #    self.speed_kt = self.max_speed

                #    GS_gene_judge = 0

                #    GS_loss_judge = 1

                #    self.brance_condition = "tracking typhoon at low speed from a distance"
                # 発電船状態入力
                #    self.ship_state = 2.5  #通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

                #    self.target_lat = target_TY_data[0,"FORE_LAT"]
                #    self.target_lon = target_TY_data[0,"FORE_LON"]

        # 次の時刻の発電船座標取得
        self.get_next_position(time_step)

        ##########現在時刻＋timestepの台風の座標を取得しておく##########
        # それを用いて台風の50km圏内に入っているかを考える分岐を作る
        if self.distance_check == 1:
            # next_time_TY_data = self.get_next_time_target_TY_data(time_step,current_time)

            self.distance_check = 0

            if len(next_time_TY_data) != 0:
                next_ship_locate = (self.ship_lat, self.ship_lon)

                self.next_TY_lat = next_time_TY_data[0, "FORE_LAT"]
                self.next_TY_lon = next_time_TY_data[0, "FORE_LON"]

                next_TY_locate = (self.next_TY_lat, self.next_TY_lon)

                self.next_ship_TY_dis = self.get_distance(next_TY_locate)

            if (
                len(next_time_TY_data) != 0
                and self.next_ship_TY_dis <= self.typhoon_effective_range
            ):
                self.brance_condition = "within 50km of a typhoon following"

                self.GS_gene_judge = 1
                self.GS_loss_judge = 0

                self.ship_state = 1  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

            else:
                # self.brance_condition = "beyond 50km of a typhoon following"

                self.GS_gene_judge = 0
                self.GS_loss_judge = 1

                self.ship_state = 2  # 通常航行、待機 = 0 , 発電状態　= 1 , 台風追従　= 2 , 台風低速追従 = 2.5 , 拠点回航 = 3 , 待機位置回航 = 4

        ##########################################################

        ############

        # 現在この関数での出力は次の時刻での　船の状態　追従目標　船速　座標　単位時間消費電力・発電量　保有電力　保有電力割合　目標地点との距離　となっている

        # その時刻〜次の時刻での消費仕事量計算
        self.loss_work = (
            self.calculate_power_consumption(wind_data, time_step) * self.GS_loss_judge
        )

        # その時刻〜次の時刻での発電量計算
        self.gene_elect = self.generator_rated_output_w * time_step * self.GS_gene_judge

        # 推進機用の電力供給・消費
        if self.GS_loss_judge == 1:  # 消費している場合
            if (
                self.electric_propulsion_storage_wh
                - self.loss_work / self.trust_efficiency
            ) >= 0:  # 推進用の電力で事足りる場合
                self.electric_propulsion_storage_wh = (
                    self.electric_propulsion_storage_wh
                    - self.loss_work / self.trust_efficiency
                )
                self.loss_elect = self.loss_work / self.trust_efficiency
                if self.loss_elect == 0:
                    self.electric_propulsion_storage_state = str("no charge")
                else:
                    self.electric_propulsion_storage_state = str(
                        "use only power storage for trust"
                    )
            else:  # 推進用の電力で事足りない場合
                loss_elect_trust = (
                    self.loss_work / self.trust_efficiency
                    - self.electric_propulsion_storage_wh
                )
                self.electric_propulsion_storage_wh = 0
                self.storage = (
                    self.storage - loss_elect_trust / self.carrier_to_elect_efficiency
                )  # 不足分をエネルギーキャリアから補う（電気変換する場合もある）
                self.loss_elect = (
                    self.electric_propulsion_storage_wh
                    + loss_elect_trust / self.carrier_to_elect_efficiency
                )
                loss_elect_trust = 0  # 念のため初期化
                self.electric_propulsion_storage_state = str(
                    "use power from trust's and Energy carrier's storage"
                )
            self.gene_carrier = 0

        elif self.GS_gene_judge == 1:  # 発電している場合
            self.loss_elect = self.loss_work
            if self.gene_elect < (
                self.electric_propulsion_max_storage_wh
                - self.electric_propulsion_storage_wh
            ):  # 推進用の電力を使用しており消費量が発電量を上回る場合
                self.electric_propulsion_storage_wh = (
                    self.electric_propulsion_storage_wh + self.gene_elect
                )
                self.gene_carrier = 0
                self.electric_propulsion_storage_state = str(
                    "charge power storage for trust"
                )
            elif (
                self.electric_propulsion_storage_wh
                < self.electric_propulsion_max_storage_wh
            ):  # 推進用の電力を使用しており消費量が発電量を下回る場合
                self.gene_carrier = (
                    self.gene_elect
                    - (
                        self.electric_propulsion_max_storage_wh
                        - self.electric_propulsion_storage_wh
                    )
                ) * self.elect_to_carrier_efficiency
                self.electric_propulsion_storage_wh = (
                    self.electric_propulsion_max_storage_wh
                )
                self.electric_propulsion_storage_state = str(
                    "charge power storage for trust (full)"
                )
            else:  # 推進用の電力を未使用または搭載しておらず、発電量をそのまま蓄える場合
                self.gene_carrier = self.gene_elect * self.elect_to_carrier_efficiency
                self.electric_propulsion_storage_state = str("no charge")
        else:  # 何もしていない場合
            self.loss_elect = self.loss_work
            self.gene_carrier = 0

        self.total_gene_elect = self.total_gene_elect + self.gene_elect
        self.total_gene_carrier = self.total_gene_carrier + self.gene_carrier
        self.total_loss_elect = self.total_loss_elect + self.loss_elect

        self.total_gene_time = self.total_gene_time + time_step * self.GS_gene_judge
        self.total_loss_time = self.total_loss_time + time_step * self.GS_loss_judge

        # 次の時刻での発電船保有電力
        self.storage = self.storage + self.gene_carrier

        if self.storage > self.max_storage:
            self.storage = self.max_storage

        self.storage_percentage = self.storage / self.max_storage * 100

        # 目標地点との距離
        target_position = (self.target_lat, self.target_lon)
        self.target_distance = self.get_distance(target_position)

    def cost_calculate(self):
        """
        ############################ def cost_calculate ############################

        [ 説明 ]

        台風発電船のコストを計算する関数です。

        修論(2025)に沿った設定となっています。

        ##############################################################################
        """

        # 硬翼帆価格[円]
        self.wing_sail_cost = (
            (0.0004444 * self.sail_area + 0.5556) * self.sail_num
        ) * 10**8
        # 水中発電機の価格[円]
        self.underwater_turbine_cost = (
            (0.82 * self.generator_turbine_radius - 3.9) * 10**8 * self.generator_num
        )

        # バッテリーの価格[円] 75ドル/kWhで1ドル=160円 240MWhの電池を必要分搭載するとする。
        n_battery = math.ceil(
            (self.electric_propulsion_max_storage_wh / 10**6) / 240
        )  # バッテリーの個数を端数切り上げで求める
        self.battery_cost = (240 * 10**3 * 75) * n_battery * 160
        # 初期電気推進機用電力の供給料金[円]25円/kWhとする。
        initial_electric_propulsion_cost = 25 * (
            self.electric_propulsion_max_storage_wh / 1000
        )

        # storage_methodで異なるコストを計算
        if self.storage_method == 1:  # 電気貯蔵(コンテナ型)
            # 船体価格+甲板補強価格[円]
            self.hull_cost = (
                0.00483 * ((self.ship_dwt / self.hull_num) ** 0.878) * 10**6 * 160
            ) * self.hull_num + 500000 * self.hull_L_oa * self.hull_B
            # 関連プラントの価格[円]
            Plant_cost = 0
            # エネルギーキャリア関連のコスト[億円]
            n_battery_st = math.ceil(
                (self.max_storage / 10**6) / 240
            )  # バッテリーの個数を端数切り上げで求める
            self.carrier_cost = (240 * 10**3 * 75) * n_battery_st * 160

        elif self.storage_method == 2:  # MCH貯蔵
            self.hull_cost = (
                0.212 * ((self.ship_dwt / self.hull_num) ** 0.5065) * 10**6 * 160
            ) * self.hull_num + 500000 * self.hull_L_oa * self.hull_B
            # 関連プラントの価格[円]
            Plant_cost = 5.0 * 10**9
            # エネルギーキャリア関連のコスト[円]
            # トルエンのコスト[億円] 1トンあたりの価格が1500ドルとし、1ドル=160円とする。self.max_storageは1GWhあたり379トンとし、self.max_storageの3倍量を確保する
            self.carrier_cost = (
                1500 * ((self.max_storage / 10**9) * 379) * 3 * 160 / 10**8
            )

        elif self.storage_method == 3:  # メタン貯蔵
            self.hull_cost = (
                4.41 * 0.212 * ((self.ship_dwt / self.hull_num) ** 0.5065) * 10**6 * 160
            ) * self.hull_num + 500000 * self.hull_L_oa * self.hull_B  # LNG船の補正あり
            # 関連プラントの価格[円]
            Plant_cost = 7.5 * 10**9  # メタン生成＋液化プラント
            # エネルギーキャリア関連のコスト[円]
            # 原料となるCO2のコスト[億円] 1トンあたりの価格が200ドルとし、1ドル=160円とする。
            # 初期のmax_storage[Wh]分のキャリア作成に必要な量に加え、total_gene_carrier[Wh]で作成したぶんのCO2を補填するものとする。
            # 物性より計算　メタン1molの完全燃焼で802kJ=802/3600kWh
            # mol数の計算
            mol_max_storage = self.max_storage / ((802 / 3600) * 1000)
            mol_consume = self.total_gene_carrier / ((802 / 3600) * 1000)
            total_mol = mol_max_storage + mol_consume
            # 同mol数のCO2の重量[t]を44g/molより計算
            CO2_t = total_mol * 44 / 10**6
            self.carrier_cost = 200 * CO2_t * 160 / 10**8

        elif self.storage_method == 4:  # メタノール貯蔵
            self.hull_cost = (
                0.212 * ((self.ship_dwt / self.hull_num) ** 0.5065) * 10**6 * 160
            ) * self.hull_num + 500000 * self.hull_L_oa * self.hull_B  # タンカーと同価格
            # 関連プラントの価格[円]
            Plant_cost = 6.0 * 10**9  # メタノール生成＋ケミカルタンカーの補正あり
            # エネルギーキャリア関連のコスト[円]
            # 原料となるCO2のコスト[億円] 1トンあたりの価格が200ドルとし、1ドル=160円とする。
            # 初期のmax_storage[Wh]分のキャリア作成に必要な量に加え、total_gene_carrier[Wh]で作成したぶんのCO2を補填するものとする。
            # 物性より計算　メタノール1molの完全燃焼で726.2kJ=726.2/3600kWh
            # mol数の計算
            mol_max_storage = self.max_storage / ((726.2 / 3600) * 1000)
            mol_consume = self.total_gene_carrier / ((726.2 / 3600) * 1000)
            total_mol = mol_max_storage + mol_consume
            # 同mol数のCO2の重量[t]を44g/molより計算
            CO2_t = total_mol * 44 / 10**6
            self.carrier_cost = 200 * CO2_t * 160 / 10**8

        elif self.storage_method == 5:  # e-ガソリン貯蔵
            self.hull_cost = (
                0.212 * ((self.ship_dwt / self.hull_num) ** 0.5065) * 10**6 * 160
            ) * self.hull_num + 500000 * self.hull_L_oa * self.hull_B  # タンカーと同価格
            # 関連プラントの価格[円]
            Plant_cost = 5.0 * 10**9  # ガソリン生成プラント
            # エネルギーキャリア関連のコスト[円]
            # 原料となるCO2のコスト[億円] 1トンあたりの価格が200ドルとし、1ドル=160円とする。
            # 初期のmax_storage[Wh]分のキャリア作成に必要な量に加え、total_gene_carrier[Wh]で作成したぶんのCO2を補填するものとする。
            # 代表の分子としてC8H18（オクタン）を用いる
            # オクタン1molの完全燃焼で5500kJ=5500/3600kWh
            # mol数の計算
            mol_max_storage = self.max_storage / ((5500 / 3600) * 1000)
            mol_consume = self.total_gene_carrier / ((5500 / 3600) * 1000)
            total_mol = mol_max_storage + mol_consume
            # 8倍のmol数のCO2の重量[t]を44g/molより計算
            CO2_t = total_mol * 8 * 44 / 10**6
            self.carrier_cost = 200 * CO2_t * 160 / 10**8

        # 電動機モーターの価格[円]　船体価格の10%
        motor_cost = 0.1 * self.hull_cost

        # 船の建造費用[億円]
        self.building_cost = (
            self.hull_cost
            + self.wing_sail_cost
            + self.underwater_turbine_cost
            + Plant_cost
            + motor_cost
            + self.battery_cost
            + initial_electric_propulsion_cost
        ) / 10**8

        # 船の維持費用[億円] 年間で建造コストの3％とする
        self.maintenance_cost = self.building_cost * 0.03
