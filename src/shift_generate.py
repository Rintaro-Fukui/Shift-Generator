import pulp
import pandas as pd

def generate_shifts(employees:list, desired_shifts:dict, required_shifts:dict) -> pd.DataFrame:
    '''
    シフトを作成する関数

    args:
        employees: 従業員のリスト
        desired_shifts: 従業員ごとの希望シフト
        required_shifts: 必要シフト数

    return:
        pivot_table: シフト表
        shortages_df: 不足人数
    '''

    # 問題のインスタンスを作成
    prob = pulp.LpProblem("Shift_Scheduling", pulp.LpMinimize)

    # 変数の辞書を作成（朝と夜のシフト）
    shifts = {}
    shortages = {}
    for day in range(1, 31):
        for shift_type in ['Morning', 'Evening']:
            shortages[(day, shift_type)] = pulp.LpVariable(f"Shortage_{day}_{shift_type}", lowBound=0, cat='Continuous')
            for employee in employees:
                shifts[(employee, day, shift_type)] = pulp.LpVariable(f"{employee}_{day}_{shift_type}", cat='Binary')

    # 目的関数を追加（不足数を最小化し、希望シフトが通った数を最大化）
    # ここでは不足人数に対するペナルティを大きく設定しています。
    prob += pulp.lpSum([shortages[(day, shift_type)] for day in range(1, 31) for shift_type in ['Morning', 'Evening']]) - \
            pulp.lpSum([shifts[(employee, day, shift_type)] for employee in employees for day in range(1, 31) for shift_type in ['Morning', 'Evening'] if (day, shift_type) in desired_shifts[employee]])

    # 制約を追加
    for day in range(1, 31):
        for shift_type in ['Morning', 'Evening']:
            # 必要人数を満たすか、不足分を計算
            prob += (pulp.lpSum(shifts[(employee, day, shift_type)] for employee in employees) + shortages[(day, shift_type)] >= required_shifts[day][shift_type])

    # 希望シフト以外に割り当てない制約
    for employee in employees:
        for day in range(1, 31):
            for shift_type in ['Morning', 'Evening']:
                if (day, shift_type) not in desired_shifts[employee]:
                    prob += shifts[(employee, day, shift_type)] == 0

    # 問題を解く
    prob.solve()

    # 結果をデータフレームに格納
    shifts_data = [(employee, f"{day}-{shift_type}", value.varValue) for (employee, day, shift_type), value in shifts.items() if value.varValue != 0]

    # 結果のデータフレームを作成
    shifts_df = pd.DataFrame(shifts_data, columns=['Employee', 'Day-Shift', 'Assigned'])

    # ピボットテーブルを作成
    pivot_table = shifts_df.pivot_table(index='Employee', columns='Day-Shift', values='Assigned', fill_value=0)

    # 不足人数のデータフレームを作成
    shortages_data = [(f"{day}-{shift_type}", value.varValue) for (day, shift_type), value in shortages.items() if value.varValue != 0]
    shortages_df = pd.DataFrame(shortages_data, columns=['Day-Shift', 'Shortage']).set_index('Day-Shift')

    return pivot_table, shortages_df
