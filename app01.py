import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import io
import math
import re

# ==========================================
# 1. 全域配置與輔助函數 (Global Helpers)
# ==========================================
SYSTEM_VERSION = "v5.8.3 (Feature: JIT WIP Control)"

# 線外製程分類與資源限制設定
OFFLINE_CONFIG = {
    # 1. 超音波熔接 (限制 1 站) -> 絕對單工
    "超音波熔接": ("線外-超音波熔接", 1), 
    
    # 2. LS 雷射 (限制 2 站)
    "LS": ("線外-組裝前LS", 2),
    "雷射": ("線外-組裝前LS", 2),
    
    # 3. PT (限制 1 站) -> 絕對單工
    "PT": ("線外-PT", 1),
    
    # 4. 線邊組裝 (限制 2 站)
    "PKM": ("線外-線邊組裝", 2),
    "裝配": ("線外-線邊組裝", 2),
    "組裝": ("線外-線邊組裝", 2),
    "AS": ("線外-線邊組裝", 2)
}

def get_base_model(product_id):
    if pd.isna(product_id): return ""
    s = str(product_id).strip()
    return s.split('/')[0].strip()

def parse_time_to_mins(time_str):
    try:
        t = datetime.strptime(time_str, "%H:%M")
        return t.hour * 60 + t.minute
    except: return 480 

def create_line_mask(start_str, end_str, days=14):
    total_minutes = days * 24 * 60
    mask = np.zeros(total_minutes, dtype=bool)
    start_min = parse_time_to_mins(start_str)
    end_min = parse_time_to_mins(end_str)
    breaks = [(600, 605), (720, 780), (900, 905), (1020, 1050)]
    
    for day in range(days):
        day_offset = day * 24 * 60
        if end_min > start_min:
            mask[day_offset + start_min : day_offset + end_min] = True
            for b_start, b_end in breaks:
                abs_b_start = day_offset + b_start
                abs_b_end = day_offset + b_end
                mask[abs_b_start : abs_b_end] = False
    return mask

def format_time_str(minute_idx):
    d = (minute_idx // 1440) + 1
    m_of_day = minute_idx % 1440
    hh = m_of_day // 60
    mm = m_of_day % 60
    return f"D{d} {hh:02d}:{mm:02d}"

# 線外分類函數
def categorize_offline(val):
    val_str = str(val)
    for kw, (name, limit) in OFFLINE_CONFIG.items():
        if kw in val_str:
            return name, limit
    return "Online", -1

# 指定線提取函數 (回傳數字 4, 5, 6, 7, 8)
def extract_line_num(val):
    val_str = str(val).upper().replace(' ', '')
    match = re.search(r'LINE(\d+)', val_str)
    if match:
        try: return int(match.group(1))
        except: return 0
    return 0

# 順序提取函數
def get_sequence(val):
    try:
        match = re.search(r'(\d+)', str(val))
        if match: return int(match.group(1))
        return 0 
    except: return 0

def analyze_idle_manpower(timeline_manpower, work_masks, total_manpower, max_sim_minutes):
    global_work_mask = np.zeros(max_sim_minutes, dtype=bool)
    for m in work_masks:
        length = min(len(m), max_sim_minutes)
        global_work_mask[:length] |= m[:length]
        
    idle_records = []
    current_excess, start_time = -1, -1
    
    for t in range(max_sim_minutes):
        if global_work_mask[t]:
            used = timeline_manpower[t]
            excess = total_manpower - used
            if excess != current_excess:
                if current_excess > 0 and start_time != -1:
                    idle_records.append({
                        '開始時間': format_time_str(start_time), '結束時間': format_time_str(t),
                        '持續分鐘': t - start_time, '閒置(多餘)人力': current_excess
                    })
                current_excess, start_time = excess, t
        else:
            if current_excess > 0 and start_time != -1:
                idle_records.append({
                    '開始時間': format_time_str(start_time), '結束時間': format_time_str(t),
                    '持續分鐘': t - start_time, '閒置(多餘)人力': current_excess
                })
            current_excess, start_time = -1, -1
    return pd.DataFrame(idle_records)

def calculate_daily_efficiency(timeline_manpower, line_masks, total_manpower, days_to_analyze=5):
    std_mask = line_masks[0] 
    efficiency_records = []
    
    for day in range(days_to_analyze):
        day_start, day_end = day * 1440, (day + 1) * 1440
        day_std_mask = std_mask[day_start:day_end]
        standard_work_mins = np.sum(day_std_mask)
        day_usage = timeline_manpower[day_start:day_end]
        global_day_mask = np.zeros(1440, dtype=bool)
        for lm in line_masks:
            global_day_mask |= lm[day_start:day_end]
            
        utilized = np.sum(day_usage[global_day_mask])
        total_capacity = total_manpower * standard_work_mins
        
        if standard_work_mins > 0:
            suggested_manpower = math.ceil(utilized / (standard_work_mins * 0.95))
        else:
            suggested_manpower = 0

        efficiency = (utilized / total_capacity * 100) if total_capacity > 0 else 0
        
        if standard_work_mins > 0:
            diff = suggested_manpower - total_manpower
            suggestion = f"需增加 {diff} 人" if diff > 0 else (f"可減少 {abs(diff)} 人" if diff < 0 else "人力完美")
            
            efficiency_records.append({
                '日期': f'D{day+1}', 
                '當日標準工時(分)': standard_work_mins, 
                '現有人力': total_manpower,
                '建議人力(95%效)': suggested_manpower,
                '調度建議': suggestion,
                '實際產出人時': utilized,
                '全廠效率(%)': round(efficiency, 2)
            })
    return pd.DataFrame(efficiency_records)

def calculate_line_utilization(line_usage_matrix, line_masks, total_lines, days_to_analyze=5):
    utilization_records = []
    for day in range(days_to_analyze):
        day_start = day * 1440
        day_end = (day + 1) * 1440
        row = {'日期': f'D{day+1}'}
        for i in range(total_lines):
            available_mask = line_masks[i][day_start:day_end]
            available_mins = np.sum(available_mask)
            busy_mask = line_usage_matrix[i][day_start:day_end]
            valid_busy_mask = busy_mask & available_mask
            busy_mins = np.sum(valid_busy_mask)
            # ★ UI 修正：index i=0 對應 Line 4
            if available_mins > 0:
                util_rate = (busy_mins / available_mins) * 100
                row[f'Line {i+4} (%)'] = round(util_rate, 1)
            else:
                row[f'Line {i+4} (%)'] = "-"
        if any(v != "-" for k, v in row.items() if k != '日期'):
            utilization_records.append(row)
    return pd.DataFrame(utilization_records)

# ==========================================
# 2. 資料讀取區
# ==========================================
def load_and_clean_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.astype(str).str.replace('\n', '').str.replace(' ', '')
        
        col_map = {}
        for col in df.columns:
            if '工單' in col: col_map['Order_ID'] = col
            elif '產品編號' in col: col_map['Product_ID'] = col
            elif '預定裝配' in col: col_map['Plan_Qty'] = col
            elif '實際裝配' in col: col_map['Actual_Qty'] = col
            elif '標準人數' in col: col_map['Manpower_Req'] = col
            elif '工時(分)' in col or '組裝工時' in col: col_map['Total_Man_Minutes'] = col
            elif '項次' in col: col_map['Priority'] = col
            elif '已領料' in col: col_map['Process_Type'] = col
            elif '備註' in col: col_map['Remarks'] = col
            elif '急單' in col: col_map['Rush_Col'] = col
            elif '指定線' in col: col_map['Line_Col'] = col
            
        df = df.rename(columns={v: k for k, v in col_map.items()})
        
        if 'Total_Man_Minutes' not in df.columns: 
            return None, "錯誤：缺少[工時(分)]欄位"
        
        if 'Process_Type' not in df.columns: df['Process_Type'] = '組裝'
        if 'Remarks' not in df.columns: df['Remarks'] = ''
        
        for col in ['Plan_Qty', 'Actual_Qty', 'Manpower_Req', 'Total_Man_Minutes']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0

        df['Qty'] = np.where(df['Actual_Qty'] > 0, df['Actual_Qty'], df['Plan_Qty'])
        df = df[(df['Qty'] > 0) & (df['Manpower_Req'] > 0)]
        df['Base_Model'] = df['Product_ID'].apply(get_base_model)
        
        temp_res = df['Process_Type'].apply(categorize_offline)
        df['Process_Category'] = temp_res.apply(lambda x: x[0])
        df['Concurrency_Limit'] = temp_res.apply(lambda x: x[1])
        df['Is_Offline'] = df['Process_Category'] != "Online"

        if 'Rush_Col' not in df.columns: df['Rush_Col'] = ''
        if 'Line_Col' not in df.columns: df['Line_Col'] = ''

        df['Is_Rush'] = df['Rush_Col'].astype(str).str.contains('急單', na=False) | df['Remarks'].astype(str).str.contains('急單', na=False)

        # 指定線判斷
        df['Target_Line'] = df['Line_Col'].apply(extract_line_num)
        mask_no_line = df['Target_Line'] == 0
        df.loc[mask_no_line, 'Target_Line'] = df.loc[mask_no_line, 'Remarks'].apply(extract_line_num)

        df['Sequence'] = df['Remarks'].apply(get_sequence)
        
        return df, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. 排程運算區
# ==========================================
def run_scheduler(df, total_manpower, total_lines, changeover_mins, line_settings, offline_settings):
    MAX_MINUTES = 14 * 24 * 60 
    
    line_masks = []
    line_cumsums = []
    for setting in line_settings:
        m = create_line_mask(setting["start"], setting["end"], 14)
        line_masks.append(m)
        line_cumsums.append(np.cumsum(m))
        
    offline_mask = create_line_mask(offline_settings["start"], offline_settings["end"], 14)
    offline_cumsum = np.cumsum(offline_mask)

    timeline_manpower = np.zeros(MAX_MINUTES, dtype=int)
    line_usage_matrix = np.zeros((total_lines, MAX_MINUTES), dtype=bool)
    results = []
    line_free_time = [parse_time_to_mins(setting["start"]) for setting in line_settings]
    
    offline_resource_usage = {}
    order_finish_times = {}

    # ★★★ JIT 優化：預先計算每張工單的目標上線產線 ★★★
    # 目的：讓線外製程知道「後續要去哪條線」，進而預判該線的空閒時間
    df_online_parts = df[df['Is_Offline'] == False]
    order_target_line_map = {}
    
    # 模擬簡單的派工邏輯來預判 Target Line
    # 注意：這裡只能做靜態預判，若有動態負載平衡可能會不準，但足以做 JIT 參考
    for _, row in df_online_parts.iterrows():
        t_line = row['Target_Line']
        # 若有指定線
        if t_line > 0:
            target_idx = t_line - 4
        # 若是 N-DE 且沒指定
        elif str(row['Base_Model']).startswith("N-DE") and total_lines >= 4:
            target_idx = 3 # Line 7
        # 其他預設 (這裡簡化取 Line 4，或取最忙碌的線做保守估計)
        else:
            target_idx = 0 
        
        # 建立 Order_ID -> Target_Line_Index 的映射
        # 若一張單有多個線上工序，取最後一個或第一個皆可，這裡取第一個
        if row['Order_ID'] not in order_target_line_map:
            order_target_line_map[row['Order_ID']] = target_idx

    # --- 排程 ---
    df_online = df[df['Is_Offline'] == False].copy()
    family_groups = df_online.groupby('Base_Model')
    
    batches = []
    for base_model, group_df in family_groups:
        rush_orders = group_df[group_df['Is_Rush']]['Order_ID'].unique()
        group_df['Order_Is_Rush'] = group_df['Order_ID'].isin(rush_orders)
        is_batch_rush = group_df['Order_Is_Rush'].any()
        rush_weight = 1000000 if is_batch_rush else 0
        total_work_load = (group_df['Manpower_Req'] * group_df['Total_Man_Minutes']).sum()
        
        target_lines = group_df['Target_Line'].unique()
        specific_requests = [t for t in target_lines if t > 0]
        
        if specific_requests:
            valid_reqs = [t-4 for t in specific_requests if 4 <= t <= (3 + total_lines)]
            candidate_lines = valid_reqs if valid_reqs else [i for i in range(total_lines)]
        else:
            candidate_lines = [i for i in range(total_lines)]

            if str(base_model).startswith("N-DE"):
                if total_lines >= 4:
                    candidate_lines = [3] 

        is_n3610 = str(base_model).startswith("N-3610")
        if not is_n3610:
            if 0 in candidate_lines:
                candidate_lines.remove(0)

        if not candidate_lines:
            candidate_lines = [i for i in range(1, total_lines)] 

        sorted_df = group_df.sort_values(
            by=['Order_Is_Rush', 'Order_ID', 'Sequence', 'Priority'], 
            ascending=[False, True, True, True]
        )

        batches.append({
            'base_model': base_model,
            'df': sorted_df,
            'is_rush': is_batch_rush,
            'weight': rush_weight + total_work_load, 
            'candidate_lines': candidate_lines
        })
    
    batches.sort(key=lambda x: (x['is_rush'], x['weight']), reverse=True)
    
    # 建立一個統一的 DataFrame 進行迭代
    # 雖然上方用了 Batch 邏輯來決定候選產線，但為了統一順序，我們這裡重新整理
    # 這裡的邏輯是：Batch 決定了「線上工單」的順序。
    # 線外工單則依附在這些 Order ID 上，或是獨立存在。
    
    # 為了實作 JIT，我們需要一個全局排序：
    # 1. 依照 Batch 順序展開線上工單
    # 2. 將相關的線外工單插入到正確位置 (Sequence 順序)
    
    # 簡化策略：使用全局排序，但保留 Batch 的候選產線邏輯 (透過 lookup)
    # 建立 Batch Lookup Map
    batch_candidate_map = {} # (Order_ID) -> candidate_lines
    for b in batches:
        for oid in b['df']['Order_ID'].unique():
            batch_candidate_map[oid] = b['candidate_lines']

    # 全局排序
    rush_orders_global = df[df['Is_Rush']]['Order_ID'].unique()
    df['Order_Is_Rush'] = df['Order_ID'].isin(rush_orders_global)
    
    df_sorted = df.sort_values(
        by=['Order_Is_Rush', 'Base_Model', 'Order_ID', 'Sequence', 'Priority'], 
        ascending=[False, True, True, True, True]
    )
    
    # 用來記錄各產線 "上一個生產的產品"
    line_last_model = {i: None for i in range(total_lines)}

    for idx, row in df_sorted.iterrows():
        manpower = int(row['Manpower_Req'])
        total_man_minutes = float(row['Total_Man_Minutes'])
        prod_duration = int(np.ceil(total_man_minutes / manpower)) if manpower > 0 else 0
        
        is_offline = row['Is_Offline']
        seq = row['Sequence']
        order_id = str(row['Order_ID'])
        base_model = row['Base_Model']

        # --- 1. 計算最早可開始時間 ---
        if is_offline:
            start_limit = parse_time_to_mins(offline_settings["start"])
            
            # ★★★ JIT 邏輯：推遲線外生產 ★★★
            # 如果這張單後面要上線，且線上產線目前很忙 (free_time 很晚)
            # 我們就不要太早開始做線外，以免堆積超過 2 天
            if order_id in order_target_line_map:
                target_line_idx = order_target_line_map[order_id]
                # 取得該產線目前的空閒時間
                line_ready_time = line_free_time[target_line_idx]
                # JIT 開始時間 = 線上空閒時間 - 2天 (2880分) - 生產時間
                # 意思是：最快在「上線前 2 天」才做完
                jit_start = line_ready_time - 2880 - prod_duration
                start_limit = max(start_limit, jit_start)
        else:
            start_limit = parse_time_to_mins(line_settings[0]["start"])
            
        min_start_time = start_limit

        if seq > 1:
            prev_seq = seq - 1
            if (order_id, prev_seq) in order_finish_times:
                min_start_time = max(min_start_time, order_finish_times[(order_id, prev_seq)])

        # --- 2. 尋找可用資源 ---
        best_choice = None 

        if is_offline:
            offline_category = row['Process_Category']
            concurrency_limit = row['Concurrency_Limit']
            
            candidate_stations = []
            if concurrency_limit == 0:
                pass 
            else:
                for i in range(1, concurrency_limit + 1):
                    res_id = f"{offline_category}-{i}"
                    if res_id not in offline_resource_usage:
                        offline_resource_usage[res_id] = np.zeros(MAX_MINUTES, dtype=bool)
                    candidate_stations.append(res_id)
            
            stations_to_try = candidate_stations if candidate_stations else [None]
            
            for station_id in stations_to_try:
                res_usage_mask = offline_resource_usage[station_id] if station_id else None
                
                found = False
                t_search = min_start_time
                
                while not found and t_search < MAX_MINUTES - prod_duration:
                    if not offline_mask[t_search]:
                        t_search += 1
                        continue
                    
                    s_val = offline_cumsum[t_search]
                    t_val = s_val + prod_duration
                    if t_val > offline_cumsum[-1]: break
                    t_end = np.searchsorted(offline_cumsum, t_val)
                    
                    if np.any(offline_mask[t_search:t_end]): 
                        i_mask = offline_mask[t_search:t_end]
                        current_max_used = np.max(timeline_manpower[t_search:t_end][i_mask]) if np.any(i_mask) else 0
                        
                        resource_conflict = False
                        if res_usage_mask is not None:
                            if np.any(res_usage_mask[t_search:t_end]):
                                resource_conflict = True
                        
                        if (current_max_used + manpower <= total_manpower) and (not resource_conflict):
                            if best_choice is None or t_search < best_choice[0]:
                                best_choice = (t_search, t_end, station_id)
                            found = True
                        else:
                            t_search += 5
                    else:
                        t_search += 5

        else:
            # 取得 Batch 計算好的候選產線
            candidate_lines = batch_candidate_map.get(row['Order_ID'], [i for i in range(total_lines)])

            for line_idx in candidate_lines:
                curr_mask = line_masks[line_idx]
                curr_cumsum = line_cumsums[line_idx]
                
                setup_time = 0
                if line_last_model[line_idx] is not None and line_last_model[line_idx] != base_model:
                    setup_time = changeover_mins
                
                t_start_search = max(line_free_time[line_idx], min_start_time)
                total_need = setup_time + prod_duration
                
                found = False
                t_search = t_start_search
                
                while not found and t_search < MAX_MINUTES - total_need:
                    if not curr_mask[t_search]:
                        t_search += 1
                        continue
                        
                    s_val = curr_cumsum[t_search]
                    t_val = s_val + total_need
                    if t_val > curr_cumsum[-1]: break
                    t_end = np.searchsorted(curr_
