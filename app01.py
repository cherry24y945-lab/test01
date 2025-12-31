import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import io
import math
import re

# ==========================================
# 1. 全域配置與輔助函數
# ==========================================
SYSTEM_VERSION = "v6.4 (Final: Absolute Rush Priority & Smart Grouping)"

# 線外製程分類與資源限制
OFFLINE_CONFIG = {
    "超音波熔接": ("線外-超音波熔接", 1), 
    "LS": ("線外-組裝前LS", 2),
    "雷射": ("線外-組裝前LS", 2),
    "PT": ("線外-PT", 1),
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
    # 固定休息時間
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

def categorize_offline(val):
    val_str = str(val)
    for kw, (name, limit) in OFFLINE_CONFIG.items():
        if kw in val_str:
            return name, limit
    return "Online", -1

def extract_line_num(val):
    val_str = str(val).upper().replace(' ', '')
    match = re.search(r'LINE(\d+)', val_str)
    if match:
        try: return int(match.group(1))
        except: return 0
    return 0

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
            # index i=0 對應 Line 4
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

        df['Target_Line'] = df['Line_Col'].apply(extract_line_num)
        mask_no_line = df['Target_Line'] == 0
        df.loc[mask_no_line, 'Target_Line'] = df.loc[mask_no_line, 'Remarks'].apply(extract_line_num)

        df['Sequence'] = df['Remarks'].apply(get_sequence)
        
        return df, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. 排程運算區 (Absolute Rush & Smart Grouping)
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
    line_free_time = [parse_time_to_mins(setting["start"]) for setting in line_settings]
    
    offline_resource_usage = {}
    order_finish_times = {}
    
    # 預判 Target Line
    df_online_parts = df[df['Is_Offline'] == False]
    order_target_line_map = {}
    for _, row in df_online_parts.iterrows():
        t_line = row['Target_Line']
        if t_line > 0: 
            target_idx = t_line - 4
        else:
            if str(row['Base_Model']).startswith("N-DE") and total_lines >= 4:
                target_idx = 3 # Line 7
            elif str(row['Base_Model']).startswith("N-3610") and total_lines >= 3:
                target_idx = 2 # Line 6
            else:
                target_idx = 0 # Default Line 4
                
        if row['Order_ID'] not in order_target_line_map:
            order_target_line_map[row['Order_ID']] = target_idx

    # 全局排序
    rush_orders_global = df[df['Is_Rush']]['Order_ID'].unique()
    df['Order_Is_Rush'] = df['Order_ID'].isin(rush_orders_global)
    
    # 排序：急單 -> Base_Model -> 順序
    df_sorted = df.sort_values(
        by=['Order_Is_Rush', 'Base_Model', 'Order_ID', 'Sequence', 'Priority'], 
        ascending=[False, True, True, True, True]
    )
    
    line_last_model = {i: None for i in range(total_lines)}
    
    pending_tasks = df_sorted.to_dict('records')
    results = []
    
    max_loops = len(pending_tasks) * 5 
    loop_count = 0
    
    while pending_tasks and loop_count < max_loops:
        loop_count += 1
        
        best_task_candidate = None 
        
        # 掃描 Pending List
        for i, task in enumerate(pending_tasks):
            manpower = int(task['Manpower_Req'])
            total_man_minutes = float(task['Total_Man_Minutes'])
            prod_duration = int(np.ceil(total_man_minutes / manpower)) if manpower > 0 else 0
            
            is_offline = task['Is_Offline']
            seq = task['Sequence']
            order_id = str(task['Order_ID'])
            base_model = task['Base_Model']
            is_rush = task['Order_Is_Rush']

            # 1. Dependency Check
            start_limit = 0
            if is_offline:
                start_limit = parse_time_to_mins(offline_settings["start"])
                if order_id in order_target_line_map:
                    t_idx = order_target_line_map[order_id]
                    if 0 <= t_idx < len(line_free_time):
                        jit_start = line_free_time[t_idx] - 2880 - prod_duration
                        start_limit = max(start_limit, jit_start)
            else:
                start_limit = parse_time_to_mins(line_settings[0]["start"])

            min_start_time = start_limit
            dependency_blocked = False
            
            if seq > 1:
                prev_seq = seq - 1
                if (order_id, prev_seq) in order_finish_times:
                    min_start_time = max(min_start_time, order_finish_times[(order_id, prev_seq)])
                else:
                    dependency_blocked = True
            
            if dependency_blocked:
                continue

            # 2. Trial Scheduling
            possible_start = 9999999
            target_line_choice = -1
            setup_cost = 0
            
            if is_offline:
                 # 線外試算
                 off_cat = task['Process_Category']
                 limit = task['Concurrency_Limit']
                 stations = []
                 if limit == 0: pass 
                 else:
                     for k in range(1, limit + 1):
                         stations.append(f"{off_cat}-{k}")
                 
                 stations = stations if stations else [None]
                 found_slot = False
                 t_check = min_start_time
                 for offset in range(0, 1000, 30): 
                     t_probe = t_check + offset
                     if t_probe >= MAX_MINUTES: break
                     if not offline_mask[t_probe]: continue
                     
                     if stations[0] and stations[0] in offline_resource_usage:
                         if offline_resource_usage[stations[0]][t_probe]: continue
                     
                     possible_start = min(possible_start, t_probe)
                     found_slot = True
                     break
                 if not found_slot: continue 
                      
            else:
                # Online Booking
                t_req = task['Target_Line']
                c_lines = []
                
                if t_req > 0: 
                    t_idx = t_req - 4
                    if 0 <= t_idx < total_lines: c_lines = [t_idx]
                else:
                    c_lines = [x for x in range(total_lines)]
                    if str(base_model).startswith("N-DE") and total_lines >= 4: c_lines = [3]
                    elif str(base_model).startswith("N-3610"):
                         if total_lines >= 3: c_lines = [2]
                         else: c_lines = []

                if not str(base_model).startswith("N-3610") and 2 in c_lines:
                    c_lines.remove(2)
                
                if not c_lines: continue

                for l_idx in c_lines:
                    s_time = 0
                    if line_last_model[l_idx] is not None and line_last_model[l_idx] != base_model:
                        s_time = changeover_mins
                    
                    l_free = line_free_time[l_idx]
                    real_start = max(l_free, min_start_time)
                    
                    # 邏輯核心 v6.4
                    if is_rush:
                        # ★ 急單：絕對優先，只看完工時間，無視任何分組偏好
                        score = real_start + s_time
                    else:
                        # ★ 一般單：有條件分組 (Smart Grouping)
                        # 如果是同型號 (s_time == 0)，給予一個 "Group Bonus" (減分)
                        # 這代表：我們願意為了同型號，多等待 60 分鐘。
                        # 如果等待時間超過 60 分鐘，Bonus 就無法抵消延遲，系統就會選擇換線。
                        
                        time_cost = real_start + s_time
                        group_bonus = -60 if s_time == 0 else 0
                        score = time_cost + group_bonus
                    
                    if score < possible_start:
                        possible_start = score
                        target_line_choice = l_idx
                        setup_cost = s_time

            # 3. Best Candidate Selection
            if possible_start < 9999999:
                if best_task_candidate is None or possible_start < best_task_candidate[1]:
                      best_task_candidate = (i, possible_start, task)
                
                # 如果找到一個完美空檔，直接執行 (加速運算)
                if possible_start <= min_start_time + 10 and setup_cost == 0:
                    break
        
        # --- End of Scanning ---
        if best_task_candidate:
            task_idx, score_est, task = best_task_candidate
            pending_tasks.pop(task_idx)
            
            # --- 實際預約資源 (Real Booking) ---
            manpower = int(task['Manpower_Req'])
            total_man_minutes = float(task['Total_Man_Minutes'])
            prod_duration = int(np.ceil(total_man_minutes / manpower)) if manpower > 0 else 0
            is_offline = task['Is_Offline']
            seq = task['Sequence']
            order_id = str(task['Order_ID'])
            base_model = task['Base_Model']

            if is_offline:
                start_limit = parse_time_to_mins(offline_settings["start"])
                if order_id in order_target_line_map:
                    t_idx = order_target_line_map[order_id]
                    if 0 <= t_idx < len(line_free_time):
                        jit_start = line_free_time[t_idx] - 2880 - prod_duration
                        start_limit = max(start_limit, jit_start)
            else:
                start_limit = parse_time_to_mins(line_settings[0]["start"])
            
            min_start_time = start_limit
            if seq > 1:
                prev_seq = seq - 1
                if (order_id, prev_seq) in order_finish_times:
                    min_start_time = max(min_start_time, order_finish_times[(order_id, prev_seq)])

            best_choice = None
            
            if is_offline:
                off_cat = task['Process_Category']
                limit = task['Concurrency_Limit']
                candidate_stations = []
                if limit == 0: pass 
                else:
                    for k in range(1, limit + 1):
                        res_id = f"{off_cat}-{k}"
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
                            max_u = np.max(timeline_manpower[t_search:t_end][i_mask]) if np.any(i_mask) else 0
                            resource_conflict = False
                            if res_usage_mask is not None:
                                if np.any(res_usage_mask[t_search:t_end]): resource_conflict = True
                            if (max_u + manpower <= total_manpower) and (not resource_conflict):
                                if best_choice is None or t_search < best_choice[0]:
                                    best_choice = (t_search, t_end, station_id, 0)
                                found = True
                            else: t_search += 5
                        else: t_search += 5
            else:
                # Online Booking
                t_req = task['Target_Line']
                c_lines = []
                if t_req > 0: 
                    t_idx = t_req - 4
                    if 0 <= t_idx < total_lines: c_lines = [t_idx]
                else:
                    c_lines = [x for x in range(total_lines)]
                    if str(base_model).startswith("N-DE") and total_lines >= 4: c_lines = [3]
                    elif str(base_model).startswith("N-3610"):
                        if total_lines >= 3: c_lines = [2] 
                        else: c_lines = []

                if not str(base_model).startswith("N-3610") and 2 in c_lines:
                    c_lines.remove(2)

                for l_idx in c_lines:
                    curr_mask = line_masks[l_idx]
                    curr_cumsum = line_cumsums[l_idx]
                    s_time = 0
                    if line_last_model[l_idx] is not None and line_last_model[l_idx] != base_model:
                        s_time = changeover_mins
                    
                    t_start_search = max(line_free_time[l_idx], min_start_time)
                    total_need = s_time + prod_duration
                    
                    found = False
                    t_search = t_start_search
                    while not found and t_search < MAX_MINUTES - total_need:
                        if not curr_mask[t_search]:
                            t_search += 1
                            continue
                        s_val = curr_cumsum[t_search]
                        t_val = s_val + total_need
                        if t_val > curr_cumsum[-1]: break
                        t_end = np.searchsorted(curr_cumsum, t_val)
                        if np.any(curr_mask[t_search:t_end]):
                            i_mask = curr_mask[t_search:t_end]
                            max_u = np.max(timeline_manpower[t_search:t_end][i_mask]) if np.any(i_mask) else 0
                            if max_u + manpower <= total_manpower:
                                 if best_choice is None or t_search < best_choice[0]:
                                     best_choice = (t_search, t_end, l_idx, s_time)
                                 found = True
                            else: t_search += 5
                        else: t_search += 5
            
            # Finalize
            if best_choice:
                status_msg = 'OK'
                if is_offline:
                    final_start, final_end, final_station, this_setup = best_choice
                    mask_slice = offline_mask[final_start:final_end]
                    timeline_manpower[final_start:final_end][mask_slice] += manpower
                    if final_station:
                        offline_resource_usage[final_station][final_start:final_end] = True
                        display_line = final_station
                    else: display_line = task['Process_Category']
                else:
                    final_start, final_end, final_line_idx, this_setup = best_choice
                    curr_mask = line_masks[final_line_idx]
                    mask_slice = curr_mask[final_start:final_end]
                    timeline_manpower[final_start:final_end][mask_slice] += manpower
                    line_usage_matrix[final_line_idx, final_start:final_end] = True
                    line_free_time[final_line_idx] = final_end
                    line_last_model[final_line_idx] = base_model
                    display_line = f"Line {final_line_idx+4}"

                if seq > 1 and prev_seq:
                    if (order_id, prev_seq) in order_finish_times:
                        prev_finish = order_finish_times[(order_id, prev_seq)]
                        if (final_start - prev_finish) > 2880: status_msg = "WIP滯留(>2天)"

                order_finish_times[(order_id, seq)] = final_end
                
                results.append({
                    '產線': display_line,
                    '工單': task['Order_ID'], '產品': task['Product_ID'], 
                    '數量': task['Qty'], '類別': '線外' if is_offline else '流水線', 
                    '換線(分)': this_setup,
                    '需求人力': manpower, '預計開始': format_time_str(final_start),
                    '完工時間': format_time_str(final_end), '線佔用(分)': (final_end - final_start), 
                    '狀態': status_msg, '排序用': final_end,
                    '備註': task.get('Remarks', ''),
                    '指定線': task.get('Line_Col', ''),
                    '急單': 'Yes' if task.get('Order_Is_Rush') else ''
                })
            else:
                results.append({'工單': task['Order_ID'], '狀態': '失敗(無資源)', '備註': '找不到空檔'})
        
        else:
            break

    if results:
        last_time = max([r['排序用'] for r in results if r.get('狀態') in ['OK', 'WIP滯留(>2天)']], default=0)
        analyze_days = (last_time // 1440) + 1
    else: last_time, analyze_days = 0, 1
        
    df_idle = analyze_idle_manpower(timeline_manpower, line_masks, total_manpower, last_time + 60)
    df_efficiency = calculate_daily_efficiency(timeline_manpower, line_masks, total_manpower, analyze_days)
    df_utilization = calculate_line_utilization(line_usage_matrix, line_masks, total_lines, analyze_days)
    return pd.DataFrame(results), df_idle, df_efficiency, df_utilization

# ==========================================
# 4. Streamlit 網頁介面設計
# ==========================================

st.set_page_config(page_title="AI 智能排程系統", layout="wide")

st.title(f"🏭 {SYSTEM_VERSION} - 線上排程平台")
st.markdown("上傳 Excel 工單，AI 自動幫您規劃產線與人力配置。")

with st.sidebar:
    st.header("⚙️ 全域參數")
    total_manpower = st.number_input("全廠總人力 (人)", min_value=1, value=50)
    total_lines = st.number_input("產線數量 (條)", min_value=1, value=5)
    changeover_mins = st.number_input("換線時間 (分)", min_value=0, value=30)
    
    st.markdown("---")
    st.header("🕒 各產線工時設定")
    
    line_settings_from_ui = []
    with st.expander("點此展開設定詳細時間", expanded=True):
        for i in range(total_lines):
            st.markdown(f"**Line {i+4}**")
            col1, col2 = st.columns(2)
            with col1:
                t_start = st.time_input(f"L{i+4} 開始", value=time(8, 0), key=f"start_{i}")
            with col2:
                t_end = st.time_input(f"L{i+4} 結束", value=time(17, 0), key=f"end_{i}")
            
            line_settings_from_ui.append({
                "start": t_start.strftime("%H:%M"), 
                "end": t_end.strftime("%H:%M")
            })
    
    st.markdown("---")
    st.markdown("**線外專區 (Offline)**")
    col1, col2 = st.columns(2)
    with col1:
        off_start = st.time_input("線外 開始", value=time(8, 0), key="off_start")
    with col2:
        off_end = st.time_input("線外 結束", value=time(17, 0), key="off_end")
    
    offline_settings_from_ui = {
        "start": off_start.strftime("%H:%M"),
        "end": off_end.strftime("%H:%M")
    }

    st.markdown("---")
    st.info("💡 邏輯說明：\n1. 流水線為 Line4 ~ Line8。\n2. N-DE* 產品優先排入 Line 7。\n3. **Line 6 僅限 N-3610* 產品使用** (其他產品不能用)。")

uploaded_file = st.file_uploader("📂 請上傳工單 Excel 檔案", type=["xlsx", "xls"])

if uploaded_file is not None:
    df_clean, err = load_and_clean_data(uploaded_file)
    
    if err:
        st.error(f"讀取失敗: {err}")
    else:
        st.success(f"讀取成功！共 {len(df_clean)} 筆有效工單。")
        with st.expander("查看原始資料預覽"):
            st.dataframe(df_clean.head())
            
        if st.button("🚀 開始 AI 排程運算", type="primary"):
            with st.spinner('正在進行百萬次模擬運算 (包含產能與工序檢查)...請稍候...'):
                df_schedule, df_idle, df_efficiency, df_utilization = run_scheduler(
                    df_clean, 
                    total_manpower, 
                    total_lines, 
                    changeover_mins, 
                    line_settings_from_ui,
                    offline_settings_from_ui
                )
                
                st.success("✅ 排程運算完成！")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_schedule.to_excel(writer, sheet_name='生產排程', index=False)
                    df_efficiency.to_excel(writer, sheet_name='每日效率分析', index=False)
                    df_utilization.to_excel(writer, sheet_name='各線稼動率', index=False)
                    df_idle.to_excel(writer, sheet_name='閒置人力明細', index=False)
                output.seek(0)
                
                st.download_button(
                    label="📥 下載完整排程報表 (Excel)",
                    data=output,
                    file_name=f'AI_Schedule_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
                tab1, tab2, tab3 = st.tabs(["📊 生產排程表", "📈 效率分析", "⚠️ 閒置人力"])
                
                with tab1:
                    st.dataframe(df_schedule, use_container_width=True)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("每日效率")
                        st.dataframe(df_efficiency)
                    with col2:
                        st.subheader("產線稼動率")
                        st.dataframe(df_utilization)
                        
                with tab3:
                    st.dataframe(df_idle, use_container_width=True)

else:
    st.info("👈 請從左側開始設定參數，再上傳檔案。")
