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
SYSTEM_VERSION = "v5.9 (Feature: Prevent Line Starvation)"

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

# 指定線提取函數
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

    # 預判目標產線
    df_online_parts = df[df['Is_Offline'] == False]
    order_target_line_map = {}
    for _, row in df_online_parts.iterrows():
        t_line = row['Target_Line']
        if t_line > 0:
            target_idx = t_line - 4
        elif str(row['Base_Model']).startswith("N-DE") and total_lines >= 4:
            target_idx = 3 
        else:
            target_idx = 0 
        if row['Order_ID'] not in order_target_line_map:
            order_target_line_map[row['Order_ID']] = target_idx

    # 全局排序：依然保持急單 > 產品 > 順序的邏輯
    rush_orders_global = df[df['Is_Rush']]['Order_ID'].unique()
    df['Order_Is_Rush'] = df['Order_ID'].isin(rush_orders_global)
    
    df_sorted = df.sort_values(
        by=['Order_Is_Rush', 'Base_Model', 'Order_ID', 'Sequence', 'Priority'], 
        ascending=[False, True, True, True, True]
    )
    
    # 轉換為 List of Dict 以便於動態移除/插入
    pending_tasks = df_sorted.to_dict('records')
    line_last_model = {i: None for i in range(total_lines)}
    
    # ★★★ 修正核心：動態填補邏輯 (Non-Blocking Loop) ★★★
    # 我們不再簡單地 for loop，而是使用 while loop 來掃描 pending_tasks
    # 每次尋找「目前最早能開始」的任務來排，而不是死板地照順序排
    # 這樣如果 Task A (Seq 2) 被卡住，Task B (Seq 1) 可以先補上
    
    # 為了簡化複雜度且避免無限迴圈，我們採取「多輪掃描」策略：
    # 1. 嘗試依序排程
    # 2. 如果某個任務因為 "Dependency (Seq > 1)" 被推遲太久 (例如 > 2天後)，我們先跳過它，看下一個任務
    # 3. 下一個任務如果是不同產品，雖然要換線，但只要能更早開始，就排入
    
    # 但考慮到 Streamlit 的執行時限，我們採用一個更簡單的「貪婪策略」：
    # 維護一個 "Ready Queue"，只有當前置工序完成的任務才進入 Queue
    # 但這樣會破壞 "同一產品連續生產" 的偏好。
    
    # 折衷方案：
    # 維持目前的順序，但在尋找 "best_choice" 時，如果發現最佳時間點離現在太遠 (例如 > 24小時空窗)，
    # 且該產線在這段空窗期是閒置的，我們就視為「無效排程」，暫時把這個任務丟回 Pool，
    # 讓後面的任務先嘗試填補這個空窗。
    
    # 實作：
    # 我們不重寫整個 while loop，而是在尋找 time slot 時，
    # 如果是 Online 任務，且 t_search (受限於 min_start_time) 遠大於 line_free_time (產線空閒時間)，
    # 這代表產線在空轉等待。這是不允許的。
    # 我們應該把這個任務 "延後處理" (Put back to end of queue? No, maybe next batch).
    
    # 更好的方式：
    # 1. 將所有任務標記狀態。
    # 2. 每次從 pending_tasks 中挑選「能最早開始」的任務。
    
    # 鑑於 python 效能，我們採用 "有限次的延後"：
    # 如果 row 因為 dependency 被卡住，我們把它 swap 到後面去？
    
    # 不，最直接的方式是：
    # 在計算 min_start_time 時，如果發現 min_start_time > line_free_time[line_idx] + threshold (e.g. 60 mins)
    # 我們就跳過這個候選產線？不，如果所有產線都這樣呢？
    
    # 讓我們回到 "排序" 解決問題：
    # 如果 Order A-2 被卡住，是因為 A-1 還沒做完。
    # 這時候我們應該先排 Order B-1 (如果 B-1 已經 ready)。
    # 原本的排序是：A-1, A-2, B-1, B-2 (假設同 Model)。
    # A-1 排了。 A-2 卡住 (等 A-1)。
    # 此時 B-1 其實可以先排！
    # 但因為我們是循序迴圈，A-2 佔據了迴圈位置。
    
    # === 最終方案：多回合排程 (Multi-Pass Scheduling) ===
    # 建立一個 waiting_list。
    # 遍歷 df_sorted。
    # 如果一個任務能 "緊接著" 排入 (start_time ~= line_free_time)，就排。
    # 如果不能 (需要等待前置)，且等待時間過長，就丟入 waiting_list。
    # 當正常隊列處理完，或產線有空檔時，再回頭看 waiting_list。
    # 這樣可以讓 B-1 插隊到 A-2 前面。
    
    scheduled_tasks = []
    deferred_tasks = [] # 暫存因相依性而卡住的任務
    
    # 第一輪：盡量排
    # 為了實作簡單，我們將邏輯封裝一下
    
    def attempt_schedule(row, is_deferred_retry=False):
        manpower = int(row['Manpower_Req'])
        total_man_minutes = float(row['Total_Man_Minutes'])
        prod_duration = int(np.ceil(total_man_minutes / manpower)) if manpower > 0 else 0
        
        is_offline = row['Is_Offline']
        seq = row['Sequence']
        order_id = str(row['Order_ID'])
        base_model = row['Base_Model']

        # 1. 計算最早可開始時間 (Dependency Check)
        if is_offline:
            start_limit = parse_time_to_mins(offline_settings["start"])
            if order_id in order_target_line_map:
                target_line_idx = order_target_line_map[order_id]
                line_ready_time = line_free_time[target_line_idx]
                jit_start = line_ready_time - 2880 - prod_duration
                start_limit = max(start_limit, jit_start)
        else:
            start_limit = parse_time_to_mins(line_settings[0]["start"])
            
        min_start_time = start_limit
        dependency_met = True
        
        if seq > 1:
            prev_seq = seq - 1
            if (order_id, prev_seq) in order_finish_times:
                min_start_time = max(min_start_time, order_finish_times[(order_id, prev_seq)])
            else:
                # 前置未完成，且不是第一輪 (第一輪我們會跳過放到 deferred)
                # 如果是 deferred retry，代表真的沒救了，只能硬排 (但通常上一輪會排完前置)
                dependency_met = False

        # 如果是線上任務，且前置已完成，但需要等待很久 (產線空轉)，我們也視為 "Dependency Not Ideal"
        # 除非是 deferred retry (補考)，那就不能挑了
        
        best_choice = None 

        if is_offline:
            offline_category = row['Process_Category']
            concurrency_limit = row['Concurrency_Limit']
            candidate_stations = []
            if concurrency_limit == 0: pass 
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
                            if np.any(res_usage_mask[t_search:t_end]): resource_conflict = True
                        if (current_max_used + manpower <= total_manpower) and (not resource_conflict):
                            if best_choice is None or t_search < best_choice[0]:
                                best_choice = (t_search, t_end, station_id, 0)
                            found = True
                        else: t_search += 5
                    else: t_search += 5
        else:
            # Online
            target_line_req = row['Target_Line']
            candidate_lines = []
            if target_line_req > 0:
                t_idx = target_line_req - 4
                if 0 <= t_idx < total_lines: candidate_lines = [t_idx]
            else:
                candidate_lines = [i for i in range(total_lines)]
                if str(base_model).startswith("N-DE"):
                    if total_lines >= 4: candidate_lines = [3]
            
            is_n3610 = str(base_model).startswith("N-3610")
            if not is_n3610:
                if 0 in candidate_lines: candidate_lines.remove(0)
            if not candidate_lines: candidate_lines = [i for i in range(1, total_lines)] if total_lines > 1 else []

            for line_idx in candidate_lines:
                curr_mask = line_masks[line_idx]
                curr_cumsum = line_cumsums[line_idx]
                setup_time = 0
                if line_last_model[line_idx] is not None and line_last_model[line_idx] != base_model:
                    setup_time = changeover_mins
                
                t_start_search = max(line_free_time[line_idx], min_start_time)
                
                # ★★★ 關鍵檢查：是否會造成產線閒置？ ★★★
                # 如果該產線目前是空的 (free_time < t_start_search)，但我們因為 dependency 必須等到 t_start_search
                # 這中間的空檔 (gap) 如果太大，我們就應該先跳過這張單，讓別的單來填
                # 只有在非 deferred retry 模式下才檢查
                if not is_deferred_retry and not is_offline:
                    gap = t_start_search - line_free_time[line_idx]
                    # 如果空轉超過 60 分鐘，且這不是因為還沒開工 (free_time=start)，則跳過
                    if gap > 60 and line_free_time[line_idx] > parse_time_to_mins(line_settings[0]["start"]):
                        continue 

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
                    t_end = np.searchsorted(curr_cumsum, t_val)
                    if np.any(curr_mask[t_search:t_end]):
                        i_mask = curr_mask[t_search:t_end]
                        max_u = np.max(timeline_manpower[t_search:t_end][i_mask]) if np.any(i_mask) else 0
                        if max_u + manpower <= total_manpower:
                             if best_choice is None or t_search < best_choice[0]:
                                 best_choice = (t_search, t_end, line_idx, setup_time)
                             found = True
                        else: t_search += 5
                    else: t_search += 5

        # --- 判斷是否成功排入 ---
        if best_choice:
            # 成功排入
            if is_offline:
                final_start, final_end, final_station, this_setup = best_choice
                mask_slice = offline_mask[final_start:final_end]
                timeline_manpower[final_start:final_end][mask_slice] += manpower
                if final_station:
                    offline_resource_usage[final_station][final_start:final_end] = True
                    display_line = final_station
                else: display_line = row['Process_Category']
            else:
                final_start, final_end, final_line_idx, this_setup = best_choice
                curr_mask = line_masks[final_line_idx]
                mask_slice = curr_mask[final_start:final_end]
                timeline_manpower[final_start:final_end][mask_slice] += manpower
                line_usage_matrix[final_line_idx, final_start:final_end] = True
                line_free_time[final_line_idx] = final_end
                line_last_model[final_line_idx] = base_model
                display_line = f"Line {final_line_idx+4}"

            status_msg = 'OK'
            if seq > 1 and prev_seq:
                if (order_id, prev_seq) in order_finish_times:
                    prev_finish = order_finish_times[(order_id, prev_seq)]
                    if (final_start - prev_finish) > 2880: status_msg = "WIP滯留(>2天)"

            order_finish_times[(str(row['Order_ID']), row['Sequence'])] = final_end
            
            results.append({
                '產線': display_line,
                '工單': row['Order_ID'], '產品': row['Product_ID'], 
                '數量': row['Qty'], '類別': '線外' if is_offline else '流水線', 
                '換線(分)': this_setup,
                '需求人力': manpower, '預計開始': format_time_str(final_start),
                '完工時間': format_time_str(final_end), '線佔用(分)': (final_end - final_start), 
                '狀態': status_msg, '排序用': final_end,
                '備註': row.get('Remarks', ''),
                '指定線': row.get('Line_Col', ''),
                '急單': 'Yes' if row.get('Order_Is_Rush') else ''
            })
            return True # Scheduled
        else:
            return False # Not scheduled (defer)

    # === Main Loop for Tasks ===
    # 用一個 while loop 處理 pending_tasks，直到沒有任務可以排入為止
    # 為了避免無限迴圈，我們最多 retry 3 次 (rounds)
    
    # 第一次遍歷：嘗試依序排
    for idx, row in df_sorted.iterrows():
        success = attempt_schedule(row, is_deferred_retry=False)
        if not success:
            deferred_tasks.append(row)
            
    # 第二次遍歷：處理被延後的任務 (Retry)
    # 這些任務之前因為會造成產線空轉而被跳過，現在不得不排了 (或者空缺已被填補)
    if deferred_tasks:
        for row in deferred_tasks:
            attempt_schedule(row, is_deferred_retry=True) # 強制排入，不再檢查空轉

    # ... (後續輸出邏輯不變) ...

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
    st.info("💡 邏輯說明：\n1. 流水線為 Line4 ~ Line8。\n2. N-DE* 產品優先排入 Line 7。\n3. Line 4 僅限 N-3610* 產品使用。")

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
