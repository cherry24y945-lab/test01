import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import io
import math
import re

# ==========================================
# 1. 核心邏輯區
# ==========================================
SYSTEM_VERSION = "v5.6.7 (Offline Capacity Constraints)"

# 線外製程分類與資源限制設定
# value: (顯示名稱, 最大並行工單數)
# 0 表示無工單數限制（只受人力限制），1 表示單一工單，2 表示雙工單並行
OFFLINE_CONFIG = {
    "超音波熔接": ("線外-超音波熔接", 1),
    "LS": ("線外-組裝前LS", 2),
    "雷射": ("線外-組裝前LS", 2), # 兼容舊稱
    "PT": ("線外-PT", 1),
    "PKM": ("線外-線邊組裝", 2),
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
            if available_mins > 0:
                util_rate = (busy_mins / available_mins) * 100
                row[f'Line {i+1} (%)'] = round(util_rate, 1)
            else:
                row[f'Line {i+1} (%)'] = "-"
        if any(v != "-" for k, v in row.items() if k != '日期'):
            utilization_records.append(row)
    return pd.DataFrame(utilization_records)

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
        
        # 線外分類與資源標記
        def categorize_offline(val):
            val_str = str(val)
            for kw, (name, limit) in OFFLINE_CONFIG.items():
                if kw in val_str:
                    return name, limit
            return "Online", -1
        
        # 拆分出兩個欄位：類別名稱、並行限制
        temp_res = df['Process_Type'].apply(categorize_offline)
        df['Process_Category'] = temp_res.apply(lambda x: x[0])
        df['Concurrency_Limit'] = temp_res.apply(lambda x: x[1])
        df['Is_Offline'] = df['Process_Category'] != "Online"

        # 急單優先權
        if 'Rush_Col' in df.columns:
            df['Is_Rush'] = df['Rush_Col'].astype(str).str.contains('急單', na=False)
        else:
            df['Is_Rush'] = df['Remarks'].astype(str).str.contains('急單', na=False)

        # 指定線判斷
        def extract_line_num(val):
            val_str = str(val).upper().replace(' ', '')
            match = re.search(r'LINE(\d+)', val_str)
            if match:
                try: return int(match.group(1))
                except: return 0
            return 0

        if 'Line_Col' in df.columns:
            df['Target_Line'] = df['Line_Col'].apply(extract_line_num)
        else:
            df['Target_Line'] = df['Remarks'].apply(extract_line_num)
        
        # 補強：若指定線無效，再看備註
        mask_no_line = df['Target_Line'] == 0
        df.loc[mask_no_line, 'Target_Line'] = df.loc[mask_no_line, 'Remarks'].apply(extract_line_num)

        # 工序順序
        def get_sequence(val):
            try:
                match = re.search(r'\d+', str(val))
                if match: return int(match.group())
                return 0 
            except: return 0
        df['Sequence'] = df['Remarks'].apply(get_sequence)
        
        # 強制補齊缺失欄位以防報錯
        if 'Rush_Col' not in df.columns: df['Rush_Col'] = ''
        if 'Line_Col' not in df.columns: df['Line_Col'] = ''

        return df, None
    except Exception as e:
        return None, str(e)

# 排程核心
def run_scheduler(df, total_manpower, total_lines, changeover_mins, line_settings):
    MAX_MINUTES = 14 * 24 * 60 
    
    line_masks = []
    line_cumsums = []
    for setting in line_settings:
        m = create_line_mask(setting["start"], setting["end"], 14)
        line_masks.append(m)
        line_cumsums.append(np.cumsum(m))
        
    offline_mask = line_masks[0]
    offline_cumsum = line_cumsums[0]

    timeline_manpower = np.zeros(MAX_MINUTES, dtype=int)
    line_usage_matrix = np.zeros((total_lines, MAX_MINUTES), dtype=bool)
    results = []
    line_free_time = [parse_time_to_mins(setting["start"]) for setting in line_settings]
    
    # ★★★ 新增：線外資源佔用表 (Resource ID -> Boolean Array) ★★★
    # 用於控制超音波(1台)與LS(2台)的佔用
    # Key: "線外-超音波熔接-1", "線外-組裝前LS-1", "線外-組裝前LS-2"
    offline_resource_usage = {}
    
    order_finish_times = {}

    # --- Phase 1: 流水線 (Online) ---
    df_online = df[df['Is_Offline'] == False].copy()
    family_groups = df_online.groupby('Base_Model')
    
    batches = []
    for base_model, group_df in family_groups:
        is_rush = group_df['Is_Rush'].any() 
        rush_weight = 1000000 if is_rush else 0
        total_work_load = (group_df['Manpower_Req'] * group_df['Total_Man_Minutes']).sum()
        
        target_lines = group_df['Target_Line'].unique()
        specific_requests = [t for t in target_lines if t > 0]
        if specific_requests:
            valid_reqs = [t-1 for t in specific_requests if t <= total_lines]
            candidate_lines = valid_reqs if valid_reqs else [i for i in range(total_lines)]
        else:
            candidate_lines = [i for i in range(total_lines)]

        sorted_df = group_df.sort_values(by=['Is_Rush', 'Priority'], ascending=[False, True])

        batches.append({
            'base_model': base_model,
            'df': sorted_df,
            'is_rush': is_rush,
            'weight': rush_weight + total_work_load, 
            'candidate_lines': candidate_lines
        })
    
    batches.sort(key=lambda x: (x['is_rush'], x['weight']), reverse=True)
    
    for batch_idx, batch in enumerate(batches):
        candidate_lines = batch['candidate_lines']
        batch_df = batch['df']
        best_line_choice = None 
        
        for line_idx in candidate_lines:
            curr_mask = line_masks[line_idx]
            curr_cumsum = line_cumsums[line_idx]
            t_search = line_free_time[line_idx]
            
            first_row = batch_df.iloc[0]
            first_manpower = int(first_row['Manpower_Req'])
            first_duration = int(np.ceil(first_row['Total_Man_Minutes'] / first_manpower))
            setup_time = changeover_mins if t_search > 480 else 0
            
            total_need = setup_time + first_duration
            found = False
            start_t = -1
            
            temp_search = t_search
            while not found and temp_search < MAX_MINUTES - total_need:
                if not curr_mask[temp_search]:
                    temp_search += 1
                    continue
                
                s_val = curr_cumsum[temp_search]
                t_val = s_val + total_need
                if t_val > curr_cumsum[-1]: break
                t_end = np.searchsorted(curr_cumsum, t_val)
                
                i_mask = curr_mask[temp_search:t_end]
                max_u = np.max(timeline_manpower[temp_search:t_end][i_mask]) if np.any(i_mask) else 0
                
                if max_u + first_manpower <= total_manpower:
                    start_t = temp_search
                    found = True
                else:
                    temp_search += 5
            
            if found:
                score = start_t
                if best_line_choice is None or score < best_line_choice[0]:
                    best_line_choice = (score, line_idx, start_t, setup_time)
                    
        if best_line_choice:
            _, target_line_idx, batch_start_time, initial_setup = best_line_choice
            current_t = batch_start_time
            
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                manpower = int(row['Manpower_Req'])
                total_man_minutes = float(row['Total_Man_Minutes'])
                prod_duration = int(np.ceil(total_man_minutes / manpower)) if manpower > 0 else 0
                this_setup = initial_setup if i == 0 else 0
                
                curr_mask = line_masks[target_line_idx]
                curr_cumsum = line_cumsums[target_line_idx]
                total_work = this_setup + prod_duration
                found_slot = False
                
                seq = row['Sequence']
                order_id = str(row['Order_ID'])
                min_start_from_dep = 0
                if seq > 1:
                    prev_seq = seq - 1
                    if (order_id, prev_seq) in order_finish_times:
                        min_start_from_dep = order_finish_times[(order_id, prev_seq)]

                t_scan = max(current_t, line_free_time[target_line_idx], min_start_from_dep)
                real_start, real_end = -1, -1
                
                while not found_slot and t_scan < MAX_MINUTES - total_work:
                    if not curr_mask[t_scan]:
                        t_scan += 1
                        continue
                    
                    s_val = curr_cumsum[t_scan]
                    t_val = s_val + total_work
                    if t_val > curr_cumsum[-1]: break
                    t_end = np.searchsorted(curr_cumsum, t_val)
                    
                    i_mask = curr_mask[t_scan:t_end]
                    max_u = np.max(timeline_manpower[t_scan:t_end][i_mask]) if np.any(i_mask) else 0
                    
                    if max_u + manpower <= total_manpower:
                        real_start, real_end, found_slot = t_scan, t_end, True
                    else:
                        t_scan += 5
                
                if found_slot:
                    mask_slice = curr_mask[real_start:real_end]
                    timeline_manpower[real_start:real_end][mask_slice] += manpower
                    line_usage_matrix[target_line_idx, real_start:real_end] = True
                    current_t = real_end
                    line_free_time[target_line_idx] = real_end 
                    
                    order_finish_times[(str(row['Order_ID']), row['Sequence'])] = real_end

                    results.append({
                        '產線': f"Line {target_line_idx+1}", 
                        '工單': row['Order_ID'], '產品': row['Product_ID'], 
                        '數量': row['Qty'], '類別': '流水線', '換線(分)': this_setup,
                        '需求人力': manpower, '預計開始': format_time_str(real_start),
                        '完工時間': format_time_str(real_end), '線佔用(分)': prod_duration, '狀態': 'OK', '排序用': real_end,
                        '備註': row.get('Remarks', ''), 
                        '指定線': row.get('Line_Col', ''), 
                        '急單': 'Yes' if row.get('Is_Rush') else ''
                    })
                else:
                    results.append({'工單': row['Order_ID'], '狀態': '失敗(資源不足)', '產線': f"Line {target_line_idx+1}"})

    # --- Phase 2: 線外工單 (Offline) ---
    df_offline = df[df['Is_Offline'] == True].copy()
    df_offline = df_offline.sort_values(by=['Is_Rush', 'Priority'], ascending=[False, True])
    
    curr_mask = offline_mask
    curr_cumsum = offline_cumsum

    for _, row in df_offline.iterrows():
        manpower = int(row['Manpower_Req'])
        total_man_minutes = float(row['Total_Man_Minutes'])
        prod_duration = int(np.ceil(total_man_minutes / manpower)) if manpower > 0 else 0
        
        offline_category = row['Process_Category']
        concurrency_limit = row['Concurrency_Limit']
        
        # 決定要使用的資源站點 (Stations)
        # 如果是 0 (無限)，就不用檢查資源佔用，只檢查人力
        # 如果是 1 (超音波)，只有一個站點 "線外-超音波熔接-1"
        # 如果是 2 (LS)，有兩個站點 "線外-組裝前LS-1", "線外-組裝前LS-2"
        candidate_stations = []
        if concurrency_limit == 0:
            pass # 無須資源檢查
        else:
            for i in range(1, concurrency_limit + 1):
                res_id = f"{offline_category}-{i}"
                if res_id not in offline_resource_usage:
                    offline_resource_usage[res_id] = np.zeros(MAX_MINUTES, dtype=bool)
                candidate_stations.append(res_id)

        if manpower > total_manpower:
             results.append({'工單': row['Order_ID'], '狀態': '失敗(人力不足)', '產線': offline_category})
             continue
        
        # Dependency Check
        seq = row['Sequence']
        order_id = str(row['Order_ID'])
        min_start_time = 480 
        if seq > 1:
            prev_seq = seq - 1
            if (order_id, prev_seq) in order_finish_times:
                min_start_time = order_finish_times[(order_id, prev_seq)]
        
        # 尋找最佳站點與時間
        best_choice = None # (start_time, end_time, station_id)

        # 若有限制資源，嘗試所有站點，找最早能塞進去的
        stations_to_try = candidate_stations if candidate_stations else [None]
        
        for station_id in stations_to_try:
            # 決定該站點的佔用時間軸
            if station_id:
                res_usage_mask = offline_resource_usage[station_id]
            
            found = False
            t_search = max(480, min_start_time)
            
            while not found and t_search < MAX_MINUTES - prod_duration:
                if not curr_mask[t_search]:
                    t_search += 1
                    continue
                
                # 檢查資源是否被佔用 (若有 station_id)
                if station_id:
                    # 快速檢查區間內是否有 True
                    # 注意：這裡要預判結束時間，但結束時間取決於 mask 的有效工時
                    # 簡化：先算結束時間，再檢查這段區間資源有沒有被用掉
                    # 這種方式比較準確
                    pass

                s_val = curr_cumsum[t_search]
                t_val = s_val + prod_duration
                if t_val > curr_cumsum[-1]: break
                t_end = np.searchsorted(curr_cumsum, t_val)
                
                # 檢查人力
                i_mask = curr_mask[t_search:t_end]
                current_max_used = np.max(timeline_manpower[t_search:t_end][i_mask]) if np.any(i_mask) else 0
                
                # 檢查資源佔用 (Resource Availability Check)
                resource_conflict = False
                if station_id:
                    if np.any(res_usage_mask[t_search:t_end]):
                        resource_conflict = True
                
                if (current_max_used + manpower <= total_manpower) and (not resource_conflict):
                    # 找到空檔了！
                    # 如果這是第一個找到的，或是比之前的更早，就選它
                    if best_choice is None or t_search < best_choice[0]:
                        best_choice = (t_search, t_end, station_id)
                    found = True # 針對這個 station 已經找到最早的了，不用往後找
                else:
                    t_search += 5 
        
        # 確定排入
        if best_choice:
            final_start, final_end, final_station = best_choice
            
            # 更新人力
            mask_slice = curr_mask[final_start:final_end]
            timeline_manpower[final_start:final_end][mask_slice] += manpower
            
            # 更新資源
            if final_station:
                offline_resource_usage[final_station][final_start:final_end] = True
                display_line_name = final_station # 顯示例如 "線外-組裝前LS-1"
            else:
                display_line_name = offline_category # 沒限制就顯示原名

            order_finish_times[(str(row['Order_ID']), row['Sequence'])] = final_end

            results.append({
                '產線': display_line_name,
                '工單': row['Order_ID'], '產品': row['Product_ID'], 
                '數量': row['Qty'], '類別': '線外', '換線(分)': 0,
                '需求人力': manpower, '預計開始': format_time_str(final_start),
                '完工時間': format_time_str(final_end), '線佔用(分)': prod_duration, '狀態': 'OK', '排序用': final_end,
                '備註': row.get('Remarks', ''),
                '指定線': row.get('Line_Col', ''),
                '急單': 'Yes' if row.get('Is_Rush') else ''
            })
        else:
             results.append({'工單': row['Order_ID'], '狀態': '失敗(資源或人力不足)', '產線': offline_line_name})


    if results:
        last_time = max([r['排序用'] for r in results if r.get('狀態')=='OK'], default=0)
        analyze_days = (last_time // 1440) + 1
    else: last_time, analyze_days = 0, 1
        
    df_idle = analyze_idle_manpower(timeline_manpower, line_masks, total_manpower, last_time + 60)
    df_efficiency = calculate_daily_efficiency(timeline_manpower, line_masks, total_manpower, analyze_days)
    df_utilization = calculate_line_utilization(line_usage_matrix, line_masks, total_lines, analyze_days)
    return pd.DataFrame(results), df_idle, df_efficiency, df_utilization

# ==========================================
# 2. Streamlit 網頁介面設計
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
            st.markdown(f"**Line {i+1}**")
            col1, col2 = st.columns(2)
            with col1:
                t_start = st.time_input(f"L{i+1} 開始", value=time(8, 0), key=f"start_{i}")
            with col2:
                t_end = st.time_input(f"L{i+1} 結束", value=time(17, 0), key=f"end_{i}")
            
            line_settings_from_ui.append({
                "start": t_start.strftime("%H:%M"), 
                "end": t_end.strftime("%H:%M")
            })

    st.markdown("---")
    st.info("💡 邏輯說明：\n1. 線外製程分為：組裝前LS、超音波熔接、線邊組裝、PT。\n2. 備註欄數字 (1, 2) 代表工序，系統會確保順序生產。\n3. 優先讀取[急單]與[指定線]獨立欄位，若無則讀取[備註]。")

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
                    line_settings_from_ui
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
