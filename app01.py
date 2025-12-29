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
SYSTEM_VERSION = "v5.6.2 (Rush Order Priority Enhanced)"

# 線外製程分類對照表
OFFLINE_MAPPING = {
    "超音波熔接": "線外-超音波熔接",
    "LS": "線外-組裝前LS",
    "PT": "線外-PT",
    "裝配前組裝(PKM)": "線外-線邊組裝"
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
