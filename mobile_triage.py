import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Page Config for Mobile
st.set_page_config(
    page_title="Issue Triage Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-first CSS
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background: #0f172a;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main {
        padding: 0;
        max-width: 100%;
    }
    
    /* Header Styles */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px;
        border-radius: 0;
        margin: 0;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .header-container h1 {
        font-size: 24px;
        margin: 0 0 4px 0;
        font-weight: 700;
    }
    
    .header-container p {
        font-size: 13px;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        padding: 12px;
        background: #1e293b;
    }
    
    .stat-card {
        background: #334155;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .stat-card .number {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-card .label {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Tabs */
    .tabs-container {
        display: flex;
        gap: 8px;
        padding: 12px;
        background: #1e293b;
        overflow-x: auto;
        border-bottom: 1px solid #334155;
    }
    
    .tab-btn {
        background: #334155;
        color: #94a3b8;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.3s ease;
    }
    
    .tab-btn.active {
        background: #667eea;
        color: white;
    }
    
    /* Issue Items */
    .issue-item {
        background: #334155;
        margin: 8px 12px;
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .issue-item:active {
        transform: scale(0.98);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .issue-item .title {
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 4px;
    }
    
    .issue-item .metrics {
        display: flex;
        gap: 12px;
        font-size: 12px;
    }
    
    .issue-item .metric {
        background: #1e293b;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .priority-high { border-left-color: #ef4444; }
    .priority-medium { border-left-color: #f59e0b; }
    .priority-low { border-left-color: #22c55e; }
    
    /* Sliders */
    .slider-container {
        padding: 12px;
        background: #1e293b;
    }
    
    .slider-label {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin-bottom: 8px;
        color: #94a3b8;
    }
    
    /* Buttons */
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    
    .btn-primary:active {
        transform: scale(0.95);
    }
    
    /* Matrix Grid */
    .matrix-container {
        padding: 12px;
        background: #0f172a;
    }
    
    .matrix-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        aspect-ratio: 1;
    }
    
    .quadrant {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 12px;
        display: flex;
        flex-direction: column;
    }
    
    .quadrant-header {
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #334155;
    }
    
    .quadrant-items {
        flex: 1;
        overflow-y: auto;
        font-size: 11px;
    }
    
    .quadrant-item {
        background: #334155;
        padding: 6px;
        margin-bottom: 4px;
        border-radius: 4px;
    }
    
    /* Bottom Navigation */
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #1e293b;
        border-top: 1px solid #334155;
        display: flex;
        justify-content: space-around;
        padding: 8px 0;
        z-index: 100;
    }
    
    .nav-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        font-size: 11px;
        color: #94a3b8;
        cursor: pointer;
        flex: 1;
        padding: 8px;
    }
    
    .nav-item.active {
        color: #667eea;
    }
    
    .nav-item-icon {
        font-size: 20px;
    }
    
    /* FAB - Floating Action Button */
    .fab {
        position: fixed;
        bottom: 80px;
        right: 16px;
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        z-index: 99;
    }
    
    .fab:active {
        transform: scale(0.9);
    }
    
    /* Content with bottom padding */
    .main-content {
        padding-bottom: 80px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'issues' not in st.session_state:
    st.session_state.issues = [
        {"id": 1, "title": "Improve Audio Quality", "description": "Remove noise and crackle", "effort": 7, "impact": 9, "status": "🟡 In Review", "tags": ["audio", "quality"]},
        {"id": 2, "title": "Support Long Videos", "description": "Handle 30+ minute videos", "effort": 8, "impact": 8, "status": "🔵 Planned", "tags": ["feature"]},
        {"id": 3, "title": "African Languages Support", "description": "Add Wolof, Fulani, Tamasheq", "effort": 9, "impact": 8, "status": "🔵 Planned", "tags": ["multilingual"]},
        {"id": 4, "title": "Processing Speed", "description": "Use GPU acceleration", "effort": 6, "impact": 8, "status": "🟡 In Review", "tags": ["performance"]},
        {"id": 5, "title": "On-Screen Translation", "description": "Display text overlay", "effort": 5, "impact": 7, "status": "🔵 Planned", "tags": ["ui"]},
        {"id": 6, "title": "Export SRT Files", "description": "Export subtitle files", "effort": 3, "impact": 6, "status": "🟢 Easy", "tags": ["export"]},
        {"id": 7, "title": "Project History", "description": "Save project history", "effort": 4, "impact": 5, "status": "🔵 Planned", "tags": ["storage"]},
        {"id": 8, "title": "Dark Mode UI", "description": "Add dark theme", "effort": 2, "impact": 4, "status": "🔵 Planned", "tags": ["ui"]},
    ]

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Matrix"

if 'selected_issue' not in st.session_state:
    st.session_state.selected_issue = None

# Helper Functions
def calculate_priority(effort, impact):
    """Calculate priority score (impact * (11 - effort))"""
    return int(impact * (11 - effort))

def get_priority_color(priority):
    if priority >= 75: return "🔴"
    if priority >= 50: return "🟠"
    return "🟢"

def get_priority_level(effort, impact):
    priority = calculate_priority(effort, impact)
    if effort <= 5 and impact >= 7:
        return "quick-win", "✅ Quick Win"
    if effort >= 7 and impact >= 8:
        return "major-feature", "🔥 Major"
    if effort <= 3:
        return "nice-to-have", "💡 Easy"
    return "medium", "⚙️ Medium"

# Header
st.markdown("""
<div class="header-container">
    <h1>📊 Issue Triage Pro</h1>
    <p>Bamanankan Video Dubber — Priority Management</p>
</div>
""", unsafe_allow_html=True)

# Stats
df_issues = pd.DataFrame(st.session_state.issues)
df_issues['priority'] = df_issues.apply(lambda row: calculate_priority(row['effort'], row['impact']), axis=1)
total_issues = len(df_issues)
total_effort = df_issues['effort'].sum()
avg_priority = df_issues['priority'].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📋 Total", total_issues)
with col2:
    st.metric("⚡ Effort", f"{total_effort}h")
with col3:
    st.metric("⭐ Avg Impact", f"{df_issues['impact'].mean():.1f}")
with col4:
    st.metric("🎯 Avg Priority", f"{avg_priority:.0f}")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Matrix", "🏆 Ranked", "📝 Add Issue", "📄 Details"])

# TAB 1: PRIORITY MATRIX
with tab1:
    st.subheader("Priority Matrix - Drag & Drop")
    st.write("*Quadrants: High Impact+Low Effort (top-left) → Quick Wins → Major Features → Avoid (bottom-right)*")
    
    # Create 2x2 matrix
    col1, col2 = st.columns(2)
    
    # Top-Left: Quick Wins (High Impact, Low Effort)
    with col1:
        st.markdown("<div style='background:#1e293b; padding:12px; border-radius:10px; border-left:4px solid #22c55e'>", unsafe_allow_html=True)
        st.write("✅ **Quick Wins** (High Impact, Low Effort)")
        quick_wins = df_issues[(df_issues['effort'] <= 5) & (df_issues['impact'] >= 7)].sort_values('priority', ascending=False)
        if len(quick_wins) > 0:
            for _, issue in quick_wins.iterrows():
                st.write(f"🎯 **{issue['title']}**\nEffort: {issue['effort']}/10 | Impact: {issue['impact']}/10 | Priority: {issue['priority']}/100")
        else:
            st.write("*No quick wins at the moment*")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Top-Right: Major Features (High Impact, High Effort)
    with col2:
        st.markdown("<div style='background:#1e293b; padding:12px; border-radius:10px; border-left:4px solid #f59e0b'>", unsafe_allow_html=True)
        st.write("🔥 **Major Features** (High Impact, High Effort)")
        major = df_issues[(df_issues['effort'] >= 7) & (df_issues['impact'] >= 8)].sort_values('priority', ascending=False)
        if len(major) > 0:
            for _, issue in major.iterrows():
                st.write(f"🔥 **{issue['title']}**\nEffort: {issue['effort']}/10 | Impact: {issue['impact']}/10 | Priority: {issue['priority']}/100")
        else:
            st.write("*No major features scheduled*")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Bottom-Left: Nice to Have (Low Impact, Low Effort)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='background:#1e293b; padding:12px; border-radius:10px; border-left:4px solid #60a5fa'>", unsafe_allow_html=True)
        st.write("💡 **Nice to Have** (Low Impact, Low Effort)")
        nice = df_issues[(df_issues['effort'] <= 5) & (df_issues['impact'] < 7)].sort_values('priority', ascending=False)
        if len(nice) > 0:
            for _, issue in nice.iterrows():
                st.write(f"💡 **{issue['title']}**\nEffort: {issue['effort']}/10 | Impact: {issue['impact']}/10")
        else:
            st.write("*No low-priority items*")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Bottom-Right: Avoid (Low Impact, High Effort)
    with col2:
        st.markdown("<div style='background:#1e293b; padding:12px; border-radius:10px; border-left:4px solid #ef4444'>", unsafe_allow_html=True)
        st.write("⏸️ **Avoid** (Low Impact, High Effort)")
        avoid = df_issues[(df_issues['effort'] >= 7) & (df_issues['impact'] < 8)].sort_values('priority', ascending=False)
        if len(avoid) > 0:
            for _, issue in avoid.iterrows():
                st.write(f"⏸️ **{issue['title']}**\nEffort: {issue['effort']}/10 | Impact: {issue['impact']}/10")
        else:
            st.write("*No issues to avoid*")
        st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: RANKED LIST
with tab2:
    st.subheader("Issues Ranked by Priority")
    
    # Sort by priority
    df_sorted = df_issues.sort_values('priority', ascending=False)
    
    for idx, (_, issue) in enumerate(df_sorted.iterrows(), 1):
        priority_type, priority_label = get_priority_level(issue['effort'], issue['impact'])
        
        col1, col2, col3 = st.columns([0.5, 3, 1])
        with col1:
            st.write(f"**#{idx}**")
        with col2:
            st.write(f"{priority_label} **{issue['title']}**")
            st.caption(f"{issue['description']}")
            st.write(f"Effort: {issue['effort']}/10 | Impact: {issue['impact']}/10 | Status: {issue['status']}")
        with col3:
            st.metric("Priority", f"{issue['priority']}/100")
        st.divider()

# TAB 3: ADD NEW ISSUE
with tab3:
    st.subheader("Add New Issue")
    
    with st.form("new_issue_form"):
        title = st.text_input("Issue Title")
        description = st.text_area("Description", height=80)
        
        col1, col2 = st.columns(2)
        with col1:
            effort = st.slider("Effort (1-10)", 1, 10, 5)
        with col2:
            impact = st.slider("Impact (1-10)", 1, 10, 5)
        
        status = st.selectbox("Status", ["🔵 Planned", "🟡 In Review", "🟢 Easy", "🔴 Blocked"])
        tags = st.multiselect("Tags", ["audio", "quality", "feature", "performance", "ui", "export", "storage", "multilingual"])
        
        if st.form_submit_button("✅ Add Issue", use_container_width=True):
            new_issue = {
                "id": max([i['id'] for i in st.session_state.issues]) + 1,
                "title": title,
                "description": description,
                "effort": effort,
                "impact": impact,
                "status": status,
                "tags": tags
            }
            st.session_state.issues.append(new_issue)
            st.success(f"✅ Issue '{title}' added successfully!")
            st.rerun()

# TAB 4: ISSUE DETAILS
with tab4:
    st.subheader("Issue Details & Management")
    
    issue_select = st.selectbox(
        "Select an issue to view details",
        options=[f"{i['title']} (Priority: {calculate_priority(i['effort'], i['impact'])}/100)" for i in st.session_state.issues],
        index=0
    )
    
    if issue_select:
        selected_idx = [i['title'] for i in st.session_state.issues].index(issue_select.split(' (')[0])
        issue = st.session_state.issues[selected_idx]
        
        st.write(f"### {issue['title']}")
        st.write(f"{issue['description']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Effort", f"{issue['effort']}/10")
        with col2:
            st.metric("Impact", f"{issue['impact']}/10")
        with col3:
            st.metric("Priority", f"{calculate_priority(issue['effort'], issue['impact'])}/100")
        with col4:
            st.metric("Status", issue['status'])
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            new_status = st.selectbox("Update Status", ["🔵 Planned", "🟡 In Review", "🟢 Easy", "🔴 Blocked"], index=["🔵 Planned", "🟡 In Review", "🟢 Easy", "🔴 Blocked"].index(issue['status']))
            if st.button("📝 Update Status"):
                st.session_state.issues[selected_idx]['status'] = new_status
                st.success("✅ Status updated!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Issue", type="secondary"):
                st.session_state.issues.pop(selected_idx)
                st.success("✅ Issue deleted!")
                st.rerun()
        
        st.divider()
        st.write(f"**Tags:** {', '.join(issue['tags']) if issue['tags'] else 'None'}")

st.markdown("""
<style>
    footer {display: none;}
    .viewerBadge_container__1QSob {display: none;}
</style>
""", unsafe_allow_html=True)
