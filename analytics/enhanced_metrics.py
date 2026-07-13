"""
Enhanced football metrics calculation module.

Provides corrected and enriched metrics for:
- Shot accuracy (categorized: blocked, off-target, saved, goal)
- Pass completion with proper StatsBomb handling
- Progressive passes and carries
- Final-third and penalty-box actions
- Turnovers and high pressure situations
- Distance-based shot quality
"""

import pandas as pd
import numpy as np


def categorize_shot_outcome(shots: pd.DataFrame) -> dict:
    """
    Properly categorize shots into blocked, off-target, saved, goal, post.
    
    Args:
        shots: DataFrame filtered to shot events
        
    Returns:
        dict with counts for each shot type
    """
    if shots.empty:
        return {'blocked': 0, 'off_target': 0, 'saved': 0, 'goal': 0, 'post': 0, 'total': 0, 'accuracy_pct': 0.0, 'on_target_pct': 0.0}
    
    categories = {
        'goal': 0,
        'saved': 0,
        'off_target': 0,
        'blocked': 0,
        'post': 0
    }
    
    for _, shot in shots.iterrows():
        outcome = str(shot.get('shot_outcome', '')).lower()
        
        if outcome == 'goal':
            categories['goal'] += 1
        elif outcome == 'saved':
            categories['saved'] += 1
        elif outcome == 'off target':
            categories['off_target'] += 1
        elif outcome == 'blocked':
            categories['blocked'] += 1
        elif outcome == 'wayward':
            categories['off_target'] += 1
        elif outcome == 'post':
            categories['post'] += 1
        else:
            # Unknown outcome - assume off target
            categories['off_target'] += 1
    
    categories['total'] = len(shots)
    if categories['total'] > 0:
        categories['on_target_pct'] = (categories['goal'] + categories['saved']) / categories['total']
        categories['accuracy_pct'] = (categories['goal'] + categories['saved']) / categories['total']
    else:
        categories['on_target_pct'] = 0
        categories['accuracy_pct'] = 0
    
    return categories


def calculate_pass_completion(passes: pd.DataFrame) -> dict:
    """
    Calculate pass completion rate with proper StatsBomb handling.
    
    StatsBomb uses 'pass_outcome' field: NaN = completed, else = incomplete type
    
    Args:
        passes: DataFrame filtered to pass events
        
    Returns:
        dict with completion stats
    """
    if passes.empty:
        return {'completed': 0, 'total': 0, 'completion_pct': 0.0, 'incomplete': 0}
    
    total = len(passes)
    # In StatsBomb, completed passes have NaN in pass_outcome
    completed = passes['pass_outcome'].isna().sum()
    incomplete = passes['pass_outcome'].notna().sum()
    
    return {
        'completed': int(completed),
        'incomplete': int(incomplete),
        'total': int(total),
        'completion_pct': float(completed / total) if total > 0 else 0.0
    }


def calculate_progressive_passes(passes: pd.DataFrame) -> int:
    """
    Count progressive passes: advance ball 10+ yards toward opponent goal.
    
    Requires 'location' and 'pass_end_location' fields.
    """
    if passes.empty or 'location' not in passes.columns:
        return 0
    
    progressive_count = 0
    
    for _, pass_event in passes.iterrows():
        try:
            start_loc = pass_event.get('location')
            end_loc = pass_event.get('pass_end_location')
            
            if not start_loc or not end_loc:
                continue
            
            # Convert to tuples if needed
            if isinstance(start_loc, (list, tuple)) and isinstance(end_loc, (list, tuple)):
                start_x, start_y = start_loc[0], start_loc[1]
                end_x, end_y = end_loc[0], end_loc[1]
                
                # Progressive if advancing toward opponent goal (x direction)
                # StatsBomb pitch: 0-120, so advancing means increasing x
                if end_x - start_x >= 10:  # 10 yards ≈ 9.144m, StatsBomb uses ~120 units for 100m
                    progressive_count += 1
        except (TypeError, ValueError, IndexError):
            continue
    
    return progressive_count


def calculate_progressive_carries(carries: pd.DataFrame) -> int:
    """
    Count progressive carries: advance ball 5+ yards toward opponent goal.
    
    Requires 'location' and 'carry_end_location' fields.
    """
    if carries.empty or 'location' not in carries.columns:
        return 0
    
    progressive_count = 0
    
    for _, carry_event in carries.iterrows():
        try:
            start_loc = carry_event.get('location')
            end_loc = carry_event.get('carry_end_location')
            
            if not start_loc or not end_loc:
                continue
            
            if isinstance(start_loc, (list, tuple)) and isinstance(end_loc, (list, tuple)):
                start_x, start_y = start_loc[0], start_loc[1]
                end_x, end_y = end_loc[0], end_loc[1]
                
                # Progressive if advancing toward opponent goal
                if end_x - start_x >= 5:  # 5 yards ≈ 4.572m
                    progressive_count += 1
        except (TypeError, ValueError, IndexError):
            continue
    
    return progressive_count


def calculate_final_third_actions(events: pd.DataFrame) -> dict:
    """
    Count actions in the final third (opponent's defensive third).
    
    StatsBomb pitch: 0-120 in x, so final third = x >= 80
    """
    if events.empty or 'location' not in events.columns:
        return {
            'final_third_passes': 0,
            'final_third_carries': 0,
            'final_third_shots': 0,
            'final_third_pressures': 0
        }
    
    final_third_events = events[events['location'].apply(
        lambda loc: isinstance(loc, (list, tuple)) and len(loc) >= 1 and loc[0] >= 80
    )]
    
    return {
        'final_third_passes': len(final_third_events[final_third_events['type'] == 'Pass']),
        'final_third_carries': len(final_third_events[final_third_events['type'] == 'Carry']),
        'final_third_shots': len(final_third_events[final_third_events['type'] == 'Shot']),
        'final_third_pressures': len(final_third_events[final_third_events['type'] == 'Pressure'])
    }


def calculate_penalty_box_actions(shots: pd.DataFrame) -> dict:
    """
    Analyze shots from penalty box area.
    
    Penalty box: x >= 102, 18 < y < 62 (StatsBomb coordinates)
    """
    if shots.empty or 'location' not in shots.columns:
        return {
            'penalty_box_shots': 0,
            'penalty_box_goals': 0,
            'avg_shot_distance': 0.0
        }
    
    penalty_box_shots = shots[shots['location'].apply(
        lambda loc: isinstance(loc, (list, tuple)) and len(loc) >= 2 and loc[0] >= 102
    )]
    
    penalty_box_goals = penalty_box_shots[
        penalty_box_shots['shot_outcome'].astype(str).str.lower() == 'goal'
    ]
    
    # Calculate average shot distance (xG typically uses distance metric)
    distances = []
    for _, shot in shots.iterrows():
        try:
            if 'shot_distance' in shots.columns:
                dist = shot.get('shot_distance')
                if isinstance(dist, (int, float)):
                    distances.append(dist)
        except:
            pass
    
    avg_distance = np.mean(distances) if distances else 0.0
    
    return {
        'penalty_box_shots': len(penalty_box_shots),
        'penalty_box_goals': len(penalty_box_goals),
        'avg_shot_distance': float(avg_distance)
    }


def calculate_turnovers(events: pd.DataFrame) -> dict:
    """
    Count high-stakes turnovers (dispossessed, miscontrol, ball recovery by opponent).
    """
    if events.empty:
        return {
            'dispossessed': 0,
            'miscontrol': 0,
            'ball_recovery_against': 0,
            'total_turnovers': 0
        }
    
    dispossessed = len(events[events['type'] == 'Dispossessed'])
    miscontrol = len(events[events['type'] == 'Miscontrol'])
    ball_recovery_against = len(events[events['type'] == 'Ball Recovery'])
    
    return {
        'dispossessed': int(dispossessed),
        'miscontrol': int(miscontrol),
        'ball_recovery_against': int(ball_recovery_against),
        'total_turnovers': int(dispossessed + miscontrol)
    }


def calculate_attacking_pressures(events: pd.DataFrame) -> int:
    """
    Count pressures applied in attacking half (x >= 60).
    """
    if events.empty or 'location' not in events.columns:
        return 0
    
    pressures = events[events['type'] == 'Pressure']
    if pressures.empty:
        return 0
    
    attacking_pressures = pressures[pressures['location'].apply(
        lambda loc: isinstance(loc, (list, tuple)) and len(loc) >= 1 and loc[0] >= 60
    )]
    
    return len(attacking_pressures)


def build_team_comparison(team1_events: pd.DataFrame, team2_events: pd.DataFrame, team1_name: str, team2_name: str) -> pd.DataFrame:
    """
    Build a comparison DataFrame between two teams.
    
    Args:
        team1_events: Events for team 1
        team2_events: Events for team 2
        team1_name: Name of team 1
        team2_name: Name of team 2
        
    Returns:
        DataFrame with metrics comparison
    """
    metrics = []
    
    # Pass completion
    team1_passes = team1_events[team1_events['type'] == 'Pass']
    team2_passes = team2_events[team2_events['type'] == 'Pass']
    
    team1_pass_stats = calculate_pass_completion(team1_passes)
    team2_pass_stats = calculate_pass_completion(team2_passes)
    
    metrics.append({
        'Metric': 'Pass Completion %',
        team1_name: f"{team1_pass_stats['completion_pct']:.1%}",
        team2_name: f"{team2_pass_stats['completion_pct']:.1%}"
    })
    
    # Shots
    team1_shots = team1_events[team1_events['type'] == 'Shot']
    team2_shots = team2_events[team2_events['type'] == 'Shot']
    
    team1_shot_cats = categorize_shot_outcome(team1_shots)
    team2_shot_cats = categorize_shot_outcome(team2_shots)
    
    metrics.append({
        'Metric': 'Shots',
        team1_name: team1_shot_cats['total'],
        team2_name: team2_shot_cats['total']
    })
    
    metrics.append({
        'Metric': 'Shots On Target',
        team1_name: team1_shot_cats['goal'] + team1_shot_cats['saved'],
        team2_name: team2_shot_cats['goal'] + team2_shot_cats['saved']
    })
    
    metrics.append({
        'Metric': 'Goals',
        team1_name: team1_shot_cats['goal'],
        team2_name: team2_shot_cats['goal']
    })
    
    # Progressive actions
    team1_prog_passes = calculate_progressive_passes(team1_passes)
    team2_prog_passes = calculate_progressive_passes(team2_passes)
    
    metrics.append({
        'Metric': 'Progressive Passes',
        team1_name: team1_prog_passes,
        team2_name: team2_prog_passes
    })
    
    team1_carries = team1_events[team1_events['type'] == 'Carry']
    team2_carries = team2_events[team2_events['type'] == 'Carry']
    
    team1_prog_carries = calculate_progressive_carries(team1_carries)
    team2_prog_carries = calculate_progressive_carries(team2_carries)
    
    metrics.append({
        'Metric': 'Progressive Carries',
        team1_name: team1_prog_carries,
        team2_name: team2_prog_carries
    })
    
    # Turnovers
    team1_turnovers = calculate_turnovers(team1_events)
    team2_turnovers = calculate_turnovers(team2_events)
    
    metrics.append({
        'Metric': 'Turnovers',
        team1_name: team1_turnovers['total_turnovers'],
        team2_name: team2_turnovers['total_turnovers']
    })
    
    # Pressures
    team1_pressures = len(team1_events[team1_events['type'] == 'Pressure'])
    team2_pressures = len(team2_events[team2_events['type'] == 'Pressure'])
    
    metrics.append({
        'Metric': 'Pressures',
        team1_name: team1_pressures,
        team2_name: team2_pressures
    })
    
    team1_attacking_pressures = calculate_attacking_pressures(team1_events)
    team2_attacking_pressures = calculate_attacking_pressures(team2_events)
    
    metrics.append({
        'Metric': 'Pressures in Attacking Half',
        team1_name: team1_attacking_pressures,
        team2_name: team2_attacking_pressures
    })
    
    return pd.DataFrame(metrics)
