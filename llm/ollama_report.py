from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests

HUGGINGFACE_DEFAULT_MODEL = 'mistral-7b-instruct'


def has_ollama() -> bool:
    return shutil.which('ollama') is not None


def has_huggingface() -> bool:
    return bool(os.getenv('HUGGINGFACE_API_KEY'))


def generate_huggingface_report(prompt: str, model: str = HUGGINGFACE_DEFAULT_MODEL) -> str:
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        raise EnvironmentError('HUGGINGFACE_API_KEY environment variable is not set. Set it before generating reports.')

    url = f'https://api-inference.huggingface.co/v1/models/{model}'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload = {
        'inputs': prompt,
        'options': {'wait_for_model': True},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f'Hugging Face generation failed ({response.status_code}): {response.text.strip()}'
        )

    data = response.json()
    if isinstance(data, dict) and data.get('error'):
        raise RuntimeError(f"Hugging Face error: {data.get('error')}")

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if 'generated_text' in data[0]:
            return data[0]['generated_text'].strip()
        if 'generated_texts' in data[0]:
            return '\n'.join(data[0]['generated_texts']).strip()

    if isinstance(data, dict) and 'generated_text' in data:
        return data['generated_text'].strip()

    if isinstance(data, str):
        return data.strip()

    raise RuntimeError(f'Unexpected Hugging Face response format: {data}')


def build_coaching_report_prompt(team: str, opponent: str, season_name: str, summary: dict, xg_summary: dict, cluster_label: int | None, cluster_desc: dict) -> str:
    lines = [
        f"Create a concise executive pre-match coaching report for {team} ahead of a match against {opponent}.",
        f"Season: {season_name}.",
        "Provide a short tactical profile, expected goals context, strengths, weaknesses, and a recommendation section.",
        "Use the data below to create a clear, football-coaching style narrative.",
        "",
        "Team summary:",
        f"- Total matches in view: {summary.get('matches', 0)}",
        f"- Wins: {summary.get('wins', 0)}, Draws: {summary.get('draws', 0)}, Losses: {summary.get('losses', 0)}",
        f"- Goals for: {summary.get('goals_for', 0)}, Goals against: {summary.get('goals_against', 0)}",
        f"- Pass completion: {summary.get('pass_completion', 'N/A')}",
        f"- Shots (team): {summary.get('shots', 'N/A')}",
        "",
        "Expected goals summary:",
        f"- Team xG: {xg_summary.get('team_xg', {}).get(team, 0):.2f}",
        f"- Opponent xG: {xg_summary.get('team_xg', {}).get(opponent, 0):.2f}",
        f"- Total xG created by team: {xg_summary.get('total_xg', 0):.2f}",
        "",
    ]
    if cluster_label is not None:
        lines.extend([
            f"Tactical cluster label: {cluster_label}.",
            "Cluster signature (top event shares):",
        ])
        cluster_features = cluster_desc.get(cluster_label, {})
        for name, value in cluster_features.items():
            lines.append(f"- {name.replace('evt_', '').replace('_', ' ')}: {value:.2%}")
        lines.append("")

    top_event_types = summary.get('event_type_counts')
    if hasattr(top_event_types, 'items'):
        lines.append("Top event types:")
        for event_type, count in list(top_event_types.items())[:5]:
            lines.append(f"- {event_type}: {count}")
        lines.append("")

    lines.append("Use bullet points and clear coaching recommendations at the end.")
    return '\n'.join(lines)


def generate_local_ollama_report(prompt: str, model: str = 'llama3') -> str:
    if not has_ollama():
        raise EnvironmentError('Ollama CLI not found. Install Ollama and pull a compatible model before generating reports.')

    command = ['ollama', 'generate', model, prompt]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if process.returncode != 0:
        raise RuntimeError(f'Ollama generation failed: {process.stderr.strip()}')
    return process.stdout.strip()


def generate_ollama_report(prompt: str, model: str = 'llama3') -> str:
    if has_ollama():
        return generate_local_ollama_report(prompt, model)
    if has_huggingface():
        return generate_huggingface_report(prompt)
    raise EnvironmentError(
        'No local Ollama installation or Hugging Face API key found. '
        'Install Ollama locally or set HUGGINGFACE_API_KEY to use LLM reports.'
    )


def prompt_preview(prompt: str) -> str:
    lines = [
        'LLM generation is not available in this environment.',
        'To use Ollama locally, install Ollama and pull a compatible model, then run the following prompt:',
        '',
        prompt,
    ]
    if has_huggingface():
        lines.extend([
            '',
            'Alternatively, you can use Hugging Face with HUGGINGFACE_API_KEY set.',
        ])
    return '\n'.join(lines)
