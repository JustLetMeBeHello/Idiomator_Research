from __future__ import annotations


def registry() -> list[dict]:
    return [
        {"id": "research", "name": "Research", "icon": "🔬", "status": "live"},
        {"id": "papers", "name": "Papers", "icon": "📄", "status": "stub"},
        {"id": "agents", "name": "Agents", "icon": "🤖", "status": "stub"},
        {"id": "oscore", "name": "OS Core", "icon": "⚙️", "status": "stub"},
    ]
