"""
vis.js 기반 인터랙티브 그래프 HTML 생성
"""
import json
from typing import Optional

def _node_color(paper: dict, is_root: bool) -> dict:
    """인용 수와 루트 여부에 따라 노드 색상 결정"""
    if is_root:
        return {
            "background": "#4f46e5",
            "border": "#818cf8",
            "highlight": {"background": "#6366f1", "border": "#a5b4fc"},
        }
    citations = paper.get("citation_count", 0) or 0
    if citations >= 500:
        return {"background": "#7c3aed", "border": "#a78bfa",
                "highlight": {"background": "#8b5cf6", "border": "#c4b5fd"}}
    elif citations >= 100:
        return {"background": "#1d4ed8", "border": "#60a5fa",
                "highlight": {"background": "#2563eb", "border": "#93c5fd"}}
    elif citations >= 20:
        return {"background": "#0f766e", "border": "#2dd4bf",
                "highlight": {"background": "#0d9488", "border": "#5eead4"}}
    else:
        return {"background": "#1f2937", "border": "#4b5563",
                "highlight": {"background": "#374151", "border": "#6b7280"}}

def _node_size(citation_count: int, is_root: bool) -> int:
    if is_root:
        return 32
    if citation_count >= 1000:
        return 28
    elif citation_count >= 200:
        return 22
    elif citation_count >= 50:
        return 17
    elif citation_count >= 10:
        return 13
    return 10

def render_graph_html(graph_data: dict, papers: dict, height: int = 650) -> str:
    nodes_raw = graph_data.get("nodes", [])
    edges_raw = graph_data.get("edges", [])

    vis_nodes = []
    for n in nodes_raw:
        doi = n["id"]
        paper = papers.get(doi, n)
        is_root = n.get("is_root", False)
        color = _node_color(paper, is_root)
        size = _node_size(paper.get("citation_count", 0) or 0, is_root)

        authors = paper.get("authors", [])
        author_str = ", ".join(authors[:2])
        if len(authors) > 2:
            author_str += " et al."

        tooltip = (
            f"<b>{paper.get('title','')}</b><br>"
            f"{author_str}<br>"
            f"Year: {paper.get('year','?')} · Citations: {paper.get('citation_count',0):,}<br>"
            f"<small>{doi}</small>"
        )

        vis_nodes.append({
            "id": doi,
            "label": n.get("label", doi[:20]),
            "title": tooltip,
            "color": color,
            "size": size,
            "font": {
                "color": "#e2e8f0",
                "size": 11,
                "face": "Space Grotesk, sans-serif",
                "strokeWidth": 2,
                "strokeColor": "#0a0e1a",
            },
            "borderWidth": 2 if is_root else 1,
            "shadow": is_root,
        })

    vis_edges = []
    for e in edges_raw:
        vis_edges.append({
            "from": e["from"],
            "to": e["to"],
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "color": {
                "color": "#312e81" if e.get("type") == "reference" else "#1e3a5f",
                "opacity": 0.7,
                "highlight": "#818cf8",
            },
            "width": 1.5,
            "smooth": {"type": "curvedCW", "roundness": 0.1},
        })

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.css"/>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0a0e1a; overflow:hidden; }}
    #network {{ width:100%; height:{height}px; }}
    #legend {{
      position:absolute; bottom:12px; left:12px;
      background:rgba(17,24,39,0.9);
      border:1px solid #1f2937;
      border-radius:8px;
      padding:10px 14px;
      font-family:'Space Grotesk',sans-serif;
      font-size:11px; color:#94a3b8;
    }}
    .leg-row {{ display:flex; align-items:center; gap:8px; margin:3px 0; }}
    .leg-dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
    #info-panel {{
      display:none;
      position:absolute; top:12px; right:12px;
      background:rgba(17,24,39,0.95);
      border:1px solid #312e81;
      border-radius:10px;
      padding:14px 16px;
      font-family:'Space Grotesk',sans-serif;
      font-size:12px; color:#e2e8f0;
      max-width:280px; line-height:1.6;
    }}
    .info-title {{ font-weight:600; color:#818cf8; margin-bottom:6px; font-size:13px; }}
    .info-meta {{ color:#64748b; font-size:11px; }}
  </style>
</head>
<body>
<div id="network"></div>
<div id="legend">
  <div style="font-weight:600; color:#e2e8f0; margin-bottom:6px;">노드 크기 = 피인용 수</div>
  <div class="leg-row"><div class="leg-dot" style="background:#4f46e5;width:14px;height:14px;"></div> 루트 논문</div>
  <div class="leg-row"><div class="leg-dot" style="background:#7c3aed;"></div> 500회+</div>
  <div class="leg-row"><div class="leg-dot" style="background:#1d4ed8;"></div> 100~499회</div>
  <div class="leg-row"><div class="leg-dot" style="background:#0f766e;"></div> 20~99회</div>
  <div class="leg-row"><div class="leg-dot" style="background:#374151;"></div> ~19회</div>
</div>
<div id="info-panel">
  <div class="info-title" id="info-title"></div>
  <div class="info-meta" id="info-meta"></div>
</div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});

const container = document.getElementById('network');
const options = {{
  nodes: {{
    shape: 'dot',
    scaling: {{ min:8, max:35 }},
  }},
  edges: {{
    width: 1.5,
    selectionWidth: 3,
  }},
  physics: {{
    enabled: true,
    barnesHut: {{
      gravitationalConstant: -8000,
      centralGravity: 0.3,
      springLength: 140,
      springConstant: 0.04,
      damping: 0.12,
    }},
    stabilization: {{ iterations: 180, updateInterval: 25 }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 150,
    zoomView: true,
    dragView: true,
    navigationButtons: false,
    keyboard: false,
  }},
  layout: {{
    improvedLayout: true,
    randomSeed: 42,
  }},
}};

const network = new vis.Network(container, {{ nodes, edges }}, options);

// 노드 클릭 시 info panel 표시
const nodeMap = {{}};
{json.dumps(vis_nodes)}.forEach(n => {{ nodeMap[n.id] = n; }});

network.on('click', function(params) {{
  if (params.nodes.length > 0) {{
    const nodeId = params.nodes[0];
    const n = nodeMap[nodeId];
    const panel = document.getElementById('info-panel');
    document.getElementById('info-title').innerHTML = n.label;
    document.getElementById('info-meta').innerHTML = n.title
      .replace(/<b>/g,'').replace(/<\\/b>/g,'')
      .replace(/<br>/g, '<br/>');
    panel.style.display = 'block';
  }} else {{
    document.getElementById('info-panel').style.display = 'none';
  }}
}});

// 안정화 완료 후 물리 엔진 끔 (성능)
network.once('stabilizationIterationsDone', function() {{
  network.setOptions({{ physics: {{ enabled: false }} }});
}});
</script>
</body>
</html>
"""
    return html
