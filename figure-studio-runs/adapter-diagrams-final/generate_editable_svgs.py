from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent

BLUE = "#0068b7"
LIGHT_BLUE = "#dff1ff"
PANEL_BLUE = "#eef9ff"
YELLOW = "#fff45a"
PURPLE = "#eee4ff"
PURPLE_STROKE = "#7a3bd1"
RED = "#ff5b61"
RED_FILL = "#ffe7e7"
GREEN = "#00a95b"
GREEN_FILL = "#e4f8ea"
ORANGE = "#ff9a16"
GRAY = "#f2f2f2"
BLACK = "#111111"

POS = "#00b366"
NEG = "#7a35c5"
BND = "#ff9412"
Q = "#0d83bd"
TEAL = "#079aa9"
AMBER = "#f5d90a"


def _style() -> str:
    return """
    <style>
      svg { background: white; }
      text { font-family: Arial, Helvetica, sans-serif; }
      .title { font-size: 36px; font-weight: 800; }
      .subtitle { font-size: 20px; font-weight: 700; fill: #0068b7; }
      .section-title { font-size: 24px; font-weight: 800; }
      .label { font-size: 14px; }
      .note { font-size: 16px; font-weight: 700; fill: #0068b7; }
      .small { font-size: 12px; }
      .tiny { font-size: 10px; }
      .box-text { font-size: 15px; font-weight: 700; }
    </style>
    """


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.title = title
        self.body: list[str] = []

    def add(self, s: str) -> None:
        self.body.append(s)

    def defs(self) -> str:
        colors = {
            "blue": BLUE,
            "red": RED,
            "green": GREEN,
            "purple": PURPLE_STROKE,
            "orange": ORANGE,
            "black": BLACK,
        }
        parts = ["<defs>"]
        for name, color in colors.items():
            parts.append(
                f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            )
        parts.append("</defs>")
        return "\n".join(parts)

    def text(
        self,
        x: float,
        y: float,
        content: str | list[str],
        size: int = 14,
        weight: str = "400",
        fill: str = BLACK,
        anchor: str = "middle",
        klass: str | None = None,
        rotate: float | None = None,
    ) -> None:
        lines = content if isinstance(content, list) else str(content).split("\n")
        cls = f' class="{klass}"' if klass else ""
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        lh = size * 1.18
        start = y - (len(lines) - 1) * lh / 2
        tspans = []
        for i, line in enumerate(lines):
            tspans.append(
                f'<tspan x="{x:.1f}" y="{start + i * lh:.1f}">{escape(str(line))}</tspan>'
            )
        self.add(
            f'<text{cls} x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}"{transform}>'
            + "".join(tspans)
            + "</text>"
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = BLACK,
        sw: float = 2,
        rx: float = 8,
        dash: str | None = None,
        opacity: float | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        op_attr = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            f'{dash_attr}{op_attr}/>'
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str | list[str],
        fill: str = YELLOW,
        stroke: str = BLACK,
        sw: float = 2,
        rx: float = 9,
        text_size: int = 15,
        text_weight: str = "700",
        shadow: bool = False,
    ) -> None:
        if shadow:
            self.rect(x + 6, y + 6, w, h, fill="#cfcfcf", stroke="none", sw=0, rx=rx)
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx)
        self.text(
            x + w / 2,
            y + h / 2,
            label,
            size=text_size,
            weight=text_weight,
            anchor="middle",
        )

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        fill: str,
        stroke: str,
        dash: str | None = None,
        title_fill: str | None = None,
        sw: float = 2,
    ) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=18, dash=dash)
        self.text(
            x + 22,
            y + 32,
            title,
            size=24,
            weight="800",
            fill=title_fill or stroke,
            anchor="start",
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        label: str = "+",
        fill: str = "white",
        stroke: str = BLACK,
        sw: float = 2,
        size: int = 22,
    ) -> None:
        self.add(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
        self.text(cx, cy + 1, label, size=size, weight="800")

    def path(
        self,
        pts: list[tuple[float, float]],
        color: str = BLUE,
        sw: float = 4,
        dash: str | None = None,
        marker: bool = True,
        fill: str = "none",
    ) -> None:
        if not pts:
            return
        name = {
            BLUE: "blue",
            RED: "red",
            GREEN: "green",
            PURPLE_STROKE: "purple",
            ORANGE: "orange",
            BLACK: "black",
        }.get(color, "blue")
        d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f} " + " ".join(
            f"L {x:.1f} {y:.1f}" for x, y in pts[1:]
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#arrow-{name})"' if marker else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="square" stroke-linejoin="miter"{dash_attr}{marker_attr}/>'
        )

    def token_row(
        self,
        x: float,
        y: float,
        n: int = 8,
        colors: list[str] | None = None,
        size: float = 18,
        gap: float = 4,
        stroke: str = BLACK,
        label: str | None = None,
        label_y: float = 24,
        label_size: int = 12,
    ) -> None:
        palette = colors or [Q, "#1685c7", POS, AMBER, ORANGE, NEG, TEAL]
        for i in range(n):
            c = palette[i % len(palette)]
            self.rect(x + i * (size + gap), y, size, size, fill=c, stroke=stroke, sw=1, rx=0)
        if label:
            self.text(
                x + (n * size + (n - 1) * gap) / 2,
                y + label_y,
                label,
                size=label_size,
                weight="400",
            )

    def grid(
        self,
        x: float,
        y: float,
        rows: int = 4,
        cols: int = 4,
        cell: float = 18,
        colors: list[str] | None = None,
        stroke: str = "#bfc7ce",
        label: str | None = None,
        label_y: float = 0,
        label_size: int = 12,
    ) -> None:
        palette = colors or ["#f2f2f2", "#ff6a7a", "#a9cfe8", "#ffffff"]
        for r in range(rows):
            for c in range(cols):
                idx = (r * cols + c * 3 + r) % len(palette)
                self.rect(
                    x + c * cell,
                    y + r * cell,
                    cell,
                    cell,
                    fill=palette[idx],
                    stroke=stroke,
                    sw=1,
                    rx=0,
                )
        if label:
            self.text(
                x + cols * cell / 2,
                y + rows * cell + label_y,
                label,
                size=label_size,
                weight="400",
            )

    def save(self, name: str) -> None:
        svg = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img">\n'
            f"<title>{escape(self.title)}</title>\n"
            f"{self.defs()}\n{_style()}\n"
            + "\n".join(self.body)
            + "\n</svg>\n"
        )
        (OUT_DIR / name).write_text(svg, encoding="utf-8")


def arr(s: Svg, x1, y1, x2, y2, color=BLUE, dash=None, sw=4) -> None:
    s.path([(x1, y1), (x2, y2)], color=color, dash=dash, sw=sw)


def bent(s: Svg, pts, color=BLUE, dash=None, sw=4, marker=True) -> None:
    s.path(pts, color=color, dash=dash, sw=sw, marker=marker)


def stage_a_lora() -> None:
    s = Svg(3000, 960, "Stage-A LoRA + refine_head")
    s.text(32, 42, "(b) Stage-A LoRA + refine_head", size=36, weight="800", anchor="start")
    s.text(350, 104, "LoRA injection path inside frozen SAM3", size=26, weight="800", anchor="start")
    s.text(1140, 80, "Stage-A target scopes: vision_encoder + mask_decoder", size=16, weight="800", fill=BLUE, anchor="start")
    s.text(350, 132, "Base LoRA branch: SAM3 base is frozen; only LoRA A/B plus the outer refine_head are trainable", size=15, weight="800", fill=BLUE, anchor="start")

    s.grid(60, 200, label="input image\n[B,3,H,W]", label_y=26)
    s.box(250, 180, 190, 115, ["Official SAM3", "image model", "(frozen)"], fill=LIGHT_BLUE, stroke=BLUE, text_size=15)
    arr(s, 150, 236, 250, 236)
    arr(s, 440, 236, 525, 236)
    for i in range(12):
        col = "#858585" if i < 8 else POS
        s.rect(530 + i * 44, 208, 36, 54, fill=col, stroke=BLACK, sw=1, rx=0)
    s.text(635, 182, "vision_encoder blocks", size=18, weight="800")
    s.text(885, 184, "late 1/3 blocks", size=14, weight="800", fill=BLUE)
    s.text(546, 282, "0", size=11)
    s.text(850, 282, "8", size=11)
    s.text(990, 282, "11", size=11)
    bent(s, [(900, 170), (900, 150), (1060, 150), (1060, 170)])
    arr(s, 1020, 236, 1140, 236)
    s.box(1140, 120, 220, 235, ["q_proj + LoRA", "v_proj + LoRA", "qkv + LoRA", "attn.proj + LoRA", "out_proj + LoRA", ".proj + LoRA"], fill="#fff000", text_size=18)
    arr(s, 1360, 236, 1445, 236)
    s.box(1140, 370, 220, 90, ["mask_decoder", "Transformer"], fill=LIGHT_BLUE, stroke=BLUE, text_size=19)
    s.path([(440, 258), (480, 258), (480, 405), (1140, 405)], color=BLUE)
    s.box(1440, 345, 230, 100, ["attention/proj", "Linear layers", "+ LoRA"], fill="#fff000", text_size=18)
    arr(s, 1360, 405, 1440, 395)
    arr(s, 1670, 395, 1755, 395)
    s.token_row(1760, 386, 9, label="decoder tokens")
    arr(s, 1960, 395, 2050, 395)
    s.box(2050, 360, 190, 90, ["mask logits", "from SAM3"], fill=LIGHT_BLUE, stroke=BLUE, text_size=18)

    s.text(1445, 136, "LoRALinear internals", size=20, weight="800", anchor="start")
    s.token_row(1450, 175, 7, colors=["#1f83c4"], label="x")
    arr(s, 1615, 184, 1680, 184)
    s.box(1680, 148, 105, 70, ["frozen", "Linear"], fill=YELLOW, text_size=14)
    arr(s, 1785, 184, 1880, 184)
    s.token_row(1885, 175, 7, colors=[TEAL], label="base")
    bent(s, [(1615, 225), (1640, 225), (1640, 300), (1680, 300)])
    s.box(1680, 270, 105, 60, "Dropout", text_size=14)
    arr(s, 1785, 300, 1840, 300)
    s.box(1840, 270, 105, 60, ["LoRA A", "C->r"], text_size=14)
    arr(s, 1945, 300, 2025, 300)
    s.token_row(2028, 286, 4, colors=[ORANGE], label="rank r")
    arr(s, 2130, 300, 2170, 300)
    s.box(2170, 270, 115, 60, ["LoRA B", "r->C"], text_size=14)
    arr(s, 2285, 300, 2350, 300)
    s.box(2350, 270, 110, 60, ["scale", "alpha/r"], text_size=14)
    s.circle(2525, 235, 34, "+")
    bent(s, [(2025, 184), (2525, 184), (2525, 201)])
    bent(s, [(2460, 300), (2525, 300), (2525, 269)])
    arr(s, 2559, 235, 2635, 235)
    s.token_row(2640, 226, 9, label="LoRA-adapted\nprojection")
    s.text(2010, 410, "base weights frozen; only LoRA A/B are trainable", size=15, weight="800", fill=BLUE)

    s.text(350, 620, "refine_head residual mask refinement", size=25, weight="800", anchor="start")
    bent(s, [(480, 258), (480, 690), (560, 690)])
    s.token_row(575, 648, 6, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], size=22, label="image_embeddings\nfeature_map [B,C,h,w]", label_y=70)
    arr(s, 730, 690, 825, 690)
    s.box(825, 650, 165, 80, ["resolve", "feature_map"], text_size=14)
    arr(s, 990, 690, 1080, 690)
    s.box(1080, 650, 145, 80, ["refine_head", "Conv 1x1"], text_size=14)
    arr(s, 1225, 690, 1310, 690)
    s.token_row(1315, 677, 8, colors=["#eaf3f7", "#ffc564"], label="delta [B,1,h,w]")
    arr(s, 1510, 690, 1570, 690)
    s.box(1570, 650, 155, 80, ["bilinear", "interpolate"], text_size=14)
    arr(s, 1725, 690, 1810, 690)
    s.token_row(1815, 677, 8, colors=["#eaf3f7", "#ffc564"], label="delta [B,1,H,W]")
    arr(s, 2010, 690, 2080, 690)
    s.box(2080, 650, 115, 80, ["scale", "0.1"], text_size=14)
    arr(s, 2195, 690, 2355, 690)
    s.circle(2385, 670, 34, "+")
    bent(s, [(2240, 405), (2325, 405), (2325, 650), (2355, 650)])
    s.grid(2385, 608, rows=4, cols=5, cell=18, label="SAM3\nmask_logits", label_y=20)
    arr(s, 2419, 670, 2520, 670)
    s.grid(2530, 642, rows=3, cols=7, cell=18, label="final mask_logits\n+ 0.1 * delta", label_y=20)
    arr(s, 2660, 670, 2765, 670)
    s.box(2765, 650, 110, 80, "sigmoid", text_size=14)
    arr(s, 2875, 670, 2930, 670)
    s.grid(2935, 634, rows=4, cols=4, cell=20, colors=["#111", "#fff", "#fff", "#fff"], label="Mask\n[B,1,H,W]", label_y=28)
    s.text(1170, 765, "refine_head is trainable in the MedEx wrapper and adds a residual correction to SAM3 mask logits", size=15, weight="800", fill=BLUE)
    s.text(1170, 790, "final_mask_logits = sam3_mask_logits + 0.1 * refine_delta", size=15, weight="800", fill=BLUE)
    s.save("00_final_StageA_LoRA_refine_head.svg")


def medical_image_adapter(width: int, height: int, name: str, portrait: bool = False) -> None:
    if portrait:
        s = Svg(width, height, "MedicalImageAdapter detailed")
        s.text(38, 50, "(c) MedicalImageAdapter", size=34, weight="800", anchor="start")
        s.text(38, 88, "Optional wrapper module: enabled by config enable_medical_adapter; not part of the default LoRA branch", size=14, weight="800", fill=BLUE, anchor="start")
        s.grid(335, 110, rows=4, cols=4, cell=22, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="feature_map X\n[B,C,H,W]", label_y=34)
        s.text(165, 275, "B. Texture conv path\n(depthwise separable conv)", size=22, weight="800", anchor="start")
        s.text(485, 275, "A. Bottleneck residual path\n(MLP adapter)", size=22, weight="800", anchor="start")
        x1, x2 = 180, 520
        y = 360
        bent(s, [(380, 205), (380, 305), (x1, 305), (x1, 360)])
        bent(s, [(380, 205), (380, 305), (x2, 305), (x2, 360)])
        for label, yy in [
            (["Depthwise", "Conv 3x3", "groups=C"], 360),
            ("GELU", 540),
            (["Pointwise", "Conv 1x1"], 700),
        ]:
            s.box(x1 - 55, yy, 110, 80, label)
        arr(s, x1, 440, x1, 540)
        s.grid(x1 - 45, 460, rows=3, cols=4, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="[B,C,H,W]", label_y=18)
        arr(s, x1, 620, x1, 700)
        arr(s, x1, 780, x1, 980)
        s.grid(x1 - 50, 815, rows=3, cols=4, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="texture_update\n[B,C,H,W]", label_y=24)
        for label, yy in [
            (["permute", "[B,H,W,C]"], 360),
            ("LayerNorm(C)", 550),
            (["Linear", "C->C/4"], 730),
            ("GELU", 900),
            ("Dropout", 1060),
            (["Linear", "C/4->C"], 1220),
            (["scale", "gamma"], 1380),
        ]:
            s.box(x2 - 55, yy, 110, 80, label)
        ys = [440, 550, 630, 730, 810, 900, 980, 1060, 1140, 1220, 1300, 1380]
        for a, b in zip(ys[::2], ys[1::2]):
            arr(s, x2, a, x2, b)
        s.grid(x2 - 45, 460, rows=1, cols=6, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="tokens\n[B,H*W,C]", label_y=22)
        s.grid(x2 - 45, 1320, rows=3, cols=4, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="adapter_update\n[B,H,W,C]", label_y=24)
        s.circle(380, 1620, 34, "+")
        bent(s, [(x1, 980), (x1, 1620), (346, 1620)])
        bent(s, [(x2, 1460), (x2, 1540), (414, 1540), (414, 1588)])
        bent(s, [(380, 205), (650, 205), (650, 1588)])
        arr(s, 380, 1654, 380, 1760)
        s.grid(330, 1765, rows=4, cols=5, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="adapted\nfeature_map\n[B,C,H,W]", label_y=28)
        arr(s, 380, 1900, 380, 2030)
        s.box(290, 2030, 180, 110, ["refine_head /", "downstream", "mask"], fill=LIGHT_BLUE, stroke=BLUE, text_size=18)
        s.save(name)
        return

    s = Svg(width, height, "MedicalImageAdapter")
    s.text(30, 48, "(c) MedicalImageAdapter", size=36, weight="800", anchor="start")
    s.text(350, 48, "Optional wrapper module: enable_medical_adapter; adapts image_embeddings after SAM3 forward", size=16, weight="800", fill=BLUE, anchor="start")
    s.grid(48, 330, rows=3, cols=6, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="feature_map X\n[B,C,H,W]", label_y=25)
    s.text(390, 112, "A. Bottleneck residual path (MLP adapter)", size=24, weight="800", anchor="start")
    s.text(390, 455, "B. Texture conv path (depthwise separable conv)", size=24, weight="800", anchor="start")
    bent(s, [(200, 350), (260, 350), (260, 176), (330, 176)])
    bent(s, [(200, 410), (260, 410), (260, 555), (330, 555)])
    top = 175
    x = 330
    modules = [
        (["permute", "[B,H,W,C]"], 135),
        ("LayerNorm(C)", 810),
        (["Linear", "C->C/4"], 1250),
        ("GELU", 1620),
        ("Dropout", 1765),
        (["Linear", "C/4->C"], 1935),
        (["scale", "gamma"], 2300),
    ]
    last_right = 330
    for label, mx in modules:
        s.box(mx, 140, 135, 70, label)
    xs = [465, 545, 810, 960, 1250, 1370, 1620, 1710, 1765, 1885, 1935, 2055, 2300]
    arr(s, 465, top, 545, top)
    s.token_row(550, 166, 9, label="tokens\n[B,H*W,C]", label_y=34)
    arr(s, 725, top, 810, top)
    arr(s, 945, top, 1035, top)
    s.token_row(1040, 166, 7, label="normed")
    arr(s, 1185, top, 1250, top)
    arr(s, 1370, top, 1440, top)
    s.token_row(1445, 166, 6, label="bottleneck\n[B,H*W,C/4]", label_y=34)
    arr(s, 1570, top, 1620, top)
    arr(s, 1710, top, 1765, top)
    arr(s, 1885, top, 1935, top)
    arr(s, 2055, top, 2130, top)
    s.token_row(2135, 152, 5, size=22, label="adapter_update\n[B,H*W,C]", label_y=58)
    arr(s, 2235, top, 2300, top)
    s.circle(2475, 175, 32, "+")
    arr(s, 2410, top, 2443, top)
    bent(s, [(260, 175), (2475, 175), (2475, 143)], marker=False)
    arr(s, 2475, 207, 2475, 250)
    s.grid(2445, 250, rows=4, cols=6, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="bottleneck output\n[B,C,H,W]", label_y=28)

    y = 555
    s.box(330, 515, 160, 80, ["Depthwise", "Conv 3x3", "groups=C"])
    arr(s, 490, y, 550, y)
    s.grid(555, 520, rows=3, cols=6, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="[B,C,H,W]", label_y=70)
    arr(s, 680, y, 750, y)
    s.box(750, 520, 95, 70, "GELU")
    arr(s, 845, y, 900, y)
    s.box(900, 515, 135, 80, ["Pointwise", "Conv 1x1"])
    arr(s, 1035, y, 1095, y)
    s.grid(1100, 520, rows=3, cols=6, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="texture_update\n[B,C,H,W]", label_y=70)
    bent(s, [(1225, y), (1970, y), (1970, 515), (2465, 515), (2465, 550)])
    bent(s, [(2475, 330), (2475, 400), (1970, 400), (1970, 550)])
    s.circle(1970, 555, 32, "+")
    arr(s, 2002, y, 2130, y)
    s.grid(2135, 522, rows=3, cols=7, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="adapted\nfeature_map\n[B,C,H,W]", label_y=70)
    arr(s, 2280, y, 2400, y)
    s.box(2400, 495, 170, 120, ["refine_head /", "downstream", "mask"], fill=LIGHT_BLUE, stroke=BLUE, text_size=18)
    s.save(name)


def boundary_adapter(name: str) -> None:
    s = Svg(2700, 980, "BoundaryAwareAdapter")
    s.text(25, 50, "(d) BoundaryAwareAdapter", size=36, weight="800", anchor="start")
    s.text(420, 86, "Optional wrapper module: enable-boundary-adapter; no fallback path is shown in this figure", size=16, weight="800", fill=BLUE, anchor="start")
    y = 170
    s.grid(90, 135, rows=4, cols=4, cell=20, label="coarse_mask_logits\n[B,1,H,W]", label_y=28)
    s.box(250, 140, 125, 70, "sigmoid")
    s.box(430, 140, 145, 70, ["threshold", ">0.5"])
    s.box(635, 140, 230, 70, "boundary_from_mask")
    s.grid(925, 130, rows=5, cols=4, cell=20, colors=["#111", "#fff", "#555", "#fff"], label="boundary_map\n[B,1,H,W]", label_y=25)
    s.box(1085, 135, 160, 80, ["Conv 1->C/2", "3x3"])
    s.token_row(1295, 145, 8, label="boundary_feat C/2\n[B,C/2,H,W]", label_y=40)
    s.box(1510, 140, 95, 70, "GELU")
    s.box(1665, 135, 160, 80, ["Conv C/2->C", "3x3"])
    s.grid(1875, 125, rows=4, cols=7, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="boundary_features\n[B,C,H,W]", label_y=28)
    s.box(2075, 140, 95, 70, "GELU")
    for x1, x2 in [(170, 250), (375, 430), (575, 635), (865, 925), (1005, 1085), (1245, 1295), (1465, 1510), (1605, 1665), (1825, 1875), (2020, 2075)]:
        arr(s, x1, y, x2, y)
    bent(s, [(2122, 210), (2122, 290), (1700, 290), (1700, 350)])

    s.grid(450, 330, rows=4, cols=4, cell=20, colors=["#111", "#fff", "#fff", "#fff"], label="GT mask (train only)\n[B,1,H,W]", label_y=25)
    s.box(620, 345, 220, 70, "boundary_from_mask")
    s.grid(930, 330, rows=5, cols=4, cell=20, colors=["#111", "#fff", "#555", "#fff"], label="boundary_target\n[B,1,H,W]", label_y=25)
    arr(s, 530, 370, 620, 370)
    arr(s, 840, 370, 930, 370)
    s.text(1300, 350, "train only", size=12)
    bent(s, [(1010, 370), (1210, 370), (1210, 410), (1605, 410), (1605, 390)], dash="12 10")
    s.box(1585, 350, 145, 80, ["Conv C->1", "1x1"])
    s.grid(1785, 350, rows=4, cols=4, cell=20, label="boundary_logits\n[B,1,H,W]", label_y=28)
    s.box(1940, 355, 160, 70, "BCEWithLogits")
    arr(s, 1730, 390, 1785, 390)
    arr(s, 1865, 390, 1940, 390)
    arr(s, 2100, 390, 2160, 390)
    s.text(2185, 390, "boundary_loss", size=18, weight="800", anchor="start")

    s.grid(35, 450, rows=4, cols=6, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="image_features X\n[B,C,H,W]", label_y=28)
    s.circle(690, 555, 34, "concat\n2C", size=18)
    bent(s, [(160, 470), (220, 470), (220, 540), (655, 540)])
    bent(s, [(2122, 210), (2122, 510), (690, 510), (690, 521)])
    arr(s, 724, 555, 805, 555)
    s.box(805, 515, 155, 80, ["Conv 2C->C", "1x1"])
    arr(s, 960, 555, 1015, 555)
    s.box(1015, 520, 95, 70, "GELU")
    arr(s, 1110, 555, 1170, 555)
    s.box(1170, 515, 150, 80, ["Conv C->C", "1x1"])
    arr(s, 1320, 555, 1380, 555)
    s.box(1380, 520, 115, 70, "Sigmoid")
    arr(s, 1495, 555, 1555, 555)
    s.grid(1560, 510, rows=3, cols=7, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="boundary_gate\n[B,C,H,W]", label_y=70)
    bent(s, [(1710, 570), (1710, 710), (1710, 710)])

    bent(s, [(160, 490), (220, 490), (220, 910), (2040, 910), (2040, 800)])
    bent(s, [(160, 470), (220, 470), (220, 735), (300, 735)])
    s.box(300, 700, 120, 70, "LayerNorm")
    arr(s, 420, 735, 480, 735)
    s.box(480, 700, 150, 70, ["Linear C->C/4"])
    arr(s, 630, 735, 720, 735)
    s.token_row(725, 722, 5, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE], label="bottleneck\n[B,C/4,H,W]", label_y=46)
    arr(s, 835, 735, 895, 735)
    s.box(895, 700, 90, 70, "GELU")
    arr(s, 985, 735, 1045, 735)
    s.box(1045, 700, 150, 70, ["Linear C/4->C"])
    arr(s, 1195, 735, 1260, 735)
    s.grid(1265, 710, rows=3, cols=6, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="adapter(X)\n[B,C,H,W]", label_y=68)
    s.circle(1705, 765, 34, "x", size=20)
    bent(s, [(1385, 735), (1668, 735), (1668, 765)])
    bent(s, [(1700, 570), (1700, 730)])
    arr(s, 1739, 765, 1840, 765)
    s.grid(1845, 725, rows=3, cols=7, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="gated_update\n[B,C,H,W]", label_y=70)
    s.circle(2065, 765, 34, "+")
    arr(s, 1990, 765, 2031, 765)
    bent(s, [(2040, 800), (2040, 765), (2031, 765)])
    arr(s, 2099, 765, 2200, 765)
    s.grid(2205, 725, rows=3, cols=7, cell=20, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="enhanced_feature\nX + gate x adapter(X)\n[B,C,H,W]", label_y=70)
    arr(s, 2360, 765, 2460, 765)
    s.box(2460, 705, 180, 120, ["refine_head /", "mask logits"], fill=LIGHT_BLUE, stroke=BLUE, text_size=18)
    s.save(name)


def exemplar_adapter(name: str, width: int = 2850) -> None:
    width = max(width, 3000)
    s = Svg(width, 980, "ExemplarPromptAdapter")
    s.text(25, 50, "(e) ExemplarPromptAdapter", size=36, weight="800", anchor="start")
    s.text(500, 82, "Independent prompt branch: positive + boundary + negative tokens all enter visual_prompt_embed", size=16, weight="800", fill=BLUE, anchor="start")
    s.text(820, 105, "Summary / Gate branch", size=26, weight="800", anchor="start")
    rows = [
        ("query_feat\n[B,C]", Q, 135, "query_summary\n[B,C]", None),
        ("positive_proto\n[B,Kp,C]", POS, 250, "positive_summary\n[B,C]", "reduce\nmean"),
        ("boundary_proto\n[B,Kb,C]", BND, 370, "boundary_summary\n[B,C]", "reduce\nmean"),
        ("negative_proto\n[B,Kn,C]", NEG, 490, "negative_summary\n[B,C]", "reduce\nmean"),
    ]
    for label, color, y, out_label, box_label in rows:
        s.text(30, y + 6, label, size=16, weight="800", anchor="start")
        s.token_row(190, y - 12, 6, colors=[color], size=23)
        if box_label:
            arr(s, 355, y, 485, y)
            s.box(485, y - 30, 100, 60, box_label, text_size=14)
            arr(s, 585, y, 700, y)
        else:
            arr(s, 410, y, 700, y)
        s.token_row(710, y - 11, 9, colors=[color], size=20, label=out_label, label_y=42, label_size=12)
        bent(s, [(900, y), (1000, y), (1000, 305), (1045, 305)], marker=(y == 490))
    s.circle(1100, 305, 42, "concat\n4C", size=18)
    arr(s, 1142, 305, 1215, 305)
    s.box(1215, 270, 125, 70, ["Linear", "4C->C"], text_size=14)
    arr(s, 1340, 305, 1405, 305)
    s.box(1405, 270, 100, 70, "GELU", text_size=14)
    arr(s, 1505, 305, 1570, 305)
    s.box(1570, 270, 125, 70, ["Linear", "C->4"], text_size=14)
    arr(s, 1695, 305, 1765, 305)
    for i, (lab, c) in enumerate([("g_pos", POS), ("g_neg", NEG), ("g_bnd", BND), ("g_sup", "#0060a8")]):
        x = 1765 + i * 75
        s.rect(x, 288, 35, 35, fill=c, stroke=BLACK, sw=1, rx=0)
        s.text(x + 17, 270, lab, size=11)
        if i < 3:
            bent(s, [(x + 17, 323), (x + 17, 580 + i * 120), (1738, 615 + i * 120)], color=[POS, NEG, BND][i])
    bent(s, [(1990, 305), (2060, 305), (2060, 305)], color=BLUE, dash="12 9")
    s.text(2070, 305, "suppression_gate", size=18, weight="800", anchor="start")

    token_rows = [
        ("positive_proto\n[B,Kp,C]", POS, 620, ["Linear", "C->C"], ["Linear", "C->4C"], "positive tokens x4\n[B,4,C]", 4),
        ("boundary_proto\n[B,Kb,C]", BND, 740, ["Linear", "C->C"], ["Linear", "C->2C"], "boundary tokens x2\n[B,2,C]", 2),
        ("negative_proto\n[B,Kn,C]", NEG, 860, ["Linear", "C->C"], ["Linear", "C->2C"], "negative tokens x2\n[B,2,C]", 2),
    ]
    for label, color, y, l1, l2, out_label, count in token_rows:
        s.text(30, y + 2, label, size=16, weight="800", anchor="start")
        s.token_row(190, y - 12, 5, colors=[color], size=23)
        arr(s, 330, y, 405, y)
        s.box(405, y - 32, 110, 64, l1, text_size=13)
        arr(s, 515, y, 575, y)
        s.token_row(580, y - 10, 5, colors=[color], size=20)
        arr(s, 700, y, 760, y)
        s.box(760, y - 32, 90, 64, "GELU", text_size=13)
        arr(s, 850, y, 915, y)
        s.box(915, y - 32, 120, 64, l2, text_size=13)
        arr(s, 1035, y, 1110, y)
        s.token_row(1115, y - 10, 5, colors=[color], size=20)
        arr(s, 1230, y, 1295, y)
        s.box(1295, y - 32, 100, 64, "reshape", text_size=13)
        arr(s, 1395, y, 1460, y)
        s.token_row(1465, y - 10, count, colors=[color], size=20, label=out_label, label_y=42, label_size=12)
        s.circle(1730, y, 32, "x", size=20)
        arr(s, 1545, y, 1698, y)
        bent(s, [(1760, y), (1865, 770), (1930, 770)], color=BLUE)
    s.circle(1935, 770, 55, "concat:\npos | bnd | neg", size=13)
    arr(s, 1990, 770, 2095, 770)
    s.token_row(2100, 758, 9, colors=[POS, POS, POS, POS, BND, BND, NEG, NEG], size=22, label="prompt_tokens [B,8,C]", label_y=45)
    s.text(2100, 835, "token order: positive x4 + boundary x2 + negative x2", size=13, weight="800", fill=BLUE, anchor="start")
    arr(s, 2310, 770, 2385, 770)
    s.box(2385, 735, 115, 70, "LayerNorm", text_size=14)
    arr(s, 2500, 770, 2585, 770)
    s.token_row(2590, 758, 8, colors=[POS, POS, POS, BND, BND, NEG, NEG, NEG], size=22, label="visual prompt embeddings\n[B,8,C]", label_y=45)
    arr(s, 2780, 770, 2820, 770)
    s.box(2820, 708, 95, 124, ["SAM3", "prompt", "encoder"], fill=LIGHT_BLUE, stroke=BLUE, text_size=15)
    s.save(name)


def rssda_adapter(name: str, width: int = 3400) -> None:
    s = Svg(width, 1500, "RSS-DA Adapter Combination")
    s.text(30, 48, "(f) RSS-DA Adapter Combination", size=36, weight="800", anchor="start")
    s.text(640, 76, "Independent retrieval branch: builds retrieval_prior for memory, decoder feature bias, and mask-logit bias", size=16, weight="800", fill=BLUE, anchor="start")
    panel_w = min(width - 820, 2280)

    s.card(70, 60, panel_w, 270, "1  PrototypeRetriever", fill="white", stroke=BLUE, title_fill=BLUE)
    y = 155
    s.grid(110, 130, rows=3, cols=4, cell=20, label="query_feature\n[B,C,H,W]", label_y=26)
    s.box(250, 120, 70, 70, "GAP", text_size=13)
    arr(s, 190, y, 250, y)
    arr(s, 320, y, 380, y)
    s.token_row(385, y - 11, 8, label="query_global\n[B,C]", label_y=38)
    s.box(580, 120, 100, 70, ["Linear", "C->C"], text_size=13)
    s.box(730, 120, 75, 70, "GELU", text_size=13)
    s.box(850, 120, 100, 70, ["Linear", "C->C"], text_size=13)
    arr(s, 555, y, 580, y)
    arr(s, 680, y, 730, y)
    arr(s, 805, y, 850, y)
    arr(s, 950, y, 1010, y)
    s.token_row(1015, y - 11, 8, label="projected_query\n[B,C]", label_y=38)
    bent(s, [(1165, y), (1245, y), (1245, 110), (1340, 110)])
    bent(s, [(1165, y), (1245, y), (1245, 245), (1340, 245)])
    s.token_row(1280, 86, 8, colors=[POS], label="positive_bank\n[Np,C]", label_y=36)
    s.token_row(1280, 220, 8, colors=[NEG], label="negative_bank\n[Nn,C]", label_y=36)
    for yy, color, feat, weight, proto in [
        (110, POS, "positive_features\n[B,Kp,C]", "pos_weights\n[B,Kp,1]", "positive_prototype\n[B,C]"),
        (245, NEG, "negative_features\n[B,Kn,C]", "neg_weights\n[B,Kn,1]", "negative_prototype\n[B,C]"),
    ]:
        s.box(1435, yy - 30, 110, 60, ["top-k", "cosine", "retrieval"], text_size=13)
        arr(s, 1380, yy, 1435, yy)
        arr(s, 1545, yy, 1615, yy)
        s.token_row(1620, yy - 11, 8, colors=[color if color == NEG else "#1679b9"], label=feat, label_y=38)
        arr(s, 1800, yy, 1875, yy)
        s.token_row(1880, yy - 10, 7, colors=["#dbe5ef" if color == POS else "#eadff6"], label=weight, label_y=38)
        arr(s, 2030, yy, 2110, yy)
        s.box(2110, yy - 30, 115, 60, ["weighted", "average"], text_size=13)
        arr(s, 2225, yy, 2310, yy)
        s.token_row(2315, yy - 11, 9, colors=[color], label=proto, label_y=38)

    s.card(70, 360, panel_w, 300, "2  SimilarityHeatmapBuilder", fill="white", stroke=BLUE, title_fill=BLUE)
    for yy, color, prefix in [(455, "#1679b9", "positive"), (575, NEG, "negative")]:
        s.grid(110, yy - 25, rows=3, cols=4, cell=20, label="query_feature")
        s.token_row(250, yy - 10, 6, colors=[color], label=f"{prefix}_features")
        arr(s, 375, yy, 440, yy)
        s.box(440, yy - 30, 110, 60, ["cosine", "similarity", "map"], text_size=13)
        arr(s, 550, yy, 620, yy)
        s.grid(625, yy - 25, rows=2, cols=9, cell=18, colors=["#eef3f8", "#ff5e65", "#a7cde8", "#f4d2e8"], label=f"{prefix}_similarity\n[B,K,H,W]", label_y=26)
        arr(s, 795, yy, 870, yy)
        s.box(870, yy - 30, 100, 60, ["weighted", "fuse"], text_size=13)
        arr(s, 970, yy, 1040, yy)
        s.grid(1045, yy - 25, rows=2, cols=9, cell=18, colors=["#eef3f8", "#ff5e65", "#a7cde8", "#f4d2e8"], label=f"{prefix}_heatmap\n[B,1,H,W]", label_y=26)
    s.circle(1310, 520, 36, "2\nmaps", size=18)
    bent(s, [(1210, 455), (1275, 455), (1275, 505)])
    bent(s, [(1210, 575), (1275, 575), (1275, 535)])
    arr(s, 1346, 520, 1425, 520)
    s.box(1425, 490, 90, 60, ["Conv", "2->1", "1x1"], text_size=13)
    arr(s, 1515, 520, 1600, 520)
    s.grid(1605, 495, rows=2, cols=9, cell=18, label="fused_similarity\n[B,1,H,W]", label_y=26)
    arr(s, 1775, 520, 1880, 520)
    s.box(1880, 490, 120, 60, ["sigmoid", "temperature"], text_size=13)
    arr(s, 2000, 520, 2115, 520)
    s.grid(2120, 495, rows=2, cols=9, cell=18, colors=["#ffe2e2", "#ff6b7a", "#f2f2f2"], label="spatial_prior\n[B,1,H,W]", label_y=26)

    s.card(70, 690, panel_w, 420, "3  RetrievalSpatialSemanticAdapter + GatedRetrievalFusion", fill="white", stroke=BLUE, title_fill=BLUE)
    s.grid(110, 745, rows=2, cols=6, cell=18, colors=["#eef3f8", "#ff5e65", "#a7cde8"], label="positive_heatmap")
    s.grid(110, 820, rows=2, cols=6, cell=18, colors=["#eee4ff", "#d5b5ef"], label="negative_heatmap")
    bent(s, [(225, 760), (290, 760), (290, 800), (330, 800)])
    bent(s, [(225, 835), (290, 835), (290, 800), (330, 800)])
    s.box(330, 770, 135, 60, ["spatial_fusion", "Conv 2->C/4"], text_size=13)
    arr(s, 465, 800, 520, 800)
    s.box(520, 770, 80, 60, "GELU", text_size=13)
    arr(s, 600, 800, 660, 800)
    s.box(660, 770, 95, 60, ["Conv", "C/4->1"], text_size=13)
    arr(s, 755, 800, 820, 800)
    s.grid(825, 785, rows=2, cols=8, cell=16, colors=["#ffe2e2", "#ff6b7a", "#f2f2f2"], label="spatial_bias")

    entries = [
        ("query_feature", Q, 910, ["query_proj", "Conv1x1"]),
        ("positive_features", POS, 980, ["prototype_proj", "Linear"]),
        ("negative_features", NEG, 1050, ["prototype_proj", "Linear"]),
    ]
    for lab, color, yy, boxlab in entries:
        s.grid(110, yy - 23, rows=2, cols=4, cell=18, colors=[color, "#eef3f8", "#ff5e65"] if color == Q else [color], label=lab)
        arr(s, 190, yy, 250, yy)
        s.box(250, yy - 30, 110, 60, boxlab, text_size=12)
        arr(s, 360, yy, 425, yy)
        s.box(425, yy - 30, 75, 60, "GELU", text_size=13)
        arr(s, 500, yy, 560, yy)
        s.box(560, yy - 30, 95, 60, "Linear" if color != Q else "Conv1x1", text_size=13)
        arr(s, 655, yy, 730, yy)
        s.token_row(735, yy - 10, 7, colors=[color], label={"query_feature": "query_feature'", "positive_features": "positive_context_map", "negative_features": "negative_context_map"}[lab])
    s.circle(1000, 985, 36, "concat\n3C", size=16)
    for yy in [910, 980, 1050]:
        bent(s, [(865, yy), (940, 985), (964, 985)])
    arr(s, 1036, 985, 1125, 985)
    s.box(1125, 955, 115, 60, ["gate", "Conv 3C->C"], text_size=13)
    arr(s, 1240, 985, 1335, 985)
    s.box(1335, 955, 90, 60, "Sigmoid", text_size=13)
    arr(s, 1425, 985, 1515, 985)
    s.grid(1520, 960, rows=2, cols=7, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="fusion_gate")
    s.circle(1760, 985, 36, "concat\n3C", size=16)
    arr(s, 1645, 985, 1724, 985)
    bent(s, [(940, 800), (1760, 800), (1760, 949)])
    arr(s, 1796, 985, 1875, 985)
    s.box(1875, 955, 120, 60, ["delta_proj", "Conv 3C->C"], text_size=13)
    arr(s, 1995, 985, 2070, 985)
    s.box(2070, 955, 75, 60, "GELU", text_size=13)
    arr(s, 2145, 985, 2225, 985)
    s.box(2225, 955, 85, 60, ["Conv", "C->C"], text_size=13)
    arr(s, 2310, 985, 2395, 985)
    s.grid(2400, 960, rows=2, cols=7, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="fused_delta")

    s.grid(2080, 740, rows=2, cols=9, cell=18, colors=["#ffe2e2", "#ff6b7a", "#f2f2f2"], label="baseline_mask_logits")
    s.box(2290, 750, 90, 55, "confidence", text_size=12)
    s.box(2420, 750, 80, 55, "entropy", text_size=12)
    s.box(2540, 745, 125, 60, ["boundary", "uncertainty"], text_size=12)
    s.grid(2700, 755, rows=2, cols=8, cell=16, colors=["#ffe2e2", "#ff6b7a"], label="policy_gate")
    arr(s, 2245, 775, 2290, 775)
    arr(s, 2380, 775, 2420, 775)
    arr(s, 2500, 775, 2540, 775)
    arr(s, 2665, 775, 2700, 775)
    s.circle(2630, 985, 32, "x", size=18)
    bent(s, [(2795, 775), (2795, 900), (2630, 900), (2630, 953)], dash="12 9")
    arr(s, 2535, 985, 2598, 985)
    arr(s, 2662, 985, 2740, 985)
    s.box(2740, 955, 100, 60, ["alpha", "residual"], text_size=13)
    arr(s, 2840, 985, 2925, 985)
    s.grid(2930, 960, rows=2, cols=7, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="localized_delta\n[B,C,H,W]")

    s.card(70, 1140, panel_w, 300, "4  Retrieval Prior Injection / Output", fill="white", stroke=BLUE, title_fill=BLUE)
    s.text(500, 1194, "retrieval_prior outputs used by Sam3TensorForwardWrapper", size=15, weight="800", fill=BLUE, anchor="start")
    s.token_row(110, 1220, 6, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="localized_delta")
    for yy, lab, color in [(1225, "encoder_memory_bias\n[B,C,H,W]", TEAL), (1305, "decoder_feature_bias_map\n[B,C,H,W]", POS), (1385, "mask_logit_bias_map\n[B,1,H,W]", RED)]:
        bent(s, [(250, 1230), (300, yy), (355, yy)])
        s.grid(360, yy - 18, rows=2, cols=8, cell=16, colors=[color, "#e4f8ea" if color == POS else "#ffe2e2"], label=lab, label_y=22)
    s.grid(790, 1210, rows=2, cols=8, cell=16, colors=["#d7f5e5", POS], label="positive_context_map")
    s.grid(790, 1300, rows=2, cols=8, cell=16, colors=["#eee4ff", NEG], label="negative_context_map")
    s.circle(1010, 1270, 30, "-", size=18)
    bent(s, [(930, 1225), (980, 1270)])
    bent(s, [(930, 1315), (980, 1270)])
    arr(s, 1040, 1270, 1110, 1270)
    s.box(1110, 1240, 110, 60, ["semantic", "prototype"], text_size=13)
    arr(s, 1220, 1270, 1315, 1270)
    s.grid(1320, 1248, rows=2, cols=9, cell=16, colors=["#d7f5e5", POS], label="semantic_prototype_map\n[B,C,H,W]", label_y=24)
    s.grid(1320, 1360, rows=2, cols=9, cell=16, colors=["#ffe2e2", "#ff6b7a"], label="spatial_bias_map\n[B,1,H,W]", label_y=24)
    wrapper_x = min(width - 560, 2520)
    s.box(wrapper_x, 1210, 360, 130, ["SAM3 Tensor Forward", "add to encoder memory", "add to mask logits"], fill=LIGHT_BLUE, stroke=BLUE, text_size=18)
    for yy in [1225, 1305, 1385, 1270, 1375]:
        bent(s, [(1510, yy), (wrapper_x, yy)])
    arr(s, wrapper_x + 360, 1275, wrapper_x + 455, 1275)
    s.grid(wrapper_x + 470, 1245, rows=5, cols=4, cell=18, colors=["#111", "#fff", "#fff", "#fff"], label="retrieval-adapted\nmask\n[B,1,H,W]", label_y=28)
    s.save(name)


def overall_insertion_map() -> None:
    s = Svg(3600, 1900, "Overall MedEx-SAM3 Insertion Map")
    s.text(40, 55, "Overall MedEx-SAM3 Insertion Map", size=42, weight="800", anchor="start")
    s.text(45, 95, "Where LoRA, Exemplar Prompt, RSS-DA, MedicalImageAdapter, BoundaryAwareAdapter, and refine_head enter the SAM3 pipeline", size=20, weight="800", fill=BLUE, anchor="start")
    s.text(45, 123, "Semantics: LoRA + refine_head are the base branch; Medical/Boundary adapters are optional wrapper modules; Exemplar Prompt and RSS-DA are independent enhancement branches.", size=16, weight="800", fill=BLUE, anchor="start")

    s.card(80, 150, 3140, 560, "A. Official SAM3 Main Flow", fill="#eaf6ff", stroke=BLUE, title_fill=BLUE)
    y = 380
    s.grid(130, 335, rows=4, cols=4, cell=20, label="Image\n[B,3,H,W]", label_y=26)
    s.box(300, 326, 190, 110, ["Forward Image", "Backbone"], fill=LIGHT_BLUE, stroke=BLUE, text_size=19)
    arr(s, 210, y, 300, y)
    arr(s, 490, y, 570, y)
    s.token_row(575, y - 12, 8, label="image features")
    arr(s, 760, y, 840, y)
    s.box(840, 326, 190, 110, "Encode Prompt", fill=LIGHT_BLUE, stroke=BLUE, text_size=19)
    arr(s, 1030, y, 1110, y)
    s.box(1110, 326, 190, 110, "Run Encoder", fill=LIGHT_BLUE, stroke=BLUE, text_size=19)
    arr(s, 1300, y, 1370, y)
    s.box(1370, 336, 210, 90, ["Apply RSS-DA", "Prior to Memory"], fill=RED_FILL, stroke=BLACK, text_size=18)
    arr(s, 1580, y, 1660, y)
    s.box(1660, 326, 190, 110, "Run Decoder", fill=LIGHT_BLUE, stroke=BLUE, text_size=19)
    arr(s, 1850, y, 1930, y)
    s.box(1930, 326, 230, 110, ["Run Segmentation", "Heads"], fill=LIGHT_BLUE, stroke=BLUE, text_size=19)
    arr(s, 2160, y, 2240, y)
    s.grid(2240, 345, rows=3, cols=8, cell=17, label="SAM3 mask_logits")
    arr(s, 2405, y, 2495, y)
    s.box(2495, 336, 185, 90, ["Apply RSS-DA", "Logit Bias"], fill=RED_FILL, stroke=BLACK, text_size=18)
    arr(s, 2680, y, 2760, y)
    s.grid(2760, 345, rows=3, cols=8, cell=17, label="conditioned\nmask_logits")
    s.text(130, 585, "Text / box prompt", size=15, weight="800", fill=BLUE, anchor="start")
    s.box(300, 520, 190, 80, "Forward Text", fill=LIGHT_BLUE, stroke=BLUE, text_size=15)
    arr(s, 490, 560, 760, 560)
    bent(s, [(760, 560), (760, 420), (840, 420)])
    s.token_row(145, 640, 6, colors=[ORANGE], label="geometric prompt")
    bent(s, [(275, 650), (745, 650), (745, 395), (840, 395)])

    s.card(540, 710, 980, 270, "B. Stage-A LoRA: inside SAM3 Linear layers", fill="#fffbc7", stroke="#c49a00", title_fill="#c49a00")
    s.text(575, 780, "Vision Encoder late 1/3 blocks", size=18, weight="800", anchor="start")
    for i in range(12):
        s.rect(575 + i * 32, 805, 28, 42, fill="#858585" if i < 8 else POS, stroke=BLACK, sw=1, rx=0)
    s.text(860, 862, "late blocks only", size=12, fill=BLUE)
    s.box(1010, 760, 230, 105, ["q_proj / v_proj / qkv", "attn.proj / out_proj", ".proj -> LoRALinear"], fill="#fff000", text_size=15)
    s.text(1260, 810, "Mask Decoder\nattention/proj\nLinear -> LoRA", size=15, weight="800", anchor="start")
    bent(s, [(410, 520), (410, 500), (925, 500), (925, 326)], color="#c49a00", dash="14 10")
    bent(s, [(1370, 760), (1370, 650), (1760, 650), (1760, 436)], color="#c49a00", dash="14 10")

    s.card(780, 1010, 850, 300, "C. ExemplarPromptAdapter: prompt-side insertion", fill=PURPLE, stroke=PURPLE_STROKE, title_fill=PURPLE_STROKE)
    s.token_row(835, 1095, 6, colors=[POS], label="positive_proto")
    s.token_row(835, 1175, 4, colors=[BND], label="boundary_proto")
    s.token_row(835, 1255, 4, colors=[NEG], label="negative_proto")
    s.box(1080, 1090, 170, 80, ["Projectors +", "Fusion Gate"], fill="#fff7fb", text_size=14)
    s.token_row(1305, 1125, 9, colors=[POS, POS, POS, BND, BND, NEG, NEG, TEAL], size=20, label="visual prompt tokens\n[B,8,C]", label_y=48)
    bent(s, [(1015, 1105), (1080, 1130)], color=PURPLE_STROKE)
    bent(s, [(1015, 1185), (1080, 1130)], color=PURPLE_STROKE)
    bent(s, [(1015, 1265), (1080, 1130)], color=PURPLE_STROKE)
    arr(s, 1250, 1130, 1305, 1130, color=PURPLE_STROKE)
    bent(s, [(820, 1010), (500, 1010), (500, 660), (820, 660), (820, 436)], color=PURPLE_STROKE, dash="14 10")
    s.text(1180, 1260, "Inserted as visual_prompt_embed before Encode Prompt", size=14, weight="800", fill=PURPLE_STROKE)
    s.text(1180, 1284, "prompt tokens contain positive, boundary, and negative prototypes", size=13, weight="800", fill=PURPLE_STROKE)

    s.card(1680, 745, 1180, 520, "D. RSS-DA: retrieval prior at memory and logit boundaries", fill=RED_FILL, stroke=RED, title_fill=RED)
    s.grid(1730, 835, rows=3, cols=5, cell=15, label="query_feature")
    s.box(1900, 818, 155, 70, ["Prototype", "Retriever"], fill="#fff7f7", text_size=13)
    s.token_row(2090, 820, 5, colors=[POS], label="positive")
    s.token_row(2090, 890, 5, colors=[NEG], label="negative")
    s.box(2260, 818, 185, 70, ["Similarity", "Heatmap Builder"], fill="#fff7f7", text_size=13)
    s.grid(2490, 835, rows=3, cols=6, cell=15, label="spatial prior")
    s.box(2685, 806, 145, 90, ["Retrieval", "Spatial-Semantic", "Adapter"], fill="#fff7f7", text_size=13)
    arr(s, 1805, 855, 1900, 855)
    arr(s, 2055, 855, 2260, 855)
    arr(s, 2445, 855, 2685, 855)
    s.text(1730, 1000, "retrieval_prior outputs", size=18, weight="800", fill=RED, anchor="start")
    s.text(1730, 1024, "applied to encoder memory / decoder feature bias / mask logits", size=13, weight="800", fill=RED, anchor="start")
    s.token_row(1735, 1040, 8, colors=[TEAL], label="encoder_memory_bias")
    s.token_row(1735, 1110, 8, colors=[POS], label="decoder_feature_bias_map")
    s.token_row(1735, 1180, 8, colors=[RED], label="mask_logit_bias_map")
    s.token_row(2220, 1040, 8, colors=[BND], label="semantic_prototype_map")
    s.token_row(2220, 1110, 8, colors=[RED], label="spatial_bias_map")
    bent(s, [(1475, 745), (1475, 690), (1880, 690), (1880, 426)], color=RED, dash="14 10")
    s.text(1510, 685, "before Run Decoder", size=12, fill=RED, anchor="start")
    bent(s, [(1880, 1265), (1880, 1300), (2585, 1300), (2585, 426)], color=RED, dash="14 10")
    s.text(2595, 1298, "after SAM3 mask logits", size=12, fill=RED, anchor="start")

    s.card(100, 1320, 3060, 420, "E. MedEx Wrapper refinement after SAM3 forward", fill=GREEN_FILL, stroke=GREEN, title_fill=GREEN)
    s.grid(185, 1460, rows=4, cols=5, cell=18, colors=[TEAL, "#2487c7", POS, AMBER, ORANGE, NEG], label="image_embeddings\nfeature_map", label_y=24)
    s.box(400, 1455, 155, 80, ["Resolve", "Feature Map"], fill="#fff", text_size=15)
    s.box(645, 1445, 210, 100, ["Medical Image", "Adapter"], fill=GREEN_FILL, stroke=GREEN, text_size=19)
    s.box(975, 1445, 230, 100, ["Boundary Aware", "Adapter"], fill=GREEN_FILL, stroke=GREEN, text_size=19)
    s.box(1265, 1455, 145, 80, ["Refine Head", "Conv 1x1"], fill="#fff", text_size=14)
    s.grid(1500, 1465, rows=3, cols=7, cell=17, label="delta")
    s.box(1725, 1455, 110, 80, ["scale", "0.1"], fill="#fff", text_size=14)
    s.circle(2030, 1495, 34, "+")
    s.grid(2175, 1465, rows=3, cols=7, cell=17, label="final mask_logits\n+ 0.1 * delta")
    s.box(2440, 1455, 110, 80, "Sigmoid", fill="#fff", text_size=14)
    s.grid(2630, 1455, rows=4, cols=4, cell=20, colors=["#111", "#fff", "#fff", "#fff"], label="Final Mask\n[B,1,H,W]", label_y=28)
    for a, b in [(285, 400), (555, 645), (855, 975), (1205, 1265), (1410, 1500), (1625, 1725), (1835, 1996), (2064, 2175), (2300, 2440), (2550, 2630)]:
        arr(s, a, 1495, b, 1495)
    s.grid(925, 1615, rows=3, cols=6, cell=17, label="coarse mask_logits")
    bent(s, [(1018, 1615), (1088, 1570), (1088, 1545)], color=GREEN, dash="14 10")
    s.text(1095, 1655, "uses coarse mask; GT mask only for train-time boundary_loss", size=12, fill=GREEN, anchor="start")
    bent(s, [(2980, 380), (2980, 1385), (2050, 1385), (2050, 1462)], color=GREEN, dash="14 10")
    s.grid(1988, 1345, rows=3, cols=6, cell=17, label="conditioned\nmask_logits")
    s.text(80, 1820, "Key order: ExemplarPromptAdapter enters before Encode Prompt; RSS-DA enters after Run Encoder and after mask_logits; Medical/Boundary/refine_head operate in the MedEx wrapper after SAM3 forward.", size=20, weight="800", fill=BLUE, anchor="start")
    s.text(80, 1850, "Do not read this as one default all-on training path; it is a compositional map of implemented branches and insertion points.", size=17, weight="800", fill=BLUE, anchor="start")
    s.save("05_final_Overall_MedEx_SAM3_insertion_map.svg")


def overall_io_framework() -> None:
    s = Svg(3200, 1980, "MedExSAM3 Overall Framework")
    s.text(55, 80, "MedExSAM3 Overall Framework", size=54, weight="800", anchor="start")
    s.text(58, 120, "Inputs, outputs, and insertion points of LoRA, ExemplarPromptAdapter, RSS-DA, MedicalImageAdapter, BoundaryAwareAdapter, and refine_head", size=25, weight="800", fill=BLUE, anchor="start")
    s.text(58, 142, "Base path, optional wrapper adapters, and independent retrieval/prompt branches are separated by colored regions.", size=15, weight="800", fill=BLUE, anchor="start")

    s.card(70, 150, 950, 640, "(A) Prompt Inputs and Prompt Tokens", fill="#f2fbff", stroke=BLACK, dash="18 10", title_fill=BLACK)
    s.box(250, 235, 220, 80, ["Text / Point / Box", "Prompt"], fill="#fff9ee", stroke=ORANGE, shadow=True, text_size=22)
    s.token_row(540, 260, 8, colors=[BND], label="text + geometry tokens", size=18)
    arr(s, 470, 275, 540, 275)
    s.box(305, 420, 305, 125, ["ExemplarPromptAdapter", "in: exemplar image + annotation", "out: visual prompt tokens"], fill=PURPLE, stroke=PURPLE_STROKE, shadow=True, text_size=17)
    s.grid(145, 540, rows=3, cols=4, cell=18, label="region annotation")
    s.token_row(690, 465, 9, colors=[POS, POS, NEG, POS, TEAL, POS, NEG, POS], size=20, label="visual_prompt_embed [B,Nv,C]")
    bent(s, [(230, 445), (305, 462)], color=PURPLE_STROKE)
    bent(s, [(230, 575), (305, 510)], color=PURPLE_STROKE)
    arr(s, 610, 482, 690, 482, color=PURPLE_STROKE)
    s.box(735, 625, 210, 74, ["prompt token set", "for Encode Prompt"], fill="#fff", stroke=BLUE, text_size=16)
    bent(s, [(800, 485), (820, 625)], color=PURPLE_STROKE)
    bent(s, [(725, 275), (970, 275), (970, 650), (945, 650)])
    s.text(118, 730, "output: prompt tokens and visual_prompt_embed", size=16, weight="800", fill=PURPLE_STROKE, anchor="start")
    s.text(118, 755, "visual prompt includes positive, boundary, and negative prototype tokens", size=14, weight="800", fill=PURPLE_STROKE, anchor="start")

    s.card(70, 835, 950, 360, "(B) Image Encoder", fill="#f2fbff", stroke=BLACK, dash="18 10", title_fill=BLACK)
    s.grid(145, 975, rows=3, cols=4, cell=18, label="image\n[B,3,H,W]")
    s.box(300, 930, 220, 105, "Preprocess", fill="#eeeeee", stroke=BLACK, shadow=True, text_size=22)
    s.box(590, 930, 225, 105, ["Forward Image", "Backbone"], fill=LIGHT_BLUE, stroke=BLUE, shadow=True, text_size=22)
    s.box(640, 860, 135, 45, "LoRA", fill="#fff000", stroke="#c49a00", text_size=17)
    s.token_row(860, 965, 9, label="image_embeddings\n[B,C,h,w]", size=18)
    s.box(695, 1090, 255, 70, ["output: image features", "used by SAM3 and wrapper"], fill="#fff", stroke=BLUE, text_size=15)
    arr(s, 225, 980, 300, 980)
    arr(s, 520, 980, 590, 980)
    arr(s, 815, 980, 860, 980)
    bent(s, [(708, 905), (708, 930)], color="#c49a00")
    bent(s, [(900, 1000), (900, 1090)])
    s.text(118, 1172, "output: image_embeddings for SAM3 decoder and MedEx wrapper", size=16, weight="800", fill=BLUE, anchor="start")
    s.text(118, 1192, "LoRA is inside frozen SAM3; wrapper refine_head remains trainable", size=14, weight="800", fill=BLUE, anchor="start")

    s.card(1080, 150, 665, 640, "(C) Prompt-Image Fusion / Memory", fill="#f2fbff", stroke=BLACK, dash="18 10", title_fill=BLACK)
    s.box(1150, 265, 225, 90, "Encode Prompt", fill=LIGHT_BLUE, stroke=BLUE, shadow=True, text_size=22)
    s.box(1185, 490, 240, 100, "Run Encoder", fill=LIGHT_BLUE, stroke=BLUE, shadow=True, text_size=22)
    s.token_row(1425, 292, 7, label="prompt embeddings")
    s.token_row(1450, 530, 3, label="memory")
    s.box(1535, 498, 165, 130, ["Apply RSS-DA", "Prior to Memory", "in: retrieval_prior", "out: biased memory"], fill=RED_FILL, stroke=RED, shadow=True, text_size=16)
    s.box(1200, 662, 445, 74, ["RSS-DA Adapter side signal: retrieval_prior", "in: query/image features    out: memory/logit bias maps"], fill=RED_FILL, stroke=RED, text_size=16)
    bent(s, [(945, 650), (1080, 650), (1080, 310), (1150, 310)])
    bent(s, [(900, 1160), (900, 1330), (1045, 1330), (1045, 540), (1185, 540)])
    bent(s, [(1375, 310), (1510, 310), (1510, 470), (1285, 470), (1285, 490)])
    arr(s, 1425, 540, 1535, 540)
    bent(s, [(1535, 628), (1535, 662)], color=RED, dash="12 8")

    s.card(1810, 150, 1280, 640, "(D) SAM3 Decoder and Mask Output", fill="#f2fbff", stroke=BLACK, dash="18 10", title_fill=BLACK)
    s.text(1865, 230, "input: prior-biased memory + prompt embeddings", size=16, weight="800", fill=BLUE, anchor="start")
    s.box(1915, 405, 180, 80, "LoRA", fill="#fff000", stroke="#c49a00", text_size=17)
    s.box(1875, 485, 220, 105, "Run Decoder", fill=LIGHT_BLUE, stroke=BLUE, shadow=True, text_size=22)
    s.box(2170, 485, 280, 105, ["Run Segmentation", "Heads"], fill=LIGHT_BLUE, stroke=BLUE, shadow=True, text_size=22)
    s.grid(2495, 505, rows=3, cols=5, cell=18, label="SAM3 mask_logits\n[B,1,H,W]")
    s.box(2690, 485, 280, 105, ["Apply RSS-DA", "Logit Bias", "in: mask_logits + bias", "out: conditioned logits"], fill=RED_FILL, stroke=RED, shadow=True, text_size=17)
    s.grid(3000, 505, rows=3, cols=5, cell=18, label="conditioned\nmask_logits")
    arr(s, 1700, 560, 1875, 540, color=RED)
    arr(s, 2095, 540, 2170, 540)
    arr(s, 2450, 540, 2495, 540)
    arr(s, 2605, 540, 2690, 540)
    arr(s, 2970, 540, 3000, 540, color=RED)
    bent(s, [(2820, 1230), (2820, 590)], color=RED, dash="12 8")
    s.text(2385, 725, "output: conditioned_mask_logits", size=16, weight="800", fill=RED, anchor="start")

    s.card(70, 1260, 3040, 560, "(E) MedEx Wrapper Refinement after SAM3 Forward", fill=GREEN_FILL, stroke=GREEN, dash="18 10", title_fill=GREEN)
    s.grid(185, 1465, rows=3, cols=5, cell=18, label="image_embeddings\nfeature_map")
    s.box(365, 1440, 190, 95, ["Resolve", "Feature Map"], fill="#fff", shadow=True, text_size=20)
    s.box(650, 1420, 255, 115, ["MedicalImageAdapter", "in: feature_map", "out: adapted features"], fill=GREEN_FILL, stroke=GREEN, shadow=True, text_size=16)
    s.box(1000, 1420, 305, 115, ["BoundaryAwareAdapter", "in: features + coarse mask", "out: boundary-aware features"], fill=GREEN_FILL, stroke=GREEN, shadow=True, text_size=16)
    s.box(1415, 1430, 185, 95, ["refine_head", "in: refined features", "out: delta logits"], fill=GREEN_FILL, stroke=GREEN, shadow=True, text_size=16)
    s.grid(1650, 1460, rows=3, cols=5, cell=18, label="delta\n[B,1,H,W]")
    s.box(1830, 1440, 145, 95, ["scale", "0.1"], fill="#fff", shadow=True, text_size=20)
    s.circle(2135, 1490, 48, "+", size=36)
    s.grid(2280, 1460, rows=3, cols=5, cell=18, label="final_mask_logits\n= mask_logits + 0.1*delta")
    s.box(2525, 1440, 150, 95, "Sigmoid", fill="#fff", shadow=True, text_size=22)
    s.grid(2790, 1460, rows=3, cols=5, cell=18, label="Final Mask\n[B,1,H,W]")
    for a, b in [(275, 365), (555, 650), (905, 1000), (1305, 1415), (1600, 1650), (1745, 1830), (1975, 2090), (2183, 2280), (2395, 2525), (2675, 2790)]:
        arr(s, a, 1490, b, 1490, color=GREEN if a >= 905 and a < 2183 else BLUE)
    bent(s, [(3070, 560), (3070, 1220), (2050, 1220), (2050, 1445)], color=GREEN, dash="12 8")
    s.grid(2005, 1300, rows=3, cols=4, cell=18, label="conditioned\nmask_logits")
    s.grid(1095, 1680, rows=3, cols=4, cell=18, label="coarse mask logits\n(+ GT mask train only)")
    bent(s, [(1130, 1680), (1130, 1580), (1130, 1535)], color=GREEN, dash="12 8")
    s.text(1170, 1625, "side input", size=15, fill=GREEN, anchor="start")
    s.text(1450, 1850, "output: final binary mask", size=16, weight="800", fill=GREEN, anchor="start")
    s.text(78, 1930, "Key route: Image/Prompt inputs -> SAM3 encoder/decoder -> conditioned mask_logits -> MedEx wrapper refinement -> final mask.", size=26, weight="800", fill=BLUE, anchor="start")
    s.save("06_final_MedExSAM3_overall_IO_framework.svg")


def main() -> None:
    stage_a_lora()
    medical_image_adapter(2600, 760, "01_final_MedicalImageAdapter.svg")
    medical_image_adapter(760, 2400, "01_MedicalImageAdapter_detailed.svg", portrait=True)
    boundary_adapter("02_final_BoundaryAwareAdapter_no_fallback.svg")
    boundary_adapter("02_BoundaryAwareAdapter_detailed_no_fallback.svg")
    exemplar_adapter("03_final_ExemplarPromptAdapter.svg", width=2850)
    exemplar_adapter("03_ExemplarPromptAdapter_detailed.svg", width=2700)
    rssda_adapter("04_final_RSSDA_Adapter_Combination.svg", width=3400)
    rssda_adapter("04_RSSDA_Adapter_Combination_detailed.svg", width=3100)
    overall_insertion_map()
    overall_io_framework()


if __name__ == "__main__":
    main()
