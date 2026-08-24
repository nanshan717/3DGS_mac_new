from PIL import Image, ImageDraw, ImageFont
import math
import os
import platform
import glob

# ============================================================
# 配置区 —— 在这里自定义字体、颜色、文字等
# ============================================================

OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 字体配置 ----
# 按优先级排列的字体文件路径列表，第一个找到的会被使用
# Windows 常见字体在 C:/Windows/Fonts/ 下
# 你可以改成任何你喜欢的 .ttf/.otf/.ttc 字体文件路径
FONT_PATHS = {
    "sans": [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑（支持中文）
        "C:/Windows/Fonts/simhei.ttf",     # 黑体（支持中文）
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "sans_bold": [
        "C:/Windows/Fonts/arialbd.ttf",    # Arial Bold
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑 Bold
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "serif": [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/simsun.ttc",     # 宋体（支持中文）
        "/System/Library/Fonts/Times New Roman.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
    "serif_bold": [
        "C:/Windows/Fonts/timesbd.ttf",
        "/System/Library/Fonts/Times New Roman.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ],
}

# 你可以在这里切换字体风格：'sans' 或 'serif'
FONT_STYLE = "sans"

# ---- 字号配置 ----
SIZE_TITLE = 34      # 标题
SIZE_HEADING = 26    # 小标题
SIZE_BODY = 22       # 正文
SIZE_SMALL = 18      # 小字
SIZE_TINY = 15       # 最小字

# ---- 颜色配置 (RGB 元组) ----
INK   = (34, 43, 58)        # 主文字色
MUTED = (90, 103, 122)      # 次要文字色
BLUE  = (43, 108, 176)      # 蓝色
GREEN = (47, 133, 90)       # 绿色
TEAL  = (44, 122, 123)      # 青色
AMBER = (181, 117, 32)      # 琥珀色
RED   = (178, 62, 62)       # 红色
BG    = (250, 252, 253)     # 背景色
CARD  = (255, 255, 255)     # 卡片背景色
LINE  = (184, 196, 207)     # 边框线色

# ---- 图片输出 DPI ----
OUTPUT_DPI = (220, 220)


# ============================================================
# 字体加载函数（跨平台）
# ============================================================

def _find_font_file(candidates):
    """从候选字体路径列表中找到第一个存在的字体文件"""
    for path in candidates:
        if os.path.exists(path):
            return path
        # 也尝试 glob 匹配（某些字体可能有变体名称）
        base, _ = os.path.splitext(path)
        matches = glob.glob(base + ".*")
        if matches:
            return matches[0]
    return None


def _load_font(size, bold=False):
    """加载字体，优先使用 TrueType，失败则回退到默认位图字体"""
    key = "serif_bold" if (bold and FONT_STYLE == "serif") else \
          "serif" if FONT_STYLE == "serif" else \
          "sans_bold" if bold else "sans"

    path = _find_font_file(FONT_PATHS.get(key, FONT_PATHS["sans"]))

    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass

    # 回退：尝试用名称加载（macOS 上可能生效）
    for name in (["Arial Bold", "Arial"] if not bold else ["Arial Bold", "Times New Roman Bold", "Arial", "Times New Roman"]):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue

    print(f"WARNING: 未找到 TrueType 字体，使用默认位图字体。"
          f"请将你的 .ttf 字体路径添加到 FONT_PATHS 中。")
    return ImageFont.load_default()


# ---- 预加载字体对象 ----
F_TITLE = _load_font(SIZE_TITLE, bold=True)
F_H     = _load_font(SIZE_HEADING, bold=True)
F_B     = _load_font(SIZE_BODY, bold=False)
F_S     = _load_font(SIZE_SMALL, bold=False)
F_TINY  = _load_font(SIZE_TINY, bold=False)


# ============================================================
# 绘图工具函数
# ============================================================

def rounded(draw, xy, r, fill, outline=LINE, width=2):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def centered(draw, xy, text, fnt, fill=INK):
    x1, y1, x2, y2 = xy
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=5, align="center")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        (x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2),
        text, font=fnt, fill=fill, spacing=5, align="center"
    )


def arrow(draw, start, end, color=INK, width=4):
    draw.line([start, end], fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 14
    pts = [
        end,
        (end[0] - size * math.cos(ang - 0.45), end[1] - size * math.sin(ang - 0.45)),
        (end[0] - size * math.cos(ang + 0.45), end[1] - size * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=color)


def save(img, name):
    img.save(os.path.join(OUT_DIR, name), dpi=OUTPUT_DPI)


# ============================================================
# 各图表绘制函数
# ============================================================

def pipeline():
    """绘制 pipeline 流程图"""
    img = Image.new("RGB", (1600, 820), BG)
    d = ImageDraw.Draw(img)

    boxes = [
        (70, 170, 310, 310, "Mobile\nmulti-view\nimages", BLUE),
        (390, 170, 630, 310, "Vanilla 3DGS\noptimization", TEAL),
        (710, 170, 950, 310, "Gaussian\ncenters and\nopacity", AMBER),
        (1030, 170, 1270, 310, "z-percentile\n+ opacity\nmask", GREEN),
        (1350, 170, 1550, 310, "Selected\nlower\nGaussians", GREEN),
    ]
    for x1, y1, x2, y2, label, color in boxes:
        rounded(d, (x1, y1, x2, y2), 18, CARD, outline=color, width=4)
        centered(d, (x1 + 12, y1 + 8, x2 - 12, y2 - 8), label, F_B)
    for i in range(len(boxes) - 1):
        arrow(d, (boxes[i][2] + 22, 240), (boxes[i + 1][0] - 22, 240), MUTED, 4)

    rounded(d, (390, 475, 630, 635), 18, CARD, outline=BLUE, width=4)
    centered(d, (405, 485, 615, 625), "RGB loss\nL1 + D-SSIM", F_B)
    rounded(d, (870, 475, 1110, 635), 18, CARD, outline=AMBER, width=4)
    centered(d, (885, 485, 1095, 625), "Learnable\n5 x 5 control\npoint grid", F_B)
    rounded(d, (1190, 475, 1430, 635), 18, CARD, outline=RED, width=4)
    centered(d, (1205, 485, 1415, 625), "BSR loss\nGaussian-to-\nsurface distance", F_B)

    arrow(d, (510, 310), (510, 475), BLUE, 4)
    arrow(d, (990, 475), (1230, 310), AMBER, 4)
    arrow(d, (1450, 310), (1320, 475), GREEN, 4)
    arrow(d, (630, 555), (870, 555), MUTED, 4)
    arrow(d, (1110, 555), (1190, 555), MUTED, 4)

    rounded(d, (455, 690, 1145, 765), 14, (239, 247, 255), outline=BLUE, width=3)
    centered(d, (470, 695, 1130, 760), "Total loss: L = Lrgb + lambda_BSR(t) LBSR", F_B, BLUE)
    save(img, "brgs_pipeline.png")


def surface_prior():
    """绘制 Bernstein-Bézier 曲面先验图"""
    img = Image.new("RGB", (1400, 900), BG)
    d = ImageDraw.Draw(img)
    d.text((55, 35), "Bernstein-Bezier support surface prior", font=F_TITLE, fill=INK)
    d.text((58, 82), "A smooth textile-support surface constrains only the selected near-ground Gaussians.",
           font=F_B, fill=MUTED)

    # Surface mesh
    origin = (470, 420)
    sx, sy = 45, 22
    pts = []
    for i in range(9):
        row = []
        for j in range(9):
            x = origin[0] + (i - j) * sx
            y = origin[1] + (i + j) * sy - 38 * math.sin(i / 8 * math.pi) - 24 * math.cos(j / 8 * math.pi)
            row.append((x, y))
        pts.append(row)

    for i in range(8):
        for j in range(8):
            p = [pts[i][j], pts[i + 1][j], pts[i + 1][j + 1], pts[i][j + 1]]
            shade = 226 + int(18 * (i + j) / 16)
            d.polygon(p, fill=(shade, 239, 234), outline=(131, 172, 151))
    for row in pts:
        d.line(row, fill=(78, 135, 101), width=2)
    for j in range(9):
        d.line([pts[i][j] for i in range(9)], fill=(78, 135, 101), width=2)

    # Control points
    for i in [0, 2, 4, 6, 8]:
        for j in [0, 2, 4, 6, 8]:
            x, y = pts[i][j]
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=AMBER, outline=(120, 74, 18))

    # Selected lower Gaussians
    for idx in range(70):
        i = (idx * 3) % 9
        j = (idx * 5 + 1) % 9
        base = pts[i][j]
        x = base[0] + ((idx % 5) - 2) * 18
        y = base[1] - 10 - ((idx * 7) % 22)
        r = 7 + (idx % 3)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(57, 129, 98), outline=(31, 88, 68))

    # Vegetation/free Gaussians
    for idx in range(45):
        x = 870 + (idx * 43) % 410
        y = 210 + (idx * 71) % 300
        r = 8 + (idx % 4)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(92, 146, 194), outline=(47, 92, 135))

    d.text((920, 145), "Vegetation/background\nnot constrained", font=F_B, fill=BLUE)
    d.text((250, 775), "Selected lower-region Gaussians", font=F_B, fill=GREEN)
    d.text((820, 705), "5 x 5 learnable\ncontrol-point grid", font=F_B, fill=AMBER)
    d.line([(792, 730), (780, 644)], fill=AMBER, width=4)
    d.line([(505, 760), (565, 588)], fill=GREEN, width=4)
    save(img, "bernstein_surface_prior.png")


def schedule():
    """绘制动态正则化调度曲线图"""
    img = Image.new("RGB", (1300, 760), BG)
    d = ImageDraw.Draw(img)
    d.text((55, 38), "Dynamic regularization schedule", font=F_TITLE, fill=INK)
    d.text((58, 84), "aaa Bernstein prior is delayed until coarse 3DGS geometry becomes stable.",
           font=F_B, fill=MUTED)

    x0, y0, x1, y1 = 150, 590, 1180, 160
    d.line([(x0, y0), (x1, y0)], fill=INK, width=4)
    d.line([(x0, y0), (x0, y1)], fill=INK, width=4)
    d.text((x1 - 70, y0 + 25), "Iteration", font=F_B, fill=INK)
    d.text((55, y1 - 20), "lambda_BSR", font=F_B, fill=INK)

    Tw = x0 + 330
    Tr = x0 + 700
    ymax = y1 + 55
    pts = [(x0, y0), (Tw, y0), (Tr, ymax), (x1, ymax)]
    d.line(pts, fill=RED, width=7, joint="curve")
    for x, lab in [(Tw, "T_w"), (Tr, "T_r")]:
        d.line([(x, y0 + 8), (x, y1 - 10)], fill=(210, 218, 226), width=3)
        d.text((x - 18, y0 + 25), lab, font=F_B, fill=MUTED)
    d.line([(x0, ymax), (x1, ymax)], fill=(210, 218, 226), width=2)
    d.text((70, ymax - 10), "lambda_max", font=F_S, fill=MUTED)

    labels = [
        (x0 + 100, y0 + 55, "Warm-up\nno surface pull", BLUE),
        (Tw + 95, y0 - 120, "Ramp\nprogressive constraint", AMBER),
        (Tr + 130, ymax - 120, "Stable stage\nfixed weight", GREEN),
    ]
    for x, y, text, color in labels:
        rounded(d, (x, y, x + 250, y + 90), 16, CARD, outline=color, width=3)
        centered(d, (x + 8, y + 8, x + 242, y + 82), text, F_S, color)
    save(img, "lambda_schedule.png")


def metric_bars():
    """Draw Figure 5 from the updated Bernstein-surface geometry metrics."""
    img = Image.new("RGB", (1500, 920), BG)
    d = ImageDraw.Draw(img)
    d.text((58, 40), "Lower-region geometry metrics", font=F_TITLE, fill=INK)
    

    panels = [
        {
            "title": "Gaussian-to-surface deviation (lower is better)",
            "metric": "GSD",
            
            "unit": "",
            "y": 170,
            "data": [
                ("Flowers", 2.722604, 2.287677),
                ("Treehill", 3.802427, 3.653765),
            ],
        },
        {
            "title": "Bernstein roughness energy (lower is better)",
            "metric": "Roughness",
            "max": 1.5,
            "unit": "",
            "y": 530,
            "data": [
                ("Flowers", 1.289653, 0.536654),
                ("Treehill", 1.271481, 0.800308),
            ],
        },
    ]

    def draw_panel(panel):
        x0, y0, x1, y1 = 90, panel["y"], 1410, panel["y"] + 290
        rounded(d, (x0, y0, x1, y1), 22, CARD, outline=(215, 224, 233), width=2)
        d.text((x0 + 28, y0 + 22), panel["title"], font=F_H, fill=INK)

        axis_x = x0 + 95
        axis_y = y1 - 58
        chart_w = x1 - axis_x - 70
        chart_h = 165
        d.line([(axis_x, axis_y), (axis_x + chart_w, axis_y)], fill=LINE, width=3)
        d.line([(axis_x, axis_y), (axis_x, axis_y - chart_h)], fill=LINE, width=3)

        for tick in [0, 0.5, 1.0]:
            x = axis_x + int(chart_w * tick)
            d.line([(x, axis_y), (x, axis_y - chart_h)], fill=(232, 237, 242), width=2)
            val = panel["max"] * tick
            d.text((x - 18, axis_y + 14), f"{val:.1f}", font=F_TINY, fill=MUTED)

        group_w = chart_w // 2
        bar_h = 38
        for idx, (scene, vanilla, brgs) in enumerate(panel["data"]):
            gx = axis_x + idx * group_w + 70
            gy = axis_y - 132
            d.text((gx, gy - 42), scene, font=F_B, fill=INK)

            for row, (label, value, color) in enumerate([
                ("Vanilla 3DGS", vanilla, BLUE),
                ("BR-GS", brgs, GREEN),
            ]):
                y = gy + row * 58
                bw = int((value / panel["max"]) * (group_w - 165))
                rounded(d, (gx, y, gx + bw, y + bar_h), 8, color, outline=color, width=1)
                d.text((gx + bw + 14, y + 7), f"{value:.3f}", font=F_S, fill=INK)
                d.text((gx - 135, y + 7), label, font=F_TINY, fill=MUTED)

    for panel in panels:
        draw_panel(panel)

    d.rectangle((965, 112, 993, 140), fill=BLUE)
    d.text((1005, 111), "Vanilla 3DGS", font=F_S, fill=INK)
    d.rectangle((1190, 112, 1218, 140), fill=GREEN)
    d.text((1230, 111), "BR-GS", font=F_S, fill=INK)
    save(img, "figure5.png")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print(f"Platform: {platform.system()}")
    print(f"Font style: '{FONT_STYLE}'")

    # Print found font paths for debugging
    for key in ["sans", "sans_bold", "serif", "serif_bold"]:
        path = _find_font_file(FONT_PATHS.get(key, []))
        print(f"  {key}: {'[OK] ' + path if path else '[MISSING]'}")

    print("Generating images...")
    pipeline()
    surface_prior()
    schedule()
    metric_bars()
    print(f"Done! Images saved to: {os.path.abspath(OUT_DIR)}")
