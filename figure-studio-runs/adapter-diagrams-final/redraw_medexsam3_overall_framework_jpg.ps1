Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

$ErrorActionPreference = "Stop"

$OutPath = "C:\Users\yep\Downloads\medexsam3_overall_framework_redrawn_paper.jpg"
$W = 3600
$H = 1850
$Scale = 1.0

function C($hex) {
    return [System.Drawing.ColorTranslator]::FromHtml($hex)
}

$Colors = @{
    Ink = C "#17202A"
    Muted = C "#52616B"
    Blue = C "#1E6AAE"
    BlueLight = C "#EAF4FC"
    BluePale = C "#F4FAFF"
    Yellow = C "#FFF3A3"
    YellowStroke = C "#C99600"
    Purple = C "#7A3BD1"
    PurpleLight = C "#F1E9FF"
    Red = C "#D85050"
    RedLight = C "#FFF0F0"
    Green = C "#14935A"
    GreenLight = C "#EAF8F0"
    Gray = C "#F7F8FA"
    GrayStroke = C "#AAB2BD"
    White = C "#FFFFFF"
    Black = C "#000000"
    Pos = C "#18A957"
    Neg = C "#7A35C5"
    Bnd = C "#F39C12"
    Teal = C "#1199A8"
    Cyan = C "#2386C8"
    Pink = C "#F46D7C"
}

function Font($size, $style = [System.Drawing.FontStyle]::Regular) {
    return New-Object System.Drawing.Font("Arial", $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}

function Pen($color, $width = 2, $dash = $null) {
    if ($null -eq $color) {
        $stack = (Get-PSCallStack | Select-Object -First 6 | ForEach-Object { "$($_.FunctionName):$($_.ScriptLineNumber)" }) -join " <- "
        throw "Pen color is null. Stack: $stack"
    }
    $p = New-Object System.Drawing.Pen($color, $width)
    $p.StartCap = [System.Drawing.Drawing2D.LineCap]::Flat
    $p.EndCap = [System.Drawing.Drawing2D.LineCap]::Flat
    $p.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
    if ($dash) {
        $p.DashPattern = $dash
    }
    return $p
}

function Brush($color) {
    return New-Object System.Drawing.SolidBrush($color)
}

function RoundedPath($x, $y, $w, $h, $r) {
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = 2 * $r
    $path.AddArc($x, $y, $d, $d, 180, 90)
    $path.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $path.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $path.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    return $path
}

function DrawText($g, $text, $x, $y, $w, $h, $font, $color, $align = "Center", $valign = "Center") {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = if ($align -eq "Left") { [System.Drawing.StringAlignment]::Near } elseif ($align -eq "Right") { [System.Drawing.StringAlignment]::Far } else { [System.Drawing.StringAlignment]::Center }
    $sf.LineAlignment = if ($valign -eq "Top") { [System.Drawing.StringAlignment]::Near } elseif ($valign -eq "Bottom") { [System.Drawing.StringAlignment]::Far } else { [System.Drawing.StringAlignment]::Center }
    $sf.Trimming = [System.Drawing.StringTrimming]::Word
    $sf.FormatFlags = 0
    $rect = New-Object System.Drawing.RectangleF($x, $y, $w, $h)
    $g.DrawString($text, $font, (Brush $color), $rect, $sf)
}

function DrawBox($g, $x, $y, $w, $h, $text, $fill, $stroke, $fontSize = 28, $radius = 16, $strokeWidth = 3, $bold = $true) {
    $path = RoundedPath $x $y $w $h $radius
    $g.FillPath((Brush $fill), $path)
    $g.DrawPath((Pen $stroke $strokeWidth), $path)
    $style = if ($bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    DrawText $g $text $x $y $w $h (Font $fontSize $style) $Colors.Ink
}

function DrawPanel($g, $x, $y, $w, $h, $title, $fill, $stroke, $accent, $dash = $false) {
    $path = RoundedPath $x $y $w $h 26
    $g.FillPath((Brush $fill), $path)
    $pen = Pen $stroke 3
    if ($dash) { $pen.DashPattern = @(10, 7) }
    $g.DrawPath($pen, $path)
    DrawText $g $title ($x + 30) ($y + 18) ($w - 60) 44 (Font 30 ([System.Drawing.FontStyle]::Bold)) $accent "Left" "Center"
}

function DrawArrow($g, [float[]]$pts, $color, $width = 5, $dash = $null) {
    $pen = Pen $color $width $dash
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(8, 10, $true)
    $pen.CustomEndCap = $cap
    for ($i = 0; $i -lt ($pts.Length / 2 - 1); $i++) {
        $x1 = $pts[$i * 2]
        $y1 = $pts[$i * 2 + 1]
        $x2 = $pts[$i * 2 + 2]
        $y2 = $pts[$i * 2 + 3]
        if ($i -lt ($pts.Length / 2 - 2)) {
            $plain = Pen $color $width $dash
            $g.DrawLine($plain, $x1, $y1, $x2, $y2)
            $plain.Dispose()
        } else {
            $g.DrawLine($pen, $x1, $y1, $x2, $y2)
        }
    }
    $cap.Dispose()
    $pen.Dispose()
}

function DrawCircle($g, $cx, $cy, $r, $text, $stroke = $null, $fill = $null) {
    if ($null -eq $stroke) { $stroke = $Colors.Ink }
    if ($null -eq $fill) { $fill = $Colors.White }
    $g.FillEllipse((Brush $fill), $cx - $r, $cy - $r, 2 * $r, 2 * $r)
    $g.DrawEllipse((Pen $stroke 3), $cx - $r, $cy - $r, 2 * $r, 2 * $r)
    DrawText $g $text ($cx - $r) ($cy - $r) (2 * $r) (2 * $r) (Font 32 ([System.Drawing.FontStyle]::Bold)) $Colors.Ink
}

function DrawTokenRow($g, $x, $y, $n, $colors, $label = "", $size = 22, $gap = 5) {
    for ($i = 0; $i -lt $n; $i++) {
        $c = $colors[$i % $colors.Count]
        $g.FillRectangle((Brush $c), $x + $i * ($size + $gap), $y, $size, $size)
        $g.DrawRectangle((Pen $Colors.Ink 1), $x + $i * ($size + $gap), $y, $size, $size)
    }
    if ($label -ne "") {
        DrawText $g $label ($x - 20) ($y + $size + 6) ($n * ($size + $gap) + 40) 44 (Font 18) $Colors.Ink
    }
}

function DrawGrid($g, $x, $y, $rows, $cols, $label = "", $kind = "mask", $cell = 22) {
    $pal = switch ($kind) {
        "feature" { @($Colors.Teal, $Colors.Cyan, $Colors.Pos, (C "#E4CD00"), $Colors.Bnd, $Colors.Neg) }
        "binary" { @($Colors.Black, $Colors.White, $Colors.White, $Colors.White) }
        "prior" { @((C "#FFE5E8"), $Colors.Pink, (C "#F4F6F7"), (C "#A9CFE8")) }
        default { @((C "#F4F6F7"), $Colors.Pink, (C "#B7D8EF"), $Colors.White) }
    }
    for ($r = 0; $r -lt $rows; $r++) {
        for ($c = 0; $c -lt $cols; $c++) {
            $idx = ($r * 2 + $c * 3 + $r * $c) % $pal.Count
            $g.FillRectangle((Brush $pal[$idx]), $x + $c * $cell, $y + $r * $cell, $cell, $cell)
            $g.DrawRectangle((Pen (C "#C4CCD4") 1), $x + $c * $cell, $y + $r * $cell, $cell, $cell)
        }
    }
    if ($label -ne "") {
        DrawText $g $label ($x - 35) ($y + $rows * $cell + 8) ($cols * $cell + 70) 54 (Font 18) $Colors.Ink
    }
}

function DrawBadge($g, $x, $y, $text, $fill, $stroke) {
    DrawBox $g $x $y 170 50 $text $fill $stroke 22 12 2 $true
}

$bmp = New-Object System.Drawing.Bitmap($W, $H, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g.Clear($Colors.White)

# Header
DrawText $g "MedEx-SAM3 Overall Framework" 70 45 1800 70 (Font 56 ([System.Drawing.FontStyle]::Bold)) $Colors.Ink "Left" "Center"
DrawText $g "Paper-style landscape redraw: module insertion points and branch boundaries" 72 112 1800 44 (Font 25 ([System.Drawing.FontStyle]::Bold)) $Colors.Blue "Left" "Center"

# Main SAM3 core panel
DrawPanel $g 360 250 2880 560 "Official SAM3 Tensor Forward Wrapper" $Colors.BluePale $Colors.Blue $Colors.Blue
DrawText $g "Base path uses Stage-A LoRA inside selected SAM3 Linear layers; image embeddings are also returned to the wrapper." 405 306 1500 34 (Font 20 ([System.Drawing.FontStyle]::Bold)) $Colors.Muted "Left" "Center"

DrawGrid $g 430 415 4 4 "image`n[B,3,H,W]" "mask" 24
DrawBox $g 610 390 260 110 "Forward Image`nBackbone" $Colors.BlueLight $Colors.Blue 28
DrawBadge $g 655 330 "LoRA" (C "#FFF000") $Colors.YellowStroke
DrawText $g "vision encoder late 1/3 blocks" 560 515 360 30 (Font 18 ([System.Drawing.FontStyle]::Bold)) $Colors.Blue "Left" "Center"

DrawBox $g 1000 390 245 110 "Encode Prompt" $Colors.BlueLight $Colors.Blue 28
DrawBox $g 1375 390 230 110 "Run Encoder" $Colors.BlueLight $Colors.Blue 28
DrawBox $g 1800 390 230 110 "Run Decoder" $Colors.BlueLight $Colors.Blue 28
DrawBadge $g 1830 330 "LoRA" (C "#FFF000") $Colors.YellowStroke
DrawText $g "mask decoder attention/proj layers" 1740 515 390 30 (Font 18 ([System.Drawing.FontStyle]::Bold)) $Colors.Blue "Left" "Center"
DrawBox $g 2185 390 300 110 "Run Segmentation`nHeads" $Colors.BlueLight $Colors.Blue 27
DrawGrid $g 2590 410 3 6 "SAM3 mask_logits`n[B,1,H,W]" "prior" 22
DrawBox $g 2775 390 250 110 "Apply RSS-DA`nLogit Bias" $Colors.RedLight $Colors.Red 25
DrawGrid $g 3125 410 3 6 "conditioned`nmask_logits" "prior" 22

DrawArrow $g @(528,462, 610,462) $Colors.Blue 5
DrawArrow $g @(870,445, 1000,445) $Colors.Blue 5
DrawArrow $g @(1245,445, 1375,445) $Colors.Blue 5
DrawArrow $g @(1605,445, 1800,445) $Colors.Blue 5
DrawArrow $g @(2030,445, 2185,445) $Colors.Blue 5
DrawArrow $g @(2485,445, 2590,445) $Colors.Blue 5
DrawArrow $g @(2725,445, 2775,445) $Colors.Blue 5
DrawArrow $g @(3025,445, 3125,445) $Colors.Red 5

# Prompt and exemplar branch
DrawPanel $g 80 200 1170 355 "Prompt Inputs + Exemplar Prompt Branch" (C "#FBFAFF") $Colors.Purple $Colors.Purple
DrawBox $g 155 300 255 88 "Text / Point / Box`nPrompt" (C "#FFF8EA") $Colors.Bnd 24
DrawTokenRow $g 455 330 8 @($Colors.Bnd) "text + geometry tokens" 22
DrawArrow $g @(410,344,455,344) $Colors.Blue 4
DrawGrid $g 150 430 3 4 "exemplar image`n+ annotation" "mask" 20
DrawBox $g 330 420 330 90 "ExemplarPromptAdapter`npositive + boundary + negative prototypes" $Colors.PurpleLight $Colors.Purple 22
DrawTokenRow $g 705 435 8 @($Colors.Pos,$Colors.Pos,$Colors.Pos,$Colors.Bnd,$Colors.Bnd,$Colors.Neg,$Colors.Neg,$Colors.Teal) "visual_prompt_embed [B,Nv,C]" 21
DrawArrow $g @(250,462,330,462) $Colors.Purple 4
DrawArrow $g @(660,462,705,462) $Colors.Purple 4
DrawText $g "All positive, boundary, and negative tokens enter the prompt sequence; suppression_gate is a side output." 140 515 1020 32 (Font 19 ([System.Drawing.FontStyle]::Bold)) $Colors.Purple "Left" "Center"
DrawArrow $g @(820,435, 955,435, 955,390, 1000,390) $Colors.Purple 4

# RSS-DA branch
DrawPanel $g 80 620 1300 480 "RSS-DA Retrieval Branch (separate enhancement experiment)" $Colors.RedLight $Colors.Red $Colors.Red
DrawGrid $g 150 735 3 5 "query / image`nfeatures" "feature" 19
DrawBox $g 360 715 210 85 "Prototype`nRetriever" (C "#FFF8EA") $Colors.YellowStroke 23
DrawTokenRow $g 620 720 6 @($Colors.Pos) "positive prototype" 22
DrawTokenRow $g 620 790 6 @($Colors.Neg) "negative prototype" 22
DrawTokenRow $g 620 860 6 @($Colors.Bnd) "boundary prototype" 22
DrawBox $g 830 735 230 95 "Similarity`nHeatmap Builder" (C "#FFF8EA") $Colors.YellowStroke 23
DrawGrid $g 1120 735 2 8 "spatial prior" "prior" 19
DrawBox $g 340 925 310 88 "RetrievalSpatialSemanticAdapter`n+ GatedRetrievalFusion" $Colors.RedLight $Colors.Red 21
DrawTokenRow $g 715 930 8 @($Colors.Teal) "encoder_memory_bias" 19
DrawTokenRow $g 715 990 8 @($Colors.Pos) "decoder_feature_bias_map" 19
DrawTokenRow $g 715 1050 8 @($Colors.Red) "mask_logit_bias_map" 19
DrawArrow $g @(245,760,360,760) $Colors.Blue 4
DrawArrow $g @(570,760,620,760) $Colors.Blue 4
DrawArrow $g @(760,790,830,790) $Colors.Blue 4
DrawArrow $g @(1060,790,1120,790) $Colors.Blue 4
DrawArrow $g @(1180,820,650,950) $Colors.Red 4
DrawArrow $g @(650,965,715,965) $Colors.Red 4
DrawText $g "retrieval_prior applies before decoder memory use and again after SAM3 mask_logits." 130 1080 1100 34 (Font 19 ([System.Drawing.FontStyle]::Bold)) $Colors.Red "Left" "Center"
DrawArrow $g @(1010,965, 1510,965, 1510,500, 1510,500) $Colors.Red 4 @(10,7)
DrawArrow $g @(1010,1055, 2900,1055, 2900,500, 2900,500) $Colors.Red 4 @(10,7)

# MedEx wrapper refinement
DrawPanel $g 180 1220 3260 430 "MedEx Wrapper Refinement after SAM3 Forward" $Colors.GreenLight $Colors.Green $Colors.Green
DrawText $g "refine_head is always present; MedicalImageAdapter and BoundaryAwareAdapter are optional explicit modules." 235 1276 1600 34 (Font 21 ([System.Drawing.FontStyle]::Bold)) $Colors.Green "Left" "Center"

DrawGrid $g 260 1390 3 6 "image_embeddings`nfeature_map" "feature" 22
DrawBox $g 520 1370 225 95 "Resolve`nFeature Map" $Colors.White $Colors.Ink 24
DrawBox $g 860 1365 270 105 "MedicalImageAdapter`noptional texture + bottleneck" $Colors.GreenLight $Colors.Green 22
DrawBox $g 1265 1365 285 105 "BoundaryAwareAdapter`noptional coarse-mask boundary gate" $Colors.GreenLight $Colors.Green 21
DrawBox $g 1680 1370 210 95 "refine_head`nConv 1x1" $Colors.White $Colors.Ink 23
DrawGrid $g 2000 1390 2 7 "delta logits" "prior" 21
DrawBox $g 2225 1378 150 80 "scale`n0.1" $Colors.White $Colors.Ink 23
DrawCircle $g 2535 1420 44 "+"
DrawGrid $g 2685 1385 3 6 "final_mask_logits`nconditioned + 0.1*delta" "prior" 22
DrawBox $g 2965 1378 145 80 "Sigmoid" $Colors.White $Colors.Ink 23
DrawGrid $g 3235 1375 4 4 "Final Mask`n[B,1,H,W]" "binary" 22

DrawArrow $g @(405,1420,520,1420) $Colors.Blue 5
DrawArrow $g @(745,1420,860,1420) $Colors.Green 5
DrawArrow $g @(1130,1420,1265,1420) $Colors.Green 5
DrawArrow $g @(1550,1420,1680,1420) $Colors.Green 5
DrawArrow $g @(1890,1420,2000,1420) $Colors.Green 5
DrawArrow $g @(2165,1420,2225,1420) $Colors.Green 5
DrawArrow $g @(2375,1420,2490,1420) $Colors.Green 5
DrawArrow $g @(2579,1420,2685,1420) $Colors.Green 5
DrawArrow $g @(2825,1420,2965,1420) $Colors.Blue 5
DrawArrow $g @(3110,1420,3235,1420) $Colors.Blue 5

DrawGrid $g 1325 1545 3 4 "coarse mask logits`n(+ GT mask only during training)" "prior" 18
DrawArrow $g @(1365,1545, 1410,1500, 1410,1470) $Colors.Green 4 @(10,7)
DrawGrid $g 2480 1180 3 5 "conditioned`nmask_logits" "prior" 18
DrawArrow $g @(2550,1235, 2550,1376) $Colors.Green 5 @(10,7)

# Cross-panel connectors
DrawArrow $g @(1015,555, 1015,390) $Colors.Purple 4
DrawArrow $g @(3235,500, 3235,1180, 2600,1180) $Colors.Green 5 @(10,7)
DrawArrow $g @(760,500, 760,780, 1000,780, 1000,390) $Colors.Blue 4

# Legend and branch-boundary note
$legendY = 1685
DrawBox $g 240 $legendY 255 54 "Blue: SAM3 main flow" $Colors.BlueLight $Colors.Blue 20 10 2 $false
DrawBox $g 520 $legendY 210 54 "Yellow: LoRA" $Colors.Yellow $Colors.YellowStroke 20 10 2 $false
DrawBox $g 755 $legendY 310 54 "Purple: Exemplar Prompt" $Colors.PurpleLight $Colors.Purple 20 10 2 $false
DrawBox $g 1090 $legendY 250 54 "Red: RSS-DA prior" $Colors.RedLight $Colors.Red 20 10 2 $false
DrawBox $g 1365 $legendY 310 54 "Green: MedEx wrapper" $Colors.GreenLight $Colors.Green 20 10 2 $false
DrawText $g "Experimental boundary: do not present LoRA, Medical/Boundary adapters, Exemplar Prompt, and RSS-DA as one default training path. Base branch = Stage-A LoRA + refine_head; other modules are optional or separate branches." 1820 1678 1550 78 (Font 22 ([System.Drawing.FontStyle]::Bold)) $Colors.Ink "Left" "Center"
DrawText $g "Key route: Image / Prompt inputs -> SAM3 encoder-decoder -> conditioned mask_logits -> MedEx wrapper refinement -> final mask" 190 1780 3000 40 (Font 28 ([System.Drawing.FontStyle]::Bold)) $Colors.Blue "Left" "Center"

# Save JPG with high quality.
$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, [int64]95)
$bmp.Save($OutPath, $encoder, $params)
$g.Dispose()
$bmp.Dispose()

Get-Item -LiteralPath $OutPath | Select-Object FullName,Length,LastWriteTime
