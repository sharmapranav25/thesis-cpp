<?php
// Standalone port of MyBB's default captcha (mybb/captcha.php). Generates
// a 200x60 PNG matching MyBB's exact rendering pipeline:
//   - white bg + lines/circles/squares (random per-image)
//   - light dots overlay
//   - 5 chars: random TTF font (size 20-32), random rotation -30..+30,
//     random color (0..200 each channel), shadow offset ±3px in lighter color
//   - 1px black border
//
// CLI usage:
//   php generate_mybb_captcha.php --out DIR --n 100000 --seed 0
// emits {0000000.png, 0000001.png, ...} and labels.csv with (filename,label).

ini_set('memory_limit', '1G');

// --- args ----------------------------------------------------------------
$opts = getopt('', ['out:', 'n:', 'seed:', 'fonts:', 'fontset:', 'bboxes',
                    'supersample:', 'mixed-scale', 'text-only',
                    'width:', 'height:', 'captcha-len:', 'charset:']);
$out_dir   = $opts['out']   ?? __DIR__ . '/../../data/synth_mybb';
$n_total   = (int)($opts['n']    ?? 10);
$seed      = (int)($opts['seed'] ?? 0);
$fonts_dir = $opts['fonts'] ?? 'C:/Windows/Fonts';
// fontset: 'windows' (default, ~30 Windows TTFs) or 'mybb' (the 2 fonts MyBB ships)
$fontset   = $opts['fontset'] ?? 'windows';
$emit_bbox = array_key_exists('bboxes', $opts);
// supersample factor: render at SS× target resolution then downsample with
// imagecopyresampled (bicubic-ish). SS=1 keeps original behaviour. SS=2 gives
// crisper anti-aliased text at the same final resolution.
$supersample = max(1, (int)($opts['supersample'] ?? 1));
// mixed-scale: 50/50 split between 200×60 and 400×120 native target sizes,
// matching the heterogeneity observed in the 651 real eval set.
$mixed_scale = array_key_exists('mixed-scale', $opts);
// text-only: render text on transparent RGBA canvas — no bg primitives,
// no border. Used by the synthetic image generation project to composite
// text over GAN/extracted bgs.
$text_only = array_key_exists('text-only', $opts);

if (!is_dir($out_dir)) mkdir($out_dir, 0777, true);

mt_srand($seed);

// --- font pool ---------------------------------------------------------
// 'windows': ~30 curated Windows TTFs (Model 1 baseline approximation)
// 'mybb':    the 2 fonts MyBB actually ships (MINYN___.ttf, edmunds.ttf)
//            — what real CryptBB captchas are rendered with
$windows_font_files = [
    'arial.ttf', 'arialbd.ttf', 'arialbi.ttf', 'ariblk.ttf',
    'calibri.ttf', 'calibrib.ttf', 'calibriz.ttf',
    'cambriab.ttf', 'cambriaz.ttf',
    'Candara.ttf', 'Candarab.ttf', 'Candaraz.ttf',
    'comic.ttf', 'comicbd.ttf',
    'consola.ttf', 'consolab.ttf',
    'cour.ttf', 'courbd.ttf',
    'georgia.ttf', 'georgiab.ttf', 'georgiaz.ttf',
    'impact.ttf',
    'lucon.ttf',
    'tahoma.ttf', 'tahomabd.ttf',
    'times.ttf', 'timesbd.ttf', 'timesbi.ttf',
    'trebuc.ttf', 'trebucbd.ttf',
    'verdana.ttf', 'verdanab.ttf', 'verdanaz.ttf',
];
$mybb_font_files = ['MINYN___.ttf', 'edmunds.ttf'];

$ttf_fonts = [];
$font_files = ($fontset === 'mybb') ? $mybb_font_files : $windows_font_files;
foreach ($font_files as $f) {
    $p = "$fonts_dir/$f";
    if (file_exists($p)) $ttf_fonts[] = $p;
}
if (count($ttf_fonts) === 0) {
    fwrite(STDERR, "no TTF fonts found in $fonts_dir (fontset=$fontset)\n");
    exit(1);
}
fwrite(STDERR, "fontset=$fontset, loaded " . count($ttf_fonts) . " fonts from $fonts_dir\n");

// --- image params (matches MyBB) ----------------------------------------
// Native MyBB renders at 200×60. Eval set has both 200×60 and 400×120;
// when --mixed-scale is on we randomise per-image. All other render params
// scale linearly with image width (so the captcha fills proportionally).
// Per-scheme overrides via CLI: --width, --height, --captcha-len, --charset
// Defaults match CryptBB / MyBB.
$base_width = (int)($opts['width']  ?? 200);
$base_height = (int)($opts['height'] ?? 60);
// Font size scales with image height so different schemes (e.g. DarkBay 200×50)
// still get reasonable char sizes filling the image.
$base_min_size = (int)round(20 * $base_height / 60);
$base_max_size = (int)round(32 * $base_height / 60);
$min_angle = -30;
$max_angle = 30;

// --- charset: configurable per scheme -----------------------------------
// Default: full alphanumeric mixed-case (CryptBB).
// DarkBay uses uppercase + digits, no I: '0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ'
$alphabet    = $opts['charset']     ?? '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
$captcha_len = (int)($opts['captcha-len'] ?? 5);

function rand_string($alphabet, $len) {
    $s = '';
    $a_len = strlen($alphabet);
    for ($i = 0; $i < $len; $i++) $s .= $alphabet[mt_rand(0, $a_len - 1)];
    return $s;
}

// --- background drawing functions (verbatim from MyBB) ------------------
function draw_lines(&$im, $img_width, $img_height) {
    for ($i = 10; $i < $img_width; $i += 10) {
        $color = imagecolorallocate($im, rand(150, 255), rand(150, 255), rand(150, 255));
        imageline($im, $i, 0, $i, $img_height, $color);
    }
    for ($i = 10; $i < $img_height; $i += 10) {
        $color = imagecolorallocate($im, rand(150, 255), rand(150, 255), rand(150, 255));
        imageline($im, 0, $i, $img_width, $i, $color);
    }
}

function draw_circles(&$im, $img_width, $img_height) {
    $circles = $img_width * $img_height / 100;
    for ($i = 0; $i <= $circles; ++$i) {
        $color = imagecolorallocate($im, rand(180, 255), rand(180, 255), rand(180, 255));
        $pos_x = rand(1, $img_width);
        $pos_y = rand(1, $img_height);
        $circ_width = ceil(rand(1, $img_width) / 2);
        $circ_height = rand(1, $img_height);
        imagearc($im, $pos_x, $pos_y, $circ_width, $circ_height, 0, rand(200, 360), $color);
    }
}

function draw_dots(&$im, $img_width, $img_height) {
    $dot_count = $img_width * $img_height / 5;
    for ($i = 0; $i <= $dot_count; ++$i) {
        $color = imagecolorallocate($im, rand(200, 255), rand(200, 255), rand(200, 255));
        imagesetpixel($im, rand(0, $img_width), rand(0, $img_height), $color);
    }
}

function draw_squares(&$im, $img_width, $img_height) {
    $square_count = 30;
    for ($i = 0; $i <= $square_count; ++$i) {
        $color = imagecolorallocate($im, rand(150, 255), rand(150, 255), rand(150, 255));
        $pos_x = rand(1, $img_width);
        $pos_y = rand(1, $img_height);
        $sq_width = $sq_height = rand(10, 20);
        $pos_x2 = $pos_x + $sq_height;
        $pos_y2 = $pos_y + $sq_width;
        imagefilledrectangle($im, $pos_x, $pos_y, $pos_x2, $pos_y2, $color);
    }
}

function draw_string_ttf(&$im, $string, $img_width, $img_height,
                         $min_size, $max_size, $min_angle, $max_angle, $ttf_fonts) {
    $string_length = strlen($string);
    $bboxes = [];
    if ($string_length === 0) return $bboxes;
    $spacing = $img_width / $string_length;
    for ($i = 0; $i < $string_length; ++$i) {
        $font_size = rand($min_size, $max_size);
        $font = $ttf_fonts[array_rand($ttf_fonts)];
        $rotation = rand($min_angle, $max_angle);
        $r = rand(0, 200);
        $g = rand(0, 200);
        $b = rand(0, 200);
        $color = imagecolorallocate($im, $r, $g, $b);

        $dimensions = imageftbbox($font_size, $rotation, $font, $string[$i], []);
        $string_height = $dimensions[3] - $dimensions[5];

        $pos_x = $spacing / 4 + $i * $spacing;
        $pos_y = ceil(($img_height - $string_height / 2));

        // shadow
        $shadow_x = rand(-3, 3) + $pos_x;
        $shadow_y = rand(-3, 3) + $pos_y;
        $shadow_color = imagecolorallocate($im, min(255, $r + 20), min(255, $g + 20), min(255, $b + 20));
        imagefttext($im, $font_size, $rotation, $shadow_x, $shadow_y, $shadow_color, $font, $string[$i], []);

        // main char
        imagefttext($im, $font_size, $rotation, $pos_x, $pos_y, $color, $font, $string[$i], []);

        // axis-aligned bbox in image coords (covers char + shadow).
        // imageftbbox returns 4 (x,y) pairs at the rotated text corners,
        // relative to baseline (0,0). Translate by pos_{x,y}; expand to
        // include shadow by unioning with shifted corners.
        $xs = [$dimensions[0], $dimensions[2], $dimensions[4], $dimensions[6]];
        $ys = [$dimensions[1], $dimensions[3], $dimensions[5], $dimensions[7]];
        $main_x0 = $pos_x + min($xs); $main_x1 = $pos_x + max($xs);
        $main_y0 = $pos_y + min($ys); $main_y1 = $pos_y + max($ys);
        $sh_x0   = $shadow_x + min($xs); $sh_x1   = $shadow_x + max($xs);
        $sh_y0   = $shadow_y + min($ys); $sh_y1   = $shadow_y + max($ys);
        $x0 = max(0, (int)floor(min($main_x0, $sh_x0)));
        $y0 = max(0, (int)floor(min($main_y0, $sh_y0)));
        $x1 = min($img_width  - 1, (int)ceil(max($main_x1, $sh_x1)));
        $y1 = min($img_height - 1, (int)ceil(max($main_y1, $sh_y1)));
        $bboxes[] = [$x0, $y0, $x1, $y1];
    }
    return $bboxes;
}

function generate_one($text, $img_width, $img_height,
                     $min_size, $max_size, $min_angle, $max_angle, $ttf_fonts,
                     $text_only = false) {
    $im = imagecreatetruecolor($img_width, $img_height);

    if ($text_only) {
        // transparent canvas: alphablending off while filling so the fill
        // *replaces* (rather than blends with) destination, then re-enable
        // for text drawing so text anti-alias edges blend with transparent bg.
        imagealphablending($im, false);
        imagesavealpha($im, true);
        $transparent = imagecolorallocatealpha($im, 0, 0, 0, 127);
        imagefill($im, 0, 0, $transparent);
        imagealphablending($im, true);
        $bboxes = draw_string_ttf($im, $text, $img_width, $img_height,
                                  $min_size, $max_size, $min_angle, $max_angle, $ttf_fonts);
        return [$im, $bboxes];
    }

    $bg = imagecolorallocate($im, 255, 255, 255);
    imagefill($im, 0, 0, $bg);

    $to_draw = rand(0, 2);
    if ($to_draw == 1)      draw_circles($im, $img_width, $img_height);
    else if ($to_draw == 2) draw_squares($im, $img_width, $img_height);
    else                    draw_lines($im, $img_width, $img_height);

    draw_dots($im, $img_width, $img_height);
    $bboxes = draw_string_ttf($im, $text, $img_width, $img_height,
                              $min_size, $max_size, $min_angle, $max_angle, $ttf_fonts);

    $border = imagecolorallocate($im, 0, 0, 0);
    imagerectangle($im, 0, 0, $img_width - 1, $img_height - 1, $border);
    return [$im, $bboxes];
}

// --- main loop -----------------------------------------------------------
$labels_path = "$out_dir/labels.csv";
$labels_fp = fopen($labels_path, 'w');
fwrite($labels_fp, "filename,label\n");

$bboxes_fp = null;
if ($emit_bbox) {
    $bboxes_fp = fopen("$out_dir/bboxes.jsonl", 'w');
}

fwrite(STDERR, sprintf("supersample=%d, mixed_scale=%d\n", $supersample, $mixed_scale ? 1 : 0));

$report_every = max(1, intdiv($n_total, 50));
$t_start = microtime(true);
for ($i = 0; $i < $n_total; $i++) {
    $text = rand_string($alphabet, $captcha_len);

    // pick target final size for this image. mixed-scale: 50/50 between base
    // and 2× base, matching the size heterogeneity of some real eval sets.
    if ($mixed_scale && rand(0, 1) === 1) {
        $target_w = $base_width * 2; $target_h = $base_height * 2;
    } else {
        $target_w = $base_width; $target_h = $base_height;
    }

    // render at supersample×target; all pixel-units scale linearly with width
    $render_w = $target_w * $supersample;
    $render_h = $target_h * $supersample;
    $scale = $render_w / $base_width;
    $min_size = (int)round($base_min_size * $scale);
    $max_size = (int)round($base_max_size * $scale);

    [$im_render, $bb_render] = generate_one($text, $render_w, $render_h,
                                            $min_size, $max_size, $min_angle, $max_angle, $ttf_fonts,
                                            $text_only);

    if ($supersample > 1) {
        // downsample render → target with bicubic (imagecopyresampled is decent quality)
        $im = imagecreatetruecolor($target_w, $target_h);
        if ($text_only) {
            imagealphablending($im, false);
            imagesavealpha($im, true);
            $transparent = imagecolorallocatealpha($im, 0, 0, 0, 127);
            imagefill($im, 0, 0, $transparent);
        }
        imagecopyresampled($im, $im_render, 0, 0, 0, 0, $target_w, $target_h, $render_w, $render_h);
        imagedestroy($im_render);
    } else {
        $im = $im_render;
    }
    // bboxes: render-space coords → target-space coords (just divide by SS)
    $bboxes = [];
    foreach ($bb_render as $bb) {
        $bboxes[] = [
            (int)floor($bb[0] / $supersample),
            (int)floor($bb[1] / $supersample),
            (int)ceil($bb[2] / $supersample),
            (int)ceil($bb[3] / $supersample),
        ];
    }

    $fn = sprintf("%07d.png", $i);
    imagepng($im, "$out_dir/$fn");
    imagedestroy($im);
    fwrite($labels_fp, "$fn,$text\n");
    if ($bboxes_fp !== null) {
        fwrite($bboxes_fp, json_encode([
            "filename" => $fn,
            "label" => $text,
            "width" => $target_w,
            "height" => $target_h,
            "bboxes" => $bboxes,
        ]) . "\n");
    }
    if (($i + 1) % $report_every === 0) {
        $elapsed = microtime(true) - $t_start;
        $rate = ($i + 1) / max(0.001, $elapsed);
        fwrite(STDERR, sprintf("  %d/%d  (%.1f img/s)\n", $i + 1, $n_total, $rate));
    }
}
fclose($labels_fp);
if ($bboxes_fp !== null) fclose($bboxes_fp);
$elapsed = microtime(true) - $t_start;
fwrite(STDERR, sprintf("\ndone: %d images in %.1fs\n", $n_total, $elapsed));
