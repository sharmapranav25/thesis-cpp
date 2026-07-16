<?php
// Standalone port of MyBB's default captcha (mybb/captcha.php), parameterized
// over a per-image difficulty d in [0,1] and a length mode (fixed:N or
// range:MIN-MAX).
//
// At d=1.0 the renderer reproduces MyBB's exact rendering pipeline:
//   - white bg + lines/circles/squares (random per-image)
//   - light dots overlay (~2400 random pixels for 200x60)
//   - chars: random TTF font (size 20-32), rotation +/-30 deg,
//     color (0..200 each channel), shadow offset +/-3px in lighter color
//   - 1px black border
//
// As d decreases the rotation, dot density, color spread, shadow offset,
// background-pattern probability, and font diversity all collapse toward
// zero. At d=0 the output is a single-font, black-on-white, axis-aligned,
// noise-free string.
//
// Length mode:
//   fixed:N      every captcha has exactly N chars
//   range:MIN-MAX  per-image length sampled uniformly in [MIN, MAX]
//
// Image width scales as 40 * length (so 5 chars -> 200px = MyBB-native).
// Image height is fixed at 60.
//
// CLI usage:
//   php generate_mybb_captcha.php --out DIR --n 100000 --seed 0 \
//       --difficulty-mode uniform --length-mode range:3-7
// emits {0000000.png, ...} and labels.csv with columns
//   filename,label,difficulty,is_fixed_length

ini_set('memory_limit', '1G');

// --- args ----------------------------------------------------------------
$opts = getopt('', ['out:', 'n:', 'seed:', 'fonts:',
                    'difficulty-mode:', 'length-mode:']);
$out_dir             = $opts['out']             ?? __DIR__ . '/../../data/synth_mybb';
$n_total             = (int)($opts['n']         ?? 10);
$seed                = (int)($opts['seed']      ?? 0);
$fonts_dir           = $opts['fonts']           ?? 'C:/Windows/Fonts';
$difficulty_mode_str = $opts['difficulty-mode'] ?? 'fixed:1.0';
$length_mode_str     = $opts['length-mode']     ?? 'fixed:5';

if (!is_dir($out_dir)) mkdir($out_dir, 0777, true);

mt_srand($seed);

// --- mode parsers --------------------------------------------------------
function parse_difficulty_mode($s) {
    if ($s === 'uniform') return ['uniform', 0.0];
    if (preg_match('/^fixed:([0-9]*\.?[0-9]+)$/', $s, $m)) {
        $f = (float)$m[1];
        if ($f < 0.0 || $f > 1.0) {
            fwrite(STDERR, "--difficulty-mode fixed:F requires F in [0,1], got $f\n");
            exit(1);
        }
        return ['fixed', $f];
    }
    fwrite(STDERR, "bad --difficulty-mode '$s'; expected 'uniform' or 'fixed:F'\n");
    exit(1);
}
function parse_length_mode($s) {
    if (preg_match('/^fixed:(\d+)$/', $s, $m)) return ['fixed', (int)$m[1], (int)$m[1]];
    if (preg_match('/^range:(\d+)-(\d+)$/', $s, $m)) {
        $a = (int)$m[1]; $b = (int)$m[2];
        if ($a > $b || $a < 1) {
            fwrite(STDERR, "bad --length-mode range '$s'\n"); exit(1);
        }
        return ['range', $a, $b];
    }
    fwrite(STDERR, "bad --length-mode '$s'; expected 'fixed:N' or 'range:MIN-MAX'\n");
    exit(1);
}
[$d_mode, $d_val]         = parse_difficulty_mode($difficulty_mode_str);
[$l_mode, $l_min, $l_max] = parse_length_mode($length_mode_str);
$is_fixed_length = ($l_mode === 'fixed') ? 1 : 0;

function sample_difficulty($mode, $val) {
    if ($mode === 'uniform') return mt_rand() / mt_getrandmax();
    return $val;
}
function sample_length($mode, $lo, $hi) {
    if ($mode === 'fixed') return $lo;
    return mt_rand($lo, $hi);
}

// --- font pool (same as model 1) ----------------------------------------
$font_files = [
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
$ttf_fonts = [];
foreach ($font_files as $f) {
    $p = "$fonts_dir/$f";
    if (file_exists($p)) $ttf_fonts[] = $p;
}
if (count($ttf_fonts) === 0) {
    fwrite(STDERR, "no TTF fonts found in $fonts_dir\n");
    exit(1);
}
// Anchor arial.ttf at index 0 so d=0 picks a deterministic single font.
$arial_idx = -1;
foreach ($ttf_fonts as $i => $p) {
    if (strtolower(basename($p)) === 'arial.ttf') { $arial_idx = $i; break; }
}
if ($arial_idx > 0) {
    $arial = $ttf_fonts[$arial_idx];
    array_splice($ttf_fonts, $arial_idx, 1);
    array_unshift($ttf_fonts, $arial);
}
fwrite(STDERR, "loaded " . count($ttf_fonts) . " fonts (anchor=" . basename($ttf_fonts[0]) . ")\n");

// --- charset ------------------------------------------------------------
$alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

function rand_string($alphabet, $len) {
    $s = '';
    $a_len = strlen($alphabet);
    for ($i = 0; $i < $len; $i++) $s .= $alphabet[mt_rand(0, $a_len - 1)];
    return $s;
}

// --- background drawing (verbatim MyBB at d=1; gated externally by d) ---
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
function draw_squares(&$im, $img_width, $img_height) {
    $square_count = 30;
    for ($i = 0; $i <= $square_count; ++$i) {
        $color = imagecolorallocate($im, rand(150, 255), rand(150, 255), rand(150, 255));
        $pos_x = rand(1, $img_width);
        $pos_y = rand(1, $img_height);
        $sq = rand(10, 20);
        imagefilledrectangle($im, $pos_x, $pos_y, $pos_x + $sq, $pos_y + $sq, $color);
    }
}
// Dots scale linearly with d (count = MyBB-default * d).
function draw_dots(&$im, $img_width, $img_height, $d) {
    $dot_count = (int)round($img_width * $img_height / 5 * $d);
    for ($i = 0; $i < $dot_count; ++$i) {
        $color = imagecolorallocate($im, rand(200, 255), rand(200, 255), rand(200, 255));
        imagesetpixel($im, rand(0, $img_width - 1), rand(0, $img_height - 1), $color);
    }
}

// --- char rendering with d-scaling --------------------------------------
function draw_string_ttf(&$im, $string, $img_width, $img_height, $d, $ttf_fonts) {
    $string_length = strlen($string);
    if ($string_length === 0) return;
    $spacing = $img_width / $string_length;

    // d-scaled ranges (linear interp from "easy" to MyBB-default).
    $rot_max     = (int)round(30 * $d);                                     // +/- deg
    $size_lo     = (int)round(26 - 6 * $d);
    $size_hi     = (int)round(26 + 6 * $d);
    $color_max   = (int)round(200 * $d);                                    // 0..C max
    $shadow_max  = (int)round(3 * $d);                                      // +/- px
    $shadow_step = (int)round(20 * $d);
    $font_pool_size = max(1, (int)ceil($d * count($ttf_fonts)));

    for ($i = 0; $i < $string_length; ++$i) {
        $font_size = ($size_lo === $size_hi) ? $size_lo : rand($size_lo, $size_hi);
        $font = $ttf_fonts[mt_rand(0, $font_pool_size - 1)];
        $rotation = ($rot_max === 0) ? 0 : rand(-$rot_max, $rot_max);
        $r = ($color_max === 0) ? 0 : rand(0, $color_max);
        $g = ($color_max === 0) ? 0 : rand(0, $color_max);
        $b = ($color_max === 0) ? 0 : rand(0, $color_max);
        $color = imagecolorallocate($im, $r, $g, $b);

        $dimensions = imageftbbox($font_size, $rotation, $font, $string[$i], []);
        $string_height = $dimensions[3] - $dimensions[5];

        $pos_x = $spacing / 4 + $i * $spacing;
        $pos_y = ceil(($img_height - $string_height / 2));

        if ($shadow_max > 0) {
            $shadow_x = rand(-$shadow_max, $shadow_max) + $pos_x;
            $shadow_y = rand(-$shadow_max, $shadow_max) + $pos_y;
            $shadow_color = imagecolorallocate(
                $im,
                min(255, $r + $shadow_step),
                min(255, $g + $shadow_step),
                min(255, $b + $shadow_step)
            );
            imagefttext($im, $font_size, $rotation, $shadow_x, $shadow_y,
                        $shadow_color, $font, $string[$i], []);
        }
        imagefttext($im, $font_size, $rotation, $pos_x, $pos_y, $color, $font, $string[$i], []);
    }
}

// --- one-image generator ------------------------------------------------
function generate_one($text, $img_width, $img_height, $d, $ttf_fonts) {
    $im = imagecreatetruecolor($img_width, $img_height);
    $bg = imagecolorallocate($im, 255, 255, 255);
    imagefill($im, 0, 0, $bg);

    // background pattern Bernoulli-gated by d
    if (mt_rand() / mt_getrandmax() < $d) {
        $to_draw = rand(0, 2);
        if      ($to_draw === 1) draw_circles($im, $img_width, $img_height);
        else if ($to_draw === 2) draw_squares($im, $img_width, $img_height);
        else                     draw_lines($im, $img_width, $img_height);
    }

    draw_dots($im, $img_width, $img_height, $d);
    draw_string_ttf($im, $text, $img_width, $img_height, $d, $ttf_fonts);

    $border = imagecolorallocate($im, 0, 0, 0);
    imagerectangle($im, 0, 0, $img_width - 1, $img_height - 1, $border);
    return $im;
}

// --- main loop ----------------------------------------------------------
$labels_path = "$out_dir/labels.csv";
$labels_fp = fopen($labels_path, 'w');
fwrite($labels_fp, "filename,label,difficulty,is_fixed_length\n");

$report_every = max(1, intdiv($n_total, 50));
$t_start = microtime(true);
for ($i = 0; $i < $n_total; $i++) {
    $d = sample_difficulty($d_mode, $d_val);
    $length = sample_length($l_mode, $l_min, $l_max);
    $img_width = 40 * $length;
    $img_height = 60;
    $text = rand_string($alphabet, $length);
    $im = generate_one($text, $img_width, $img_height, $d, $ttf_fonts);
    $fn = sprintf("%07d.png", $i);
    imagepng($im, "$out_dir/$fn");
    imagedestroy($im);
    fwrite($labels_fp, sprintf("%s,%s,%.4f,%d\n", $fn, $text, $d, $is_fixed_length));
    if (($i + 1) % $report_every === 0) {
        $elapsed = microtime(true) - $t_start;
        $rate = ($i + 1) / max(0.001, $elapsed);
        fwrite(STDERR, sprintf("  %d/%d  (%.1f img/s)\n", $i + 1, $n_total, $rate));
    }
}
fclose($labels_fp);
$elapsed = microtime(true) - $t_start;
fwrite(STDERR, sprintf("\ndone: %d images in %.1fs\n", $n_total, $elapsed));
