<?php
// Unified PHP captcha generator: one script that dispatches across multiple
// publicly-available PHP captcha libraries to produce a varied training corpus.
//
// Libraries used:
//   - Securimage     (rich parameter sweep, the canonical PHP captcha)
//   - Gregwar/Captcha (modern, distortion + scatter + lines)
//   - KCAPTCHA       (distinctive PNG-font glyphs + wave warp)
//   - Cool PHP Captcha (background + wave + rotation)
//   - Simple PHP Captcha (textured background + shadow text)
//
// Usage:
//   php gen_corpus.php --count=N [--out=DIR] [--min=4] [--max=8] [--seed=S]
//                       [--libs=securimage,gregwar,kcaptcha,coolcaptcha,simplecaptcha]
//
// Output:
//   <out>/shard_NNNNN/<seq>_<label>.png|.jpg
//   <out>/shard_NNNNN/labels.tsv  (filename<TAB>label<TAB>length<TAB>params_json)
//
// The TSV `params_json` always includes a `lib` field so subsets can be filtered downstream.

require __DIR__ . '/lib_common.php';
require __DIR__ . '/../vendor/securimage/securimage.php';
require __DIR__ . '/../vendor/kcaptcha/KCaptcha.php';

/**
 * Securimage normally generates its own random code inside createCode(), and
 * the `display_value` constructor option only takes effect when a captchaId
 * is set (which forces use_database=true). To bind every render to a known
 * label without touching the database, we override createCode() so it copies
 * `forced_code` into $this->code and $this->code_display.
 */
class SecurimageForced extends Securimage {
    public string $forced_code = '';
    public function createCode() {
        if ($this->forced_code !== '') {
            $this->code = $this->case_sensitive ? $this->forced_code : strtolower($this->forced_code);
            $this->code_display = $this->forced_code;
            return;
        }
        parent::createCode();
    }
}

// Minimal PSR-4 autoloader for Gregwar/Captcha (composer-less install)
spl_autoload_register(function ($class) {
    $prefixes = [
        'Gregwar\\Captcha\\' => __DIR__ . '/../vendor/gregwar_captcha/src/Gregwar/Captcha/',
    ];
    foreach ($prefixes as $prefix => $base_dir) {
        if (strpos($class, $prefix) === 0) {
            $rel = str_replace('\\', '/', substr($class, strlen($prefix)));
            $file = $base_dir . $rel . '.php';
            if (is_file($file)) require $file;
        }
    }
});

// Cool PHP Captcha defines class SimpleCaptcha + runs code at top-level
// (session_start, $captcha = new SimpleCaptcha(), $captcha->CreateImage()).
// We need the class without the side effects, so we inline it via a guard.
require_once __DIR__ . '/cool_captcha_class.php';

$opts = getopt('', ['count::', 'out::', 'min::', 'max::', 'seed::', 'libs::']);
$count    = isset($opts['count']) ? (int)$opts['count'] : 50;
$out_base = $opts['out'] ?? (__DIR__ . '/../train_corpus');
$min_len  = isset($opts['min'])   ? (int)$opts['min']   : 4;
$max_len  = isset($opts['max'])   ? (int)$opts['max']   : 8;
$seed     = isset($opts['seed'])  ? (int)$opts['seed']  : null;

$active_libs = isset($opts['libs'])
    ? array_map('trim', explode(',', $opts['libs']))
    : ['securimage', 'gregwar', 'kcaptcha', 'coolcaptcha', 'simplecaptcha'];

seed_random($seed);

$FONTS = font_pool();
$SECURIMAGE_BG = glob(realpath(__DIR__ . '/../vendor/securimage/backgrounds') . '/*.{jpg,png,gif}', GLOB_BRACE) ?: [];
$SIMPLE_BG = glob(realpath(__DIR__ . '/../vendor/simplecaptcha/backgrounds') . '/*.png') ?: [];

// ---- per-library generator functions ---------------------------------------

/** Returns ['bytes'=>..., 'ext'=>..., 'label'=>..., 'params'=>...]. */
function gen_securimage(int $min_len, int $max_len, array $fonts, array $bgs): array {
    [$cs_name, $charset] = pick_charset();
    $length = rand_length($min_len, $max_len);
    $label  = rand_text($charset, $length);

    $width  = mt_rand(160, 280);
    $height = mt_rand(60, 100);
    $bg_rgb   = rand_rgb(200, 255);
    $text_rgb = rand_rgb(0, 110);
    $line_rgb = rand_rgb(80, 200);
    $noise_rgb = rand_rgb(80, 200);
    $font = pick($fonts);

    // Colors must be set via constructor — Securimage wraps them into
    // Securimage_Color objects in __construct(); reassigning a hex string
    // afterward leaves the property as a raw string and breaks rendering.
    $si = new SecurimageForced([
        'send_headers'   => false,
        'no_exit'        => true,
        'no_session'     => true,
        'image_bg_color' => rgb_to_hex($bg_rgb),
        'text_color'     => rgb_to_hex($text_rgb),
        'line_color'     => rgb_to_hex($line_rgb),
        'noise_color'    => rgb_to_hex($noise_rgb),
    ]);
    $si->forced_code  = $label;
    $si->image_width  = $width;
    $si->image_height = $height;
    $si->code_length  = $length;
    $si->charset      = $charset;
    $si->case_sensitive = true;
    $si->use_transparent_text         = (bool)mt_rand(0, 1);
    $si->text_transparency_percentage = mt_rand(0, 50);
    $si->use_random_spaces   = (bool)mt_rand(0, 1);
    $si->use_text_angles     = (bool)mt_rand(0, 1);
    $si->use_random_baseline = (bool)mt_rand(0, 1);
    $si->use_random_boxes    = (mt_rand(0, 4) === 0);
    $si->perturbation = mt_rand(0, 120) / 100.0;
    $si->num_lines    = mt_rand(0, 12);
    $si->noise_level  = mt_rand(0, 8);
    $si->ttf_file = $font;
    $si->signature_font = $font;
    $si->image_signature = '';
    $si->image_type = Securimage::SI_IMAGE_PNG;
    $si->display_value = $label;

    $bg_path = '';
    if ($bgs && mt_rand(0, 4) === 0) $bg_path = pick($bgs);

    ob_start();
    try {
        $si->show($bg_path);
    } catch (Throwable $e) {
        ob_end_clean();
        return [];
    }
    $bytes = ob_get_clean();
    if (!$bytes || strlen($bytes) < 100) return [];

    return [
        'bytes' => $bytes,
        'ext'   => 'png',
        'label' => $label,
        'params' => [
            'lib' => 'securimage', 'w' => $width, 'h' => $height, 'cs' => $cs_name,
            'font' => basename($font),
            'bg' => rgb_to_hex($bg_rgb), 'text' => rgb_to_hex($text_rgb),
            'line' => rgb_to_hex($line_rgb), 'noise' => rgb_to_hex($noise_rgb),
            'tt' => $si->use_transparent_text, 'tp' => $si->text_transparency_percentage,
            'rs' => $si->use_random_spaces, 'ta' => $si->use_text_angles,
            'rb' => $si->use_random_baseline, 'rx' => $si->use_random_boxes,
            'pert' => $si->perturbation, 'lines' => $si->num_lines, 'nl' => $si->noise_level,
            'bgimg' => $bg_path ? basename($bg_path) : '',
        ],
    ];
}

function gen_gregwar(int $min_len, int $max_len, array $fonts): array {
    [$cs_name, $charset] = pick_charset();
    $length = rand_length($min_len, $max_len);
    $label  = rand_text($charset, $length);

    $width  = mt_rand(150, 280);
    $height = mt_rand(40, 100);
    $font   = pick($fonts);

    $cb = new \Gregwar\Captcha\CaptchaBuilder($label);
    $cb->setDistortion((bool)mt_rand(0, 1));
    $cb->setMaxAngle(mt_rand(0, 18));
    $cb->setMaxOffset(mt_rand(0, 8));
    $cb->setIgnoreAllEffects(false);
    $cb->setInterpolation((bool)mt_rand(0, 1));
    $cb->setMaxBehindLines(mt_rand(0, 6));
    $cb->setMaxFrontLines(mt_rand(0, 6));
    $cb->setScatterEffect((bool)mt_rand(0, 1));

    $tc = rand_rgb(0, 110);
    $bg = rand_rgb(200, 255);
    $lc = rand_rgb(0, 220);
    $cb->setTextColor($tc[0], $tc[1], $tc[2]);
    $cb->setBackgroundColor($bg[0], $bg[1], $bg[2]);
    $cb->setLineColor($lc[0], $lc[1], $lc[2]);

    try {
        $cb->build($width, $height, $font);
    } catch (Throwable $e) {
        return [];
    }
    // get() returns JPEG bytes
    $bytes = $cb->get(mt_rand(70, 95));
    if (!$bytes) return [];

    return [
        'bytes' => $bytes, 'ext' => 'jpg', 'label' => $label,
        'params' => [
            'lib' => 'gregwar', 'w' => $width, 'h' => $height, 'cs' => $cs_name,
            'font' => basename($font),
            'bg' => rgb_to_hex($bg), 'text' => rgb_to_hex($tc), 'line' => rgb_to_hex($lc),
            'angle' => $cb->getMaxAngle ?? 0, 'offset' => $cb->getMaxOffset ?? 0,
        ],
    ];
}

function gen_kcaptcha(): array {
    // KCaptcha uses fixed PNG-rendered fonts and a restricted alphabet (23456789abcdegikpqsvxyz).
    // It generates its own random key; we capture the JPEG and read keyString.
    $kc = null;
    ob_start();
    try {
        // Allow length override (5, 6, or 7)
        $kc = new \k_captcha\KCaptcha();
    } catch (Throwable $e) {
        ob_end_clean();
        return [];
    }
    $bytes = ob_get_clean();
    if (!$kc || !$bytes || strlen($bytes) < 100) return [];
    $label = $kc->getKeyString();

    return [
        'bytes' => $bytes, 'ext' => 'jpg', 'label' => $label,
        'params' => [
            'lib' => 'kcaptcha', 'w' => $kc->width, 'h' => $kc->height,
            'fluct' => $kc->fluctuationAmplitude, 'wnd' => $kc->whiteNoiseDensity,
            'bnd' => $kc->blackNoiseDensity, 'no_spaces' => $kc->noSpaces,
            'fg' => sprintf('#%02x%02x%02x', $kc->foregroundColor[0], $kc->foregroundColor[1], $kc->foregroundColor[2]),
            'bg' => sprintf('#%02x%02x%02x', $kc->backgroundColor[0], $kc->backgroundColor[1], $kc->backgroundColor[2]),
        ],
    ];
}

function gen_coolcaptcha(int $min_len, int $max_len, array $fonts): array {
    [$cs_name, $charset] = pick_charset();
    $length = rand_length($min_len, $max_len);
    $label  = rand_text($charset, $length);

    $cap = new \CoolCaptchaWrapper();
    $cap->width  = mt_rand(160, 260);
    $cap->height = mt_rand(50, 90);
    $cap->scale  = mt_rand(2, 3);
    $cap->blur   = (mt_rand(0, 3) === 0);
    $cap->difficulty = mt_rand(50, 180) / 100.0;
    $cap->Yperiod    = mt_rand(8, 16);
    $cap->Yamplitude = mt_rand(6, 18);
    $cap->Xperiod    = mt_rand(8, 14);
    $cap->Xamplitude = mt_rand(2, 8);
    $cap->maxRotation = mt_rand(2, 14);
    $cap->lineWidth  = mt_rand(0, 4);
    $cap->wordsFile  = '';                // no dictionary, force random
    $cap->minWordLength = $length;
    $cap->maxWordLength = $length;
    $cap->backgroundColor = rand_rgb(220, 255);
    $cap->colors = [rand_rgb(0, 100), rand_rgb(0, 100), rand_rgb(0, 100)];
    $cap->shadowColor = (mt_rand(0, 2) === 0) ? rand_rgb(150, 255) : null;
    $cap->resourcesPath = realpath(__DIR__ . '/../vendor/coolcaptcha/resources');
    $cap->imageFormat = 'png';
    // Force our label
    $cap->forced_text = $label;
    // Limit fonts to library's bundled list (sizes are tuned)
    // (already wired via $cap->fonts)
    // Pick a font to record
    $fontkey = array_rand($cap->fonts);

    ob_start();
    try {
        $cap->CreateImage();
    } catch (Throwable $e) {
        ob_end_clean();
        return [];
    }
    $bytes = ob_get_clean();
    if (!$bytes || strlen($bytes) < 100) return [];

    return [
        'bytes' => $bytes, 'ext' => 'png', 'label' => $label,
        'params' => [
            'lib' => 'coolcaptcha', 'w' => $cap->width, 'h' => $cap->height,
            'cs' => $cs_name, 'difficulty' => $cap->difficulty,
            'scale' => $cap->scale, 'blur' => $cap->blur,
            'Yp' => $cap->Yperiod, 'Ya' => $cap->Yamplitude,
            'Xp' => $cap->Xperiod, 'Xa' => $cap->Xamplitude,
            'rot' => $cap->maxRotation, 'line' => $cap->lineWidth,
            'font_key' => $fontkey,
        ],
    ];
}

function gen_simplecaptcha(int $min_len, int $max_len, array $fonts, array $bgs): array {
    [$cs_name, $charset] = pick_charset();
    $length = rand_length($min_len, $max_len);
    $label  = rand_text($charset, $length);

    if (!$bgs) return [];
    $bg = pick($bgs);
    $info = @getimagesize($bg);
    if (!$info) return [];
    [$w, $h] = $info;
    $im = @imagecreatefrompng($bg);
    if (!$im) return [];

    $color_rgb = rand_rgb(0, 110);
    $color = imagecolorallocate($im, $color_rgb[0], $color_rgb[1], $color_rgb[2]);
    $shadow_rgb = rand_rgb(180, 255);
    $shadow = imagecolorallocate($im, $shadow_rgb[0], $shadow_rgb[1], $shadow_rgb[2]);

    $angle_min = 0;
    $angle_max = mt_rand(0, 25);
    $angle = mt_rand($angle_min, $angle_max) * (mt_rand(0, 1) ? -1 : 1);

    $font = pick($fonts);
    $size = mt_rand(20, 36);

    $bbox = @imagettfbbox($size, $angle, $font, $label);
    if (!$bbox) { imagedestroy($im); return []; }
    $box_w = abs($bbox[6] - $bbox[2]);
    $box_h = abs($bbox[5] - $bbox[1]);
    if ($box_w >= $w - 4 || $box_h >= $h - 4) {
        // text won't fit, scale down once
        $size = max(14, (int)($size * 0.7));
        $bbox = @imagettfbbox($size, $angle, $font, $label);
        if (!$bbox) { imagedestroy($im); return []; }
        $box_w = abs($bbox[6] - $bbox[2]);
        $box_h = abs($bbox[5] - $bbox[1]);
        if ($box_w >= $w - 4 || $box_h >= $h - 4) { imagedestroy($im); return []; }
    }
    $px = mt_rand(2, max(2, $w - $box_w - 2));
    $py = mt_rand(max($box_h, 2), max($box_h + 1, $h - 4));

    // Shadow
    if (mt_rand(0, 1)) {
        $sx = mt_rand(-2, 2); $sy = mt_rand(-2, 2);
        @imagettftext($im, $size, $angle, $px + $sx, $py + $sy, $shadow, $font, $label);
    }
    @imagettftext($im, $size, $angle, $px, $py, $color, $font, $label);

    ob_start();
    imagepng($im);
    $bytes = ob_get_clean();
    imagedestroy($im);

    return [
        'bytes' => $bytes, 'ext' => 'png', 'label' => $label,
        'params' => [
            'lib' => 'simplecaptcha', 'w' => $w, 'h' => $h, 'cs' => $cs_name,
            'font' => basename($font), 'size' => $size, 'angle' => $angle,
            'bg' => basename($bg), 'text' => rgb_to_hex($color_rgb),
            'shadow' => rgb_to_hex($shadow_rgb),
        ],
    ];
}

// ---- main loop -------------------------------------------------------------

$libs_pool = [];
foreach ($active_libs as $lib) {
    if (in_array($lib, ['securimage','gregwar','kcaptcha','coolcaptcha','simplecaptcha'], true)) {
        $libs_pool[] = $lib;
    }
}
if (!$libs_pool) {
    fwrite(STDERR, "no valid libs selected\n");
    exit(1);
}

$produced = 0;
$attempts = 0;
$max_attempts = $count * 4;
$lib_counts = array_fill_keys($libs_pool, 0);

while ($produced < $count && $attempts < $max_attempts) {
    $attempts++;
    $lib = pick($libs_pool);
    $r = [];
    try {
        switch ($lib) {
            case 'securimage':    $r = gen_securimage($min_len, $max_len, $FONTS, $SECURIMAGE_BG); break;
            case 'gregwar':       $r = gen_gregwar($min_len, $max_len, $FONTS); break;
            case 'kcaptcha':      $r = gen_kcaptcha(); break;
            case 'coolcaptcha':   $r = gen_coolcaptcha($min_len, $max_len, $FONTS); break;
            case 'simplecaptcha': $r = gen_simplecaptcha($min_len, $max_len, $FONTS, $SIMPLE_BG); break;
        }
    } catch (Throwable $e) {
        fwrite(STDERR, "[$lib] error: " . $e->getMessage() . "\n");
        continue;
    }
    if (!$r || empty($r['bytes'])) continue;

    [$shard_dir, $shard_id] = next_shard($out_base);
    $produced++;
    $lib_counts[$lib]++;
    // Combine high-resolution time, PID, and produced count to make filenames
    // unique even when many workers write to the same train_corpus concurrently.
    $seq = sprintf('%d_%d_%05d', (int)(microtime(true) * 1e6), getmypid(), $produced);
    $fname = output_filename_v2($seq, $r['label'], $r['ext']);
    file_put_contents($shard_dir . '/' . $fname, $r['bytes']);
    $r['params']['shard'] = $shard_id;
    append_label($shard_dir, $fname, $r['label'], $r['params']);

    if ($produced % 100 === 0) {
        $msg = "[corpus] $produced/$count: ";
        foreach ($lib_counts as $k => $v) $msg .= "$k=$v ";
        fprintf(STDERR, "%s\n", $msg);
    }
}

fprintf(STDERR, "[corpus] done: %d images (%d attempts). breakdown: %s\n",
    $produced, $attempts, json_encode($lib_counts));
