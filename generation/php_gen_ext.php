<?php
/**
 * php_gen_ext.php — color-aware fork of cryptbb_sourcefree/src/php_gen.php.
 *
 * Provenance: copied from cryptbb_sourcefree/src/php_gen.php (renders OUR chosen
 * text with generic fonts + GD primitives: imagettftext + sine warp + noise, so
 * synth is pixel-faithful to how real PHP captchas render, and stays source-free).
 *
 * ADDED (backward-compatible): an optional cfg["colors"] block
 *   { "bg_hex":"#rrggbb", "glyph_hex":"#rrggbb", "noise_hex":"#rrggbb" }
 * read from the real captcha's pixels by director.py. When present, the bg base
 * and (for bw glyphs) the glyph colour are pulled toward those hues instead of the
 * original hardcoded random ranges. Absent => identical to the original generator.
 *
 *   php php_gen_ext.php jobs.tsv config.json
 */

function rnd($a, $b) { return $a + mt_rand() / mt_getrandmax() * ($b - $a); }
function rndi($a, $b) { return mt_rand($a, (int)$b); }
function jit($v, $d) { return max(0, min(255, (int)$v + mt_rand(-$d, $d))); }
function hex2rgb($h) { $h = ltrim($h, '#'); if (strlen($h) < 6) return null;
    return [hexdec(substr($h,0,2)), hexdec(substr($h,2,2)), hexdec(substr($h,4,2))]; }
function cfg_color($cfg, $key) {
    if (empty($cfg['colors']) || empty($cfg['colors'][$key])) return null;
    return hex2rgb($cfg['colors'][$key]);
}

function make_bg($im, $W, $H, $cfg) {
    $bghex = cfg_color($cfg, 'bg_hex');
    if ($cfg['bg'] === 'dark') {
        $base = $bghex
            ? imagecolorallocate($im, jit($bghex[0],12), jit($bghex[1],12), jit($bghex[2],12))
            : imagecolorallocate($im, rndi(40,90), rndi(40,90), rndi(40,90));
        imagefilledrectangle($im, 0, 0, $W, $H, $base);
        for ($i = 0; $i < rndi(14,34); $i++) {
            $c = $bghex
                ? imagecolorallocatealpha($im, jit($bghex[0],28), jit($bghex[1],28), jit($bghex[2],28), rndi(30,90))
                : imagecolorallocatealpha($im, rndi(35,130), rndi(35,130), rndi(35,130), rndi(30,90));
            $x = rndi(0,$W); $y = rndi(0,$H);
            imagefilledrectangle($im, $x, $y, $x+rndi(10,60), $y+rndi(8,$H), $c);
        }
    } else {
        $base = $bghex
            ? imagecolorallocate($im, jit($bghex[0],8), jit($bghex[1],8), jit($bghex[2],8))
            : imagecolorallocate($im, rndi(235,255), rndi(235,255), rndi(235,255));
        imagefilledrectangle($im, 0, 0, $W, $H, $base);
        for ($i = 0; $i < rndi(35,70); $i++) {           // pastel tile mosaic
            $c = $bghex
                ? imagecolorallocatealpha($im, jit($bghex[0],22), jit($bghex[1],22), jit($bghex[2],22), rndi(60,110))
                : imagecolorallocatealpha($im, rndi(175,245), rndi(175,245), rndi(175,245), rndi(60,110));
            $x = rndi(0,$W); $y = rndi(0,$H); $s = rndi(7,20);
            imagefilledrectangle($im, $x, $y, $x+$s, $y+$s, $c);
        }
        for ($i = 0; $i < $W*$H*rnd(0.03,0.09); $i++)     // white speckle
            imagesetpixel($im, rndi(0,$W-1), rndi(0,$H-1), imagecolorallocate($im,255,255,255));
        if (!empty($cfg['grid'])) {
            $gc = imagecolorallocatealpha($im, rndi(150,210), rndi(150,210), rndi(180,220), rndi(75,105));
            $step = rndi(10,18); $off = rndi(0,$step);
            for ($gx = $off; $gx < $W; $gx += $step) imageline($im, $gx, 0, $gx, $H, $gc);
            for ($gy = $off; $gy < $H; $gy += $step) imageline($im, 0, $gy, $W, $gy, $gc);
        }
    }
}

function glyph_color($im, $cfg) {
    if ($cfg['charset_style'] === 'bw') {
        $gh = cfg_color($cfg, 'glyph_hex');
        if ($gh) return imagecolorallocate($im, jit($gh[0],18), jit($gh[1],18), jit($gh[2],18));
        return (mt_rand(0,1) ? imagecolorallocate($im, rndi(0,45),rndi(0,45),rndi(0,45))
                             : imagecolorallocate($im, rndi(210,255),rndi(210,255),rndi(210,255)));
    }
    $h = rnd(0,1)*6; $x = 1 - abs(fmod($h,2)-1);
    $r=$g=$b=0;
    if ($h<1){$r=1;$g=$x;} elseif($h<2){$r=$x;$g=1;} elseif($h<3){$g=1;$b=$x;}
    elseif($h<4){$g=$x;$b=1;} elseif($h<5){$r=$x;$b=1;} else {$r=1;$b=$x;}
    $v = rnd(0.45,0.8)*255;
    return imagecolorallocate($im, (int)($r*$v), (int)($g*$v), (int)($b*$v));
}

function wave_warp($src, $W, $H, $cfg) {
    $dst = imagecreatetruecolor($W, $H);
    $ampx = rnd($cfg['wave']['ampx'][0], $cfg['wave']['ampx'][1]);
    $ampy = rnd($cfg['wave']['ampy'][0], $cfg['wave']['ampy'][1]);
    $perx = rnd($cfg['wave']['per'][0], $cfg['wave']['per'][1]) / $H;
    $pery = rnd($cfg['wave']['per'][0], $cfg['wave']['per'][1]) / $W;
    $phx = rnd(0, 6.28); $phy = rnd(0, 6.28);
    for ($y = 0; $y < $H; $y++) {
        for ($x = 0; $x < $W; $x++) {
            $sx = (int)round($x + $ampx * sin(6.283 * $perx * $y + $phx));
            $sy = (int)round($y + $ampy * sin(6.283 * $pery * $x + $phy));
            $sx = max(0, min($W-1, $sx)); $sy = max(0, min($H-1, $sy));
            imagesetpixel($dst, $x, $y, imagecolorat($src, $sx, $sy));
        }
    }
    return $dst;
}

function add_noise($im, $W, $H, $cfg) {
    $nh = cfg_color($cfg, 'noise_hex');
    for ($i = 0; $i < rndi($cfg['lines'][0], $cfg['lines'][1]); $i++) {
        if ($nh) $c = imagecolorallocatealpha($im, jit($nh[0],25), jit($nh[1],25), jit($nh[2],25), rndi(45,95));
        elseif ($cfg['charset_style']==='bw' && mt_rand(0,1))
             $c = imagecolorallocatealpha($im, 255,255,255, rndi(40,90));
        else $c = imagecolorallocatealpha($im, rndi(80,190),rndi(80,190),rndi(80,190), rndi(50,100));
        imageline($im, rndi(0,$W), rndi(0,$H), rndi(0,$W), rndi(0,$H), $c);
    }
    for ($i = 0; $i < rndi($cfg['arcs'][0], $cfg['arcs'][1]); $i++) {
        $c = $nh ? imagecolorallocate($im, $nh[0],$nh[1],$nh[2])
                 : imagecolorallocate($im, rndi(0,200),rndi(0,200),rndi(0,200));
        imagearc($im, rndi(0,$W), rndi(0,$H), rndi($W/3,$W), rndi($H/2,$H*2), rndi(0,360), rndi(0,360), $c);
    }
    $nd = $W*$H*rnd($cfg['dots'][0], $cfg['dots'][1]);
    for ($i = 0; $i < $nd; $i++)
        imagesetpixel($im, rndi(0,$W-1), rndi(0,$H-1), imagecolorallocate($im, rndi(0,255),rndi(0,255),rndi(0,255)));
}

function render($text, $out, $cfg) {
    $W = $cfg['W']; $H = $cfg['H'];
    $im = imagecreatetruecolor($W, $H);
    make_bg($im, $W, $H, $cfg);
    if (!empty($cfg['bg_only'])) {   // background+noise only (real glyphs composited in Python)
        add_noise($im, $W, $H, $cfg);
        imagepng($im, $out); imagedestroy($im); return;
    }
    $n = strlen($text);
    $sz = rndi($H*$cfg['min_size_frac'], $H*$cfg['max_size_frac']);
    $font = $cfg['fonts'][array_rand($cfg['fonts'])];
    $widths = []; $tot = 0;
    for ($i = 0; $i < $n; $i++) {
        $bb = imagettfbbox($sz, 0, $font, $text[$i]);
        $w = max(abs($bb[2]-$bb[0]), $sz/3); $widths[] = $w; $tot += $w;
    }
    $gap = rndi(2, 6); $tot += $gap*($n-1);
    $x = max(2, (int)(($W-$tot)/2 + rndi(-4,4)));
    for ($i = 0; $i < $n; $i++) {
        $ang = rnd(-$cfg['max_rot'], $cfg['max_rot']);
        $col = glyph_color($im, $cfg);
        $y = (int)($H*0.5 + $sz*0.35 + rndi(-3,3));
        imagettftext($im, $sz, $ang, $x, $y, $col, $font, $text[$i]);
        $x += $widths[$i] + $gap;
    }
    if (!empty($cfg['wave']['on'])) { $w = wave_warp($im, $W, $H, $cfg); imagedestroy($im); $im = $w; }
    add_noise($im, $W, $H, $cfg);
    imagepng($im, $out);
    imagedestroy($im);
}

$jobs = $argv[1]; $cfg = json_decode(file_get_contents($argv[2]), true);
$fh = fopen($jobs, 'r'); $k = 0;
while (($line = fgets($fh)) !== false) {
    $line = trim($line); if ($line === '') continue;
    $parts = explode("\t", $line); if (count($parts) < 2) continue;
    render($parts[0], $parts[1], $cfg);
    if (++$k % 2000 === 0) { fwrite(STDERR, "rendered $k\n"); }
}
fclose($fh);
fwrite(STDERR, "done $k\n");
