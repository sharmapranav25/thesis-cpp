<?php
// Shared helpers for all PHP captcha generators.
// All generators share the same output format and font pool so the corpus is uniform.

/**
 * Returns absolute paths to TTF fonts available on this machine
 * (vendored captcha library fonts + a curated set of Windows system fonts).
 */
function font_pool(): array {
    $vendor = realpath(__DIR__ . '/../vendor');
    $vendored = [
        'coolcaptcha/resources/fonts/AntykwaBold.ttf',
        'coolcaptcha/resources/fonts/Candice.ttf',
        'coolcaptcha/resources/fonts/Ding-DongDaddyO.ttf',
        'coolcaptcha/resources/fonts/Duality.ttf',
        'coolcaptcha/resources/fonts/Heineken.ttf',
        'coolcaptcha/resources/fonts/Jura.ttf',
        'coolcaptcha/resources/fonts/StayPuft.ttf',
        'coolcaptcha/resources/fonts/TimesNewRomanBold.ttf',
        'coolcaptcha/resources/fonts/VeraSansBold.ttf',
        'gregwar_captcha/src/Gregwar/Captcha/Font/captcha0.ttf',
        'gregwar_captcha/src/Gregwar/Captcha/Font/captcha1.ttf',
        'gregwar_captcha/src/Gregwar/Captcha/Font/captcha2.ttf',
        'gregwar_captcha/src/Gregwar/Captcha/Font/captcha3.ttf',
        'gregwar_captcha/src/Gregwar/Captcha/Font/captcha4.ttf',
        'gregwar_captcha/src/Gregwar/Captcha/Font/captcha5.ttf',
        'gregwar_image/demo/fonts/CaviarDreams.ttf',
        'mewebstudio_captcha/assets/fonts/ABeeZee_regular.ttf',
        'mewebstudio_captcha/assets/fonts/Asap_700.ttf',
        'mewebstudio_captcha/assets/fonts/Khand_500.ttf',
        'mewebstudio_captcha/assets/fonts/Open_Sans_regular.ttf',
        'mewebstudio_captcha/assets/fonts/Roboto_regular.ttf',
        'mewebstudio_captcha/assets/fonts/Ubuntu_regular.ttf',
        'securimage/AHGBold.ttf',
        'simplecaptcha/fonts/times_new_yorker.ttf',
    ];
    $system = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/arialbi.ttf',
        'C:/Windows/Fonts/ariali.ttf',
        'C:/Windows/Fonts/ariblk.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/calibrib.ttf',
        'C:/Windows/Fonts/calibrii.ttf',
        'C:/Windows/Fonts/calibril.ttf',
        'C:/Windows/Fonts/comic.ttf',
        'C:/Windows/Fonts/comicbd.ttf',
        'C:/Windows/Fonts/consola.ttf',
        'C:/Windows/Fonts/consolab.ttf',
        'C:/Windows/Fonts/consolai.ttf',
        'C:/Windows/Fonts/cour.ttf',
        'C:/Windows/Fonts/courbd.ttf',
        'C:/Windows/Fonts/couri.ttf',
        'C:/Windows/Fonts/Gabriola.ttf',
        'C:/Windows/Fonts/georgia.ttf',
        'C:/Windows/Fonts/georgiab.ttf',
        'C:/Windows/Fonts/georgiai.ttf',
        'C:/Windows/Fonts/impact.ttf',
        'C:/Windows/Fonts/lucon.ttf',
        'C:/Windows/Fonts/l_10646.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
        'C:/Windows/Fonts/segoeuib.ttf',
        'C:/Windows/Fonts/segoeuii.ttf',
        'C:/Windows/Fonts/segoeuil.ttf',
        'C:/Windows/Fonts/segoeuisl.ttf',
        'C:/Windows/Fonts/seguibl.ttf',
        'C:/Windows/Fonts/seguisb.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        'C:/Windows/Fonts/tahomabd.ttf',
        'C:/Windows/Fonts/times.ttf',
        'C:/Windows/Fonts/timesbd.ttf',
        'C:/Windows/Fonts/timesbi.ttf',
        'C:/Windows/Fonts/timesi.ttf',
        'C:/Windows/Fonts/trebuc.ttf',
        'C:/Windows/Fonts/trebucbd.ttf',
        'C:/Windows/Fonts/trebucbi.ttf',
        'C:/Windows/Fonts/trebucit.ttf',
        'C:/Windows/Fonts/verdana.ttf',
        'C:/Windows/Fonts/verdanab.ttf',
        'C:/Windows/Fonts/verdanai.ttf',
        'C:/Windows/Fonts/verdanaz.ttf',
        'C:/Windows/Fonts/BAUHS93.TTF',
        'C:/Windows/Fonts/BRADHITC.TTF',
        'C:/Windows/Fonts/BRLNSB.TTF',
        'C:/Windows/Fonts/BRLNSDB.TTF',
        'C:/Windows/Fonts/BRLNSR.TTF',
        'C:/Windows/Fonts/BRUSHSCI.TTF',
        'C:/Windows/Fonts/CENTAUR.TTF',
        'C:/Windows/Fonts/CENTURY.TTF',
        'C:/Windows/Fonts/CHILLER.TTF',
        'C:/Windows/Fonts/COLONNA.TTF',
        'C:/Windows/Fonts/COOPBL.TTF',
        'C:/Windows/Fonts/FORTE.TTF',
        'C:/Windows/Fonts/FRABK.TTF',
        'C:/Windows/Fonts/FRADM.TTF',
        'C:/Windows/Fonts/FRAHV.TTF',
        'C:/Windows/Fonts/FREESCPT.TTF',
        'C:/Windows/Fonts/FRSCRIPT.TTF',
        'C:/Windows/Fonts/GIL_____.TTF',
        'C:/Windows/Fonts/GILBI___.TTF',
        'C:/Windows/Fonts/GILB____.TTF',
        'C:/Windows/Fonts/GILI____.TTF',
        'C:/Windows/Fonts/GILSANUB.TTF',
        'C:/Windows/Fonts/GOTHIC.TTF',
        'C:/Windows/Fonts/GOTHICB.TTF',
        'C:/Windows/Fonts/GOTHICBI.TTF',
        'C:/Windows/Fonts/GOTHICI.TTF',
        'C:/Windows/Fonts/HARLOWSI.TTF',
        'C:/Windows/Fonts/HTOWERT.TTF',
        'C:/Windows/Fonts/HTOWERTI.TTF',
        'C:/Windows/Fonts/JUICE___.TTF',
        'C:/Windows/Fonts/LBRITE.TTF',
        'C:/Windows/Fonts/LBRITED.TTF',
        'C:/Windows/Fonts/LBRITEI.TTF',
        'C:/Windows/Fonts/LCALLIG.TTF',
        'C:/Windows/Fonts/LFAX.TTF',
        'C:/Windows/Fonts/LSANS.TTF',
        'C:/Windows/Fonts/MAGNETOB.TTF',
        'C:/Windows/Fonts/MOD20.TTF',
        'C:/Windows/Fonts/MTCORSVA.TTF',
        'C:/Windows/Fonts/NIAGENG.TTF',
        'C:/Windows/Fonts/NIAGSOL.TTF',
        'C:/Windows/Fonts/OCRAEXT.TTF',
        'C:/Windows/Fonts/OLDENGL.TTF',
        'C:/Windows/Fonts/PALSCRI.TTF',
        'C:/Windows/Fonts/PAPYRUS.TTF',
        'C:/Windows/Fonts/PARCHM.TTF',
        'C:/Windows/Fonts/PERTILI.TTF',
        'C:/Windows/Fonts/PER_____.TTF',
        'C:/Windows/Fonts/PERBI___.TTF',
        'C:/Windows/Fonts/PERB____.TTF',
        'C:/Windows/Fonts/PERI____.TTF',
        'C:/Windows/Fonts/PLAYBILL.TTF',
        'C:/Windows/Fonts/RAGE.TTF',
        'C:/Windows/Fonts/RAVIE.TTF',
        'C:/Windows/Fonts/ROCK.TTF',
        'C:/Windows/Fonts/SCRIPTBL.TTF',
        'C:/Windows/Fonts/SHOWG.TTF',
        'C:/Windows/Fonts/SNAP____.TTF',
        'C:/Windows/Fonts/STENCIL.TTF',
        'C:/Windows/Fonts/TCB_____.TTF',
        'C:/Windows/Fonts/TCBI____.TTF',
        'C:/Windows/Fonts/TCCB____.TTF',
        'C:/Windows/Fonts/TCCEB.TTF',
        'C:/Windows/Fonts/TCCM____.TTF',
        'C:/Windows/Fonts/TEMPSITC.TTF',
        'C:/Windows/Fonts/VINERITC.TTF',
        'C:/Windows/Fonts/VIVALDII.TTF',
        'C:/Windows/Fonts/VLADIMIR.TTF',
    ];
    $out = [];
    foreach ($vendored as $rel) {
        $p = $vendor . '/' . $rel;
        if (is_file($p)) $out[] = $p;
    }
    foreach ($system as $p) {
        if (is_file($p)) $out[] = $p;
    }
    return $out;
}

/** Pick one element from an array uniformly. */
function pick(array $a) {
    return $a[mt_rand(0, count($a) - 1)];
}

/** Random captcha length in [$min, $max] inclusive. */
function rand_length(int $min = 4, int $max = 8): int {
    return mt_rand($min, $max);
}

/** Random character set descriptor. Returns [name, charset_string]. */
function pick_charset(): array {
    $sets = [
        ['mixed_alnum_safe', 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789'],
        ['mixed_alnum_full', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'],
        ['lower_alnum',      'abcdefghijklmnopqrstuvwxyz0123456789'],
        ['upper_alnum',      'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'],
        ['lower_only',       'abcdefghijklmnopqrstuvwxyz'],
        ['upper_only',       'ABCDEFGHIJKLMNOPQRSTUVWXYZ'],
        ['digits_only',      '0123456789'],
        ['mixed_no_amb',     'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'],
    ];
    return pick($sets);
}

/** Random RGB triple. */
function rand_rgb(int $lo = 0, int $hi = 255): array {
    return [mt_rand($lo, $hi), mt_rand($lo, $hi), mt_rand($lo, $hi)];
}

/** Generate a random text of given length from charset. */
function rand_text(string $charset, int $length): string {
    $out = '';
    $n = strlen($charset);
    for ($i = 0; $i < $length; $i++) {
        $out .= $charset[mt_rand(0, $n - 1)];
    }
    return $out;
}

/** RGB array -> "#rrggbb" hex. */
function rgb_to_hex(array $rgb): string {
    return sprintf('#%02x%02x%02x', $rgb[0], $rgb[1], $rgb[2]);
}

/** Make a sharded output dir for a generator. Returns array [shard_dir, shard_id]. */
function next_shard(string $base, int $images_per_shard = 1000): array {
    if (!is_dir($base)) mkdir($base, 0777, true);
    $shards = glob($base . '/shard_*', GLOB_ONLYDIR);
    $next = 0;
    if ($shards) {
        $last = end($shards);
        if (preg_match('/shard_(\d+)$/', $last, $m)) {
            $count = count(glob($last . '/*.png')) + count(glob($last . '/*.jpg'));
            if ($count < $images_per_shard) {
                return [$last, (int)$m[1]];
            }
            $next = (int)$m[1] + 1;
        }
    }
    $dir = sprintf('%s/shard_%05d', $base, $next);
    if (!is_dir($dir)) mkdir($dir, 0777, true);
    return [$dir, $next];
}

/** Append one TSV row: filename<TAB>label<TAB>length<TAB>params_json. */
function append_label(string $shard_dir, string $filename, string $label, array $params): void {
    $line = $filename . "\t" . $label . "\t" . strlen($label) . "\t" . json_encode($params, JSON_UNESCAPED_SLASHES) . "\n";
    file_put_contents($shard_dir . '/labels.tsv', $line, FILE_APPEND | LOCK_EX);
}

/** Filesystem-safe filename from a sequence number and label. */
function output_filename(int $seq, string $label, string $ext = 'png'): string {
    // labels can have characters like 0-9A-Za-z; strip anything else just in case
    $safe = preg_replace('/[^A-Za-z0-9]/', '_', $label);
    return sprintf('%07d_%s.%s', $seq, $safe, $ext);
}

/** Variant accepting a string-form id (e.g. 'time_pid_count'). */
function output_filename_v2(string $seq, string $label, string $ext = 'png'): string {
    $safe = preg_replace('/[^A-Za-z0-9]/', '_', $label);
    return sprintf('%s_%s.%s', $seq, $safe, $ext);
}

/** Seed mt_rand from a strong source, optionally per process. */
function seed_random(?int $seed = null): void {
    if ($seed === null) {
        $seed = (int)(microtime(true) * 1e6) ^ getmypid();
    }
    mt_srand($seed);
    srand($seed);
}
