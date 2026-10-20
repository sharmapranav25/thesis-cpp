<?php
// Separate adapter: downloaded source bytes are never edited.
declare(strict_types=1);
$cfg = json_decode(file_get_contents($argv[1]), true, 512, JSON_THROW_ON_ERROR);
$warnings = [];
set_error_handler(function($severity, $message) use (&$warnings) {
    if (!isset($warnings[$message])) $warnings[$message] = 0;
    $warnings[$message]++;
    if ($severity !== E_DEPRECATED && $severity !== E_USER_DEPRECATED) {
        throw new ErrorException($message, 0, $severity);
    }
    return true;
});
function snapshot($im, $name) {
    global $cfg;
    imagepng($im, $cfg['trace_dir'].'/'.$name.'.png');
}
try {
    $allowed = ['backend','source','font','font_dir','label','seed','width','height','mode','output','trace_dir','background','font_size','random_tape_replay','capture_random_tape'];
    if (array_diff(array_keys($cfg), $allowed)) throw new InvalidArgumentException('Unknown option');
    if (!preg_match('/^[A-Za-z0-9]{3,8}$/D', $cfg['label'])) throw new InvalidArgumentException('Invalid text');
    if ($cfg['width'] < 160 || $cfg['width'] > 400 || $cfg['height'] < 60 || $cfg['height'] > 160) throw new InvalidArgumentException('Unsupported canvas');
    if (!is_file($cfg['font'])) throw new InvalidArgumentException('Missing explicit font');
    $modes = [
        'gregwar'=>['clean','lines','warp','lines_warp'],
        'securimage'=>['clean','curves','dots','curves_dots_warp'],
        'mobicms'=>['single_font','mixed_fonts','single_font','mixed_fonts'],
        'simple'=>['clean','shadow','rotation','shadow_rotation']
    ];
    if (!isset($modes[$cfg['backend']]) || !in_array($cfg['mode'], $modes[$cfg['backend']], true)) throw new InvalidArgumentException('Unsupported mode');
    mt_srand($cfg['seed']);
    $meta = ['requested'=>$cfg, 'actual'=>[], 'output_type'=>'complete_image',
             'feature_labels'=>'unknown_pending_visibility_and_attribution_checks'];
    $src=$cfg['source']; $mode=$cfg['mode'];
    if ($cfg['backend']==='gregwar') {
        spl_autoload_register(function($name) use ($src) {
            if (str_starts_with($name, 'Gregwar\\')) require $src.'/src/'.str_replace('\\','/',$name).'.php';
        });
        class GregwarPilot extends \Gregwar\Captcha\CaptchaBuilder {
            public int $lineCalls=0;
            protected function drawLine(GdImage $im, int $w=150, int $h=40, ?int $col=null): void {
                parent::drawLine($im,$w,$h,$col); $this->lineCalls++;
            }
            protected function writePhrase(GdImage $im, ?string $phrase, string $font, int $w, int $h): ?int {
                snapshot($im,'01_before_text');
                $result=parent::writePhrase($im,$phrase,$font,$w,$h);
                snapshot($im,'02_after_text'); return $result;
            }
        }
        $b=new GregwarPilot($cfg['label']);
        $b->setBackgroundColor(245,245,245)->setTextColor(30,40,65)->setScatterEffect(false)
          ->setMaxAngle(0)->setMaxOffset(0)->setMaxBehindLines(0)
          ->setMaxFrontLines(str_contains($mode,'lines') ? 3 : 0)->setDistortion(str_contains($mode,'warp'));
        $b->build($cfg['width'],$cfg['height'],$cfg['font']);
        imagepng($b->getGd(),$cfg['output']);
        $meta['actual']=['label'=>$b->getPhrase(),'line_calls'=>$b->lineCalls,'fingerprint'=>$b->getFingerprint(),
                         'post_effects'=>'disabled by explicit colors', 'line_width'=>'random 1..3 pixels, not independently configurable'];
    } elseif ($cfg['backend']==='securimage') {
        require $src.'/securimage.php';
        class SecurimagePilot extends Securimage {
            public function runPilot() { $this->doImage(); }
            protected function drawWord() {
                snapshot($this->im,'01_before_text'); parent::drawWord();
                snapshot($this->perturbation>0 ? $this->tmpimg : $this->im,'02_text_canvas');
            }
            protected function output() { global $cfg; imagepng($this->im,$cfg['output']); }
        }
        $b=new SecurimagePilot(['no_session'=>true,'no_exit'=>true,'send_headers'=>false,
           'display_value'=>$cfg['label'],'ttf_file'=>$cfg['font'],
           'image_width'=>$cfg['width'],'image_height'=>$cfg['height'],
           'perturbation'=>str_contains($mode,'warp') ? 0.65 : 0,
           'noise_level'=>str_contains($mode,'dots') ? 2 : 0,
           'num_lines'=>str_contains($mode,'curves') ? 3 : 0,
           'use_random_baseline'=>false,'use_random_boxes'=>false,'use_random_spaces'=>false,'use_text_angles'=>false]);
        $b->runPilot();
        $meta['actual']=['label'=>$cfg['label'],'curves_requested'=>$b->num_lines,'noise_level'=>$b->noise_level,
                         'random_choices'=>'seed only; per-draw choices not instrumented'];
    } elseif ($cfg['backend']==='mobicms') {
        if ($cfg['capture_random_tape'] ?? false) {
            require __DIR__.'/random_tape.php';
            $randomTape=[]; $replayIndex=0;
            $replayTape=isset($cfg['random_tape_replay']) ? json_decode(file_get_contents($cfg['random_tape_replay']),true,512,JSON_THROW_ON_ERROR) : null;
        }
        spl_autoload_register(function($name) use ($src) {
            $prefix='Mobicms\\Captcha\\';
            if (str_starts_with($name,$prefix)) require $src.'/src/'.str_replace('\\','/',substr($name,strlen($prefix))).'.php';
        });
        $b=new \Mobicms\Captcha\Image($cfg['label']);
        $b->fontFolders=[$cfg['font_dir']]; $b->fontsTune=[];
        $b->fontMix=($mode==='mixed_fonts'); $b->imageWidth=$cfg['width']; $b->imageHeight=$cfg['height'];
        $b->defaultFontSize=$cfg['font_size'];
        file_put_contents($cfg['output'],$b->build());
        $meta['actual']=['label'=>$b->getCode(),'random_choices'=>'crypto random_int; seed cannot replay',
                         'font_case_tuning'=>'disabled explicitly; generic OFL font folder'];
        $meta['output_type']='RGBA_complete_image';
        if ($cfg['capture_random_tape'] ?? false) {
            if ($replayTape!==null && $replayIndex!==count($replayTape)) throw new RuntimeException('Unconsumed random tape');
            $meta['actual']['random_choices']=$randomTape;
            file_put_contents($cfg['output'].'.random.json',json_encode($randomTape,JSON_THROW_ON_ERROR));
        }
    } else {
        // Populate an isolated session for the original script's image route.
        // No webserver or network is involved; session files stay in this case.
        session_save_path($cfg['trace_dir']); session_id('pilot'.hash('sha256',$cfg['output'])); session_start();
        $configuration=['code'=>$cfg['label'],'backgrounds'=>[$cfg['background']], 'fonts'=>[$cfg['font']],
          'min_font_size'=>$cfg['font_size'],'max_font_size'=>$cfg['font_size'],'color'=>'#223355',
          'angle_min'=>str_contains($mode,'rotation') ? 8 : 0,'angle_max'=>str_contains($mode,'rotation') ? 8 : 0,
          'shadow'=>str_contains($mode,'shadow'),'shadow_color'=>'#aaaaaa','shadow_offset_x'=>3,'shadow_offset_y'=>3];
        $_SESSION['_CAPTCHA']['config']=serialize($configuration); session_write_close();
        $_GET['_CAPTCHA']=1;
        ob_start(); require $src.'/simple-php-captcha.php'; $png=ob_get_clean();
        file_put_contents($cfg['output'],$png); session_write_close();
        $points=[];
        for($i=0;$i<8;$i+=2) $points[]=[$text_box_size[$i]+$text_pos_x,$text_box_size[$i+1]+$text_pos_y];
        $meta['actual']=['label'=>$captcha_config['code'],'text_bbox_corners'=>$points,
          'angle'=>$angle,'font_size'=>$font_size,'position'=>[$text_pos_x,$text_pos_y],'configuration'=>$configuration];
    }
    $meta['warnings']=$warnings;
    $meta['loaded_files']=array_map(fn($p)=>['path'=>$p,'sha256'=>hash_file('sha256',$p)],get_included_files());
    file_put_contents($cfg['output'].'.json',json_encode($meta,JSON_PRETTY_PRINT|JSON_THROW_ON_ERROR));
} catch (Throwable $e) {
    fwrite(STDERR,json_encode(['error'=>get_class($e).': '.$e->getMessage(),'warnings'=>$warnings])); exit(2);
}
