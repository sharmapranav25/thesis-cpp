<?php
// Wrapper that loads cool-php-captcha's SimpleCaptcha class without its
// top-level side effects (session_start + immediate CreateImage call).
//
// We accomplish this by reading captcha.php and stripping the lines outside
// the class, then eval'ing the remainder. The result is an extended class
// `CoolCaptchaWrapper` that:
//   1. Lets us inject `forced_text` (so the corpus uses our random label).
//   2. Captures the JPEG/PNG bytes via stdout buffering instead of HTTP headers.
//
// We only do this once per process.

if (!class_exists('SimpleCaptcha', false)) {
    $src = file_get_contents(__DIR__ . '/../vendor/coolcaptcha/captcha.php');
    if ($src === false) throw new RuntimeException('cool-php-captcha source missing');
    // Drop everything before the SimpleCaptcha class declaration
    $pos = strpos($src, 'class SimpleCaptcha');
    if ($pos === false) throw new RuntimeException('SimpleCaptcha class not found in vendor file');
    $code = substr($src, $pos);
    // Strip the leading <?php tag is not present in the substring; we eval as PHP fragment.
    eval($code);
}

class CoolCaptchaWrapper extends SimpleCaptcha {
    public string $forced_text = '';

    /**
     * Override of GetCaptchaText: use our forced label if provided,
     * otherwise fall back to the parent's random/dictionary behavior.
     */
    protected function GetCaptchaText() {
        if ($this->forced_text !== '') return $this->forced_text;
        // No dictionary if wordsFile is empty
        if (empty($this->wordsFile)) {
            return $this->GetRandomCaptchaText();
        }
        return parent::GetCaptchaText();
    }

    /**
     * Override WriteImage to skip HTTP headers (we capture bytes via ob_*).
     */
    protected function WriteImage() {
        if ($this->imageFormat === 'png' && function_exists('imagepng')) {
            imagepng($this->im);
        } else {
            imagejpeg($this->im, null, 80);
        }
    }
}
