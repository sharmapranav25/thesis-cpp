# Feature-bank resource acquisition 01

Date: 2026-09-28 UTC.

**OBSERVED:** Downloaded and extracted five exact-revision source archives. Rechecked all archive hashes and recorded every extracted source file in [source_files.json](../../resources/feature_bank_01/source_files.json). Original archives are retained; extracted sources were not edited or executed.

| Resource | Exact revision | Role |
|---|---|---|
| [Gregwar/Captcha](https://github.com/Gregwar/Captcha/tree/c1e67b3d7ac4f873f785daa366051d7c73d19ca0) | `c1e67b3d7ac4f873f785daa366051d7c73d19ca0` | Candidate rendering resource |
| [dapphp/securimage](https://github.com/dapphp/securimage/tree/864ba7505e1bd36ee73f04d8471d86e1f821d754) | `864ba7505e1bd36ee73f04d8471d86e1f821d754` | Candidate rendering resource |
| [mobicms/captcha](https://github.com/mobicms/captcha/tree/5f75482ee385632ee5977af94d7e11ae4a5b20d1) | `5f75482ee385632ee5977af94d7e11ae4a5b20d1` | Candidate rendering resource |
| [yasirmturk/simple-php-captcha](https://github.com/yasirmturk/simple-php-captcha/tree/338880b4a5e3a6aa23d42e87b1e7243ba9542b8c) | `338880b4a5e3a6aa23d42e87b1e7243ba9542b8c` | Candidate rendering resource |
| [laminas/laminas-captcha](https://github.com/laminas/laminas-captcha/tree/c97c326cefaf4d1731c2c52a53a63381e3ec2270) | `c97c326cefaf4d1731c2c52a53a63381e3ec2270` | Withheld generalization renderer |

**OBSERVED:** Portable PHP 8.5.11 CLI starts successfully with GD enabled. `gd_info()` reports FreeType and PNG support. The runtime archive SHA-256 matches the publisher value recorded in [php_receipt.json](../../resources/feature_bank_01/php_receipt.json). No system installation, PATH change, or shared Python environment change was made.

**READ:** Runtime download URL/checksum came from the [official PHP Windows download page](https://www.php.net/downloads.php?os=windows). This establishes the acquisition reference, not compatibility of the five libraries.

**OBSERVED:** The earlier sandbox network request failed with a socket-permission error. A later authorized acquisition succeeded. The interrupted download request left no resource directory before the successful run.

**PROPOSED / PENDING:** Inspect code and separate font/background licenses, resolve dependency locks, test adapters, verify randomness and text-label integrity, then render and inspect 64 pilot images per renderer. Downloaded does not mean verified, trained, or eligible for the supervised corpus. No classifier or full-generation gate passed in this acquisition step.

These are generic resources; none is identified as the original generator of the target datasets. Shared ancestry with target data remains unknown. Laminas remains withheld from training.

## Existing stage-1 work

**OBSERVED:** The separate [selection manifest](../../runs/feature_bank_01/manifest.json) contains 40 images from each historical validation partition, selected without classifier predictions. All 6,708 historical unique images matched saved file/pixel hashes and dimensions; exclusion checks and partition checks passed. Ten contact sheets were opened for assistant visual review. Feature-label drafts and user auditing remain pending; no audited real-image scoring is available.

## Evidence

- [Archive revisions and hashes](../../resources/feature_bank_01/acquisition.json)
- [Source file hashes](../../resources/feature_bank_01/source_files.json)
- [Runtime receipt](../../resources/feature_bank_01/php_receipt.json)
- [Acquisition script](../../scripts/acquire_feature_bank.ps1)
- [Receipt verification script](../../scripts/record_acquisition.py)
