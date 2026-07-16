<?php
require __DIR__ . '/lib_common.php';
$f = font_pool();
echo count($f) . " fonts available\n";
foreach (array_slice($f, 0, 6) as $p) echo "  $p\n";
echo "...\n";
foreach (array_slice($f, -6) as $p) echo "  $p\n";
