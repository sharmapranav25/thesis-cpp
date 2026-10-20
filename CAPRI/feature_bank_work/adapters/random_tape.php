<?php
namespace FeatureBank\Pilot;
// Native random_int is used during recording. Replays consume the captured
// values, with exact range/order validation. No cryptographic claim for replays.
function random_int(int $min, int $max): int {
    global $randomTape, $replayTape, $replayIndex;
    if ($replayTape !== null) {
        if (!isset($replayTape[$replayIndex])) throw new \RuntimeException('Random tape exhausted');
        $event=$replayTape[$replayIndex++];
        if ($event['min']!==$min || $event['max']!==$max || $event['value']<$min || $event['value']>$max)
            throw new \RuntimeException('Random tape bounds/order mismatch');
        $value=$event['value'];
    } else $value=\random_int($min,$max);
    $randomTape[]=['min'=>$min,'max'=>$max,'value'=>$value];
    return $value;
}
